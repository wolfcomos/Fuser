// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/utils.h>
#include <iter_visitor.h>
#include <logical_domain_map.h>

#include <device_lower/analysis/trivial_broadcast.h>

namespace nvfuser {

ConcretizedBroadcastDomains::ConcretizedBroadcastDomains(Fusion* fusion) {
  exact_map_ = std::make_unique<ExactLogicalDomainMap>(fusion);

  // Initialize the origin map with input broadcast domains
  auto inputs = fusion->inputsAndCreated();
  for (const auto fusion_input_tv :
       ir_utils::filterByType<TensorView>(inputs)) {
    for (auto logical_id : fusion_input_tv->getLogicalDomain()) {
      if (logical_id->isBroadcast()) {
        broadcast_origin_map_.emplace(
            logical_id, std::unordered_set<IterDomain*>({logical_id}));
      }
    }
  }
  traverse(fusion);
}

bool ConcretizedBroadcastDomains::isConcretized(IterDomain* id) const {
  return !allConcretizedDomains(id).empty();
}

bool ConcretizedBroadcastDomains::isUniquelyConcretized(IterDomain* id) const {
  return allConcretizedDomains(id).size() == 1;
}

bool ConcretizedBroadcastDomains::maybeNonUniquelyConcretized(
    IterDomain* id) const {
  return allConcretizedDomains(id).size() > 1;
}

std::unordered_set<IterDomain*> ConcretizedBroadcastDomains::
    allConcretizedDomains(IterDomain* id) const {
  auto it = broadcast_to_concrete_map_.find(id);
  if (it != broadcast_to_concrete_map_.end()) {
    return it->second;
  }
  return {};
}

// In some cases an op like pad or slice will introduce a broadcast domain by
// truncating a longer dimension or expanding an empty dimension to size 1. In
// these cases tv will have logical Broadcast IterDomains that are not present
// in the root domain. Contrast this with BroadcastOp, whose output does not
// have logical domains and instead places new broadcast domains in the output
// root domain.
void ConcretizedBroadcastDomains::handle(TensorView* tv) {
  if (!tv->hasRoot()) {
    return;
  }
  for (auto id : tv->getLogicalDomain()) {
    // Register broadcast logical domains that are not root domains as new
    // broadcast origins.
    if (id->isBroadcast() &&
        std::find(tv->getRootDomain().begin(), tv->getRootDomain().end(), id) ==
            tv->getRootDomain().end()) {
      broadcast_origin_map_.emplace(id, std::unordered_set<IterDomain*>({id}));
    }
  }
}

// Most broadcasts are handled with this method, since Broadcast domains are
// usually introduced through a BroadcastOp. Others are handled by the
// handle(TensorView*) method.
void ConcretizedBroadcastDomains::handle(BroadcastOp* bop) {
  // Create a new entry for each of new broadcast domains
  auto out = bop->out()->as<TensorView>();
  for (const auto i : arange(out->getLogicalDomain().size())) {
    if (bop->getBroadcastDimFlags().at(i)) {
      auto new_bcast_id = out->getLogicalDomain().at(i);
      broadcast_origin_map_.emplace(
          new_bcast_id, std::unordered_set<IterDomain*>({new_bcast_id}));
    }
  }
}

void ConcretizedBroadcastDomains::dispatch(Expr* expr) {
  IterVisitor::dispatch(expr);

  // Propagate broadcast origin info from producers to consumers
  for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
    std::unordered_set<IterDomain*> producer_broadcasts;
    // This assumes there's no merged broadcast axes between root and rfactor
    // domains which is not possible at the moment. If this assumption is ever
    // invalidated we would need to manaually propagate root IDs to rfactor IDs.
    for (auto producer_id : producer->getLogicalDomain()) {
      if (producer_id->isBroadcast()) {
        producer_broadcasts.insert(producer_id);
      }
    }
    if (producer_broadcasts.empty()) {
      continue;
    }

    for (auto consumer : ir_utils::filterByType<TensorView>(expr->outputs())) {
      auto p2c_map = PairwiseLogicalDomainMap(producer, consumer)
                         .mapProducerToConsumer(&producer_broadcasts);
      for (const auto& kv : p2c_map) {
        auto p_id = kv.first;
        auto c_id = kv.second;
        // If the consumer ID is a reduction (i.e., a trivial
        // reduction), do not consider it's concretized.
        const bool is_concretized =
            !c_id->isBroadcast() && !c_id->isReduction();
        auto it = broadcast_origin_map_.find(p_id);
        NVF_ERROR(
            it != broadcast_origin_map_.end(),
            "Broadcast origin info not found for producer broadcast domain: ",
            p_id->toString(),
            " of ",
            producer->toString());
        const auto& producer_origins = it->second;
        if (is_concretized) {
          // Keep track of all the origin domains as concretized
          for (auto origin : producer_origins) {
            markAsConcretized(origin, c_id);
          }
        } else {
          // Not concretized yet. Propagate forward the origin info.
          auto& consumer_origins = broadcast_origin_map_[c_id];
          for (auto origin : producer_origins) {
            consumer_origins.insert(origin);
          }
          consumer_origins.insert(c_id);
        }
      }
    }
  }
}

void ConcretizedBroadcastDomains::markAsConcretized(
    IterDomain* broadcast_root_domain,
    IterDomain* concrete_root_domain) {
  std::deque<IterDomain*> child_domains({broadcast_root_domain});
  while (!child_domains.empty()) {
    auto child = child_domains.front();
    child_domains.pop_front();
    auto& concrete_ids = broadcast_to_concrete_map_[child];
    auto inserted =
        insertRootDomainToConcreteDomainSet(concrete_root_domain, concrete_ids);
    if (!inserted) {
      continue;
    }
    const auto& child_uses = child->uses();
    for (auto child_use : child_uses) {
      for (auto out_id :
           ir_utils::filterByType<IterDomain>(child_use->outputs())) {
        child_domains.push_back(out_id);
      }
    }
  }
}

bool ConcretizedBroadcastDomains::insertRootDomainToConcreteDomainSet(
    IterDomain* new_root_id,
    std::unordered_set<IterDomain*>& id_set) {
  auto has_exactly_mapped_id =
      std::any_of(id_set.begin(), id_set.end(), [&](IterDomain* existing_id) {
        return exact_map_->areMapped(new_root_id, existing_id);
      });
  if (has_exactly_mapped_id) {
    return false;
  } else {
    id_set.emplace(new_root_id);
    return true;
  }
}

} // namespace nvfuser
