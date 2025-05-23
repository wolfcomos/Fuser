// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <id_model/id_model.h>
#include <ir/interface_nodes.h>
#include <scheduler/tools/maxinfo_propagator.h>
#include <transform_replay.h>
#include <visibility.h>

#include <memory>
#include <unordered_set>

namespace nvfuser {

class MaxPosCalculator {
  // Root domains in producer that's unmappable to any of its consumers
  std::unordered_set<IterDomain*> unmappable_dims_;

  // User set IterDomains to not inline
  std::unordered_set<IterDomain*> uninlinable_ids_;

  std::unique_ptr<IdModel> id_model_;

  // Iterate through all TVs and collect the dimensions of each TV that don't
  // map to all its consumer TVs.
  void buildUnmappableDims(bool compute_at_only);

  // Get the IdModel graph for inlining analysis (i.e., the Broadcast
  // graph). The graph is lazily created.
  const ValGraph& inliningGraph();

 public:
  // Utility function to return if an id of tv is a valid iter domain to inline
  // within. This is used in getMaxPos{PasC,CasP}. Different variations of the
  // bool values are used if checking max position of PasC, CasP, or checking
  // for a max "self" position.
  bool isAllowedID(
      IterDomain* id,
      TensorView* tv,
      bool best_effort,
      bool allow_reduction,
      bool allow_vectorize,
      bool allow_unmappable) const;

  // Returns the position at which tv can be inlined within.
  size_t getMaxPosSelf(
      TensorView* tv,
      bool best_effort,
      bool allow_reduction,
      bool allow_vectorize,
      bool allow_unmappable) const;

  // Returns the maximum position producer can be inlined based on consumer
  // given the set ComputeAtMode
  size_t getMaxProducerPosFromConsumer(
      TensorView* producer,
      TensorView* consumer,
      bool best_effort);

  // Checks producers, consumers, and siblings to see what the maximum position
  // in tv is that can be shared across both directions.
  size_t getMaxPosAll(
      TensorView* tv,
      bool best_effort = false,
      bool check_siblings = true);

  int64_t getConsumerPosAlignedToProducerCA(
      TensorView* consumer,
      TensorView* producer,
      int64_t producer_pos);

  MaxPosCalculator(
      std::unordered_set<IterDomain*> uninlinable_ids = {},
      bool compute_at_only = false);
};

// Inline to the right most allowed position for all tensors in the current
// fusion.
NVF_API void inlineMost(
    const std::unordered_set<IterDomain*>& uninlinable_ids = {});
// Inline to the right most allowed position for the selected tensors in the
// current fusion.
NVF_API void inlineMost(
    const std::vector<TensorView*>& tvs,
    const std::unordered_set<IterDomain*>& uninlinable_ids = {});
// Inline to the right most allowed position for the selected tensors in the
// current fusion.
void inlineMost(
    const std::unordered_set<TensorView*>& tvs,
    const std::unordered_set<IterDomain*>& uninlinable_ids = {});

// Inline to the position corresponding to the reference position in the
// reference tensor for all tensors in the current fusion.
NVF_API void inlineAllAt(
    TensorView* reference_tv,
    int64_t reference_pos,
    bool best_effort = false,
    const std::unordered_set<IterDomain*>& uninlinable_ids = {});

// Inline to the position corresponding to the reference position in the
// reference tensor for selected tensors in the current fusion.
NVF_API void inlineSelectedAt(
    const std::unordered_set<TensorView*>& selected,
    TensorView* reference_tv,
    int64_t reference_pos,
    bool best_effort = false,
    const std::unordered_set<IterDomain*>& uninlinable_ids = {});

} // namespace nvfuser
