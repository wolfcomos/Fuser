// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <parallel_dimension_map.h>

#include <ATen/cuda/CUDAContext.h>
#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <expr_simplifier.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <scheduler/utils.h>

#include <functional>
#include <sstream>
#include <string>
#include <utility>

using PAndID = std::pair<nvfuser::ParallelType, nvfuser::IterDomain*>;

namespace std {

template <>
struct hash<PAndID> {
  std::size_t operator()(const PAndID& data) const noexcept {
    size_t ptype = static_cast<size_t>(data.first);
    size_t address = reinterpret_cast<size_t>(data.second);
    size_t combined = (address << 8) | ptype;
    return std::hash<size_t>()(combined);
  }
};

} // namespace std

namespace nvfuser {

void ParallelDimensionMap::build(Fusion* fusion) {
  VectorOfUniqueEntries<PAndID> all_concrete_ids;
  auto all_vals = fusion->producedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(all_vals)) {
    if (tv->isCircularBuffered() &&
        std::holds_alternative<WarpSpecialized>(
            tv->circularBufferOptions().type)) {
      const auto& warp_specialized =
          std::get<WarpSpecialized>(tv->circularBufferOptions().type);
      warp_specialized_types_.insert(warp_specialized.on);
      if (warp_specialized.num_registers.has_value()) {
        ws_with_register_sharing_pt_ = warp_specialized.on;
      }
    }
    for (auto id : tv->domain()->allIDs()) {
      auto ptype = id->getParallelType();
      if (!isParallelTypeThread(ptype)) {
        continue;
      }
      auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
          id, IdMappingMode::EXACT);
      if (concrete_id->isBroadcast()) {
        // Broadcasted concrete id's don't specify anything about shape
        continue;
      }
      all_concrete_ids.pushBack(std::make_pair(ptype, concrete_id));
    }
  }

  // Scan all TVs to build dim_map_
  for (auto [ptype, concrete_id] : all_concrete_ids) {
    exact_types_.insert(ptype); // insert now and cleanup later
    if (dim_map_.count(ptype) == 0) {
      dim_map_[ptype] = concrete_id->extent();
    } else {
      dim_map_.at(ptype) = SimplifyingIrBuilder::maxExpr(
          dim_map_.at(ptype), concrete_id->extent());
    }
  }

  // Simplify dim_map_
  for (auto& [k, v] : dim_map_) {
    v = simplifyExpr(v);
  }

  // Compute exact_types_
  for (auto [ptype, concrete_id] : all_concrete_ids) {
    auto expr_val = simplifyExpr(SimplifyingIrBuilder::eqExpr(
                                     dim_map_.at(ptype), concrete_id->extent()))
                        ->value();
    if (!expr_val.hasValue() || !expr_val.as<bool>()) {
      exact_types_.erase(ptype);
    }
  }

  adjustMappingsForWarpPadding();
  adjustMappingsForWarpSpecialization();
}

void ParallelDimensionMap::adjustMappingsForWarpPadding() {
  const auto gpu_lower = GpuLower::current();

  // If TIDx is padded to a multiple of the warp size, mark it as
  // non-exact.

  auto& warp_info = gpu_lower->getWarpPaddedParallelInfo();
  // TIDx isn't really padded if there isn't a warp reduction (this could
  // change)
  if (!(warp_info.is_tidx_padded && warp_info.has_warp_reduction)) {
    return;
  }

  const auto tidx_pt = ParallelType::TIDx;
  auto warp_size_val = IrBuilder::create<Val>(32L, DataType::Index);
  auto tidx_dim = getRaw(tidx_pt);

  NVF_ERROR(tidx_dim != nullptr);

  // If tidx is strictly defined as blockDim.x then it must be set to a
  // multiple of the warp, there is nothing to do
  if (tidx_dim->sameAs(NamedScalar::getParallelDim(tidx_pt))) {
    return;
  }

  auto expr_val =
      simplifyExpr(SimplifyingIrBuilder::eqExpr(
                       SimplifyingIrBuilder::modExpr(tidx_dim, warp_size_val),
                       tidx_dim->container()->zeroVal()))
          ->value();

  // If already multiple of warp, nothing to do
  if (expr_val.is<bool>() && expr_val.as<bool>()) {
    return;
  }

  // TIDx is padded to a multiple of warp. If it's known to be a
  // single warp, use the constant warp size as the dimension of
  // TIDx. Otherwise, just use blockDim.x.
  if (warp_info.is_tidx_single_warp) {
    dim_map_.at(ParallelType::TIDx) = warp_size_val;
  } else {
    dim_map_.at(ParallelType::TIDx) =
        simplifyExpr(SimplifyingIrBuilder::mulExpr(
            SimplifyingIrBuilder::ceilDivExpr(tidx_dim, warp_size_val),
            warp_size_val));
  }

  // TIDx is no longer exact
  exact_types_.erase(ParallelType::TIDx);
}

void ParallelDimensionMap::adjustMappingsForWarpSpecialization() {
  // shortcut for case without register sharing
  if (!ws_with_register_sharing_pt_.has_value()) {
    for (auto pt : warp_specialized_types_) {
      auto dim_it = dim_map_.find(pt);
      if (dim_it == dim_map_.end()) {
        dim_map_[pt] = IrBuilder::create<Val>(2, DataType::Index);
      } else {
        // Intentionally not using SimplifyingIrBuilder::addExpr here so that
        // we still have access to the pointer to the original IR node.
        // We need the pointer to the original IR node because we want
        // getRawCompute to be callable in an environment without FusionGuard,
        // that is, when the IR container is read-only. In such an environment,
        // we can't create new IR nodes for (x - 1). By using
        // IrBuilder::addExpr, we can always create IR nodes like addExpr(x, 1),
        // and SimplifyingIrBuilder::addExpr in getRawCompute will be able to
        // simplify find the x when we do addExpr(addExpr(x, 1) - 1).
        dim_map_[pt] = IrBuilder::addExpr(
            dim_it->second, dim_it->second->fusion()->oneVal());
      }
      exact_types_.erase(pt);
    }
    return;
  }
  // For register sharing, require contiguous 128 threads calling the same
  // setreg instruction.
  // Not used: 1, Const: n, Dynamic: -1
  auto get_threads_count_in_dim = [&](ParallelType pt) {
    if (!dim_map_.contains(pt)) {
      return 1L;
    }
    if (dim_map_.at(pt)->isConstScalar()) {
      return dim_map_.at(pt)->value().as<int64_t>();
    }
    // Return -1 for dynamic dimensions, this disables register sharing on
    // dynamic dimensions since we can't guarantee the number of threads is
    // divisible by 128. We may allow this in the future and delegate this
    // check to a point where the launch parameters are known.
    return -1L;
  };
  // Warp specialization with register sharing on parallel type pt
  // index = TIDx + TIDy * bdimx + TIDz * bdimx * bdimy
  auto pt = ws_with_register_sharing_pt_.value();
  auto dim_it = dim_map_.find(pt);
  int64_t pad_n_threads = 0;
  int64_t after_pad = 0;

  // switch is not used to avoid explicitly handle all parallel types
  if (pt == ParallelType::TIDx) {
    // If on TIDx, pad by 128
    pad_n_threads = 128;
    after_pad = get_threads_count_in_dim(pt) + pad_n_threads;
    NVF_ERROR(
        after_pad % 128 == 0,
        "Illegal register sharing on TIDx, bdimx = ",
        after_pad);
  } else if (pt == ParallelType::TIDy) {
    // If on TIDy, pad by 128 / bdimx
    int64_t bdimx = get_threads_count_in_dim(ParallelType::TIDx);
    pad_n_threads = scheduler_utils::safeDiv(128, bdimx);
    after_pad = get_threads_count_in_dim(pt) + pad_n_threads;
    NVF_ERROR(
        (after_pad * bdimx) % 128 == 0,
        "Illegal register sharing on TIDy, bdimx = ",
        bdimx,
        ", bdimy = ",
        after_pad);
  } else if (pt == ParallelType::TIDz) {
    // If on TIDz, pad by 128 / (bdimx * bdimy)
    int64_t bdimx = get_threads_count_in_dim(ParallelType::TIDx);
    int64_t bdimy = get_threads_count_in_dim(ParallelType::TIDy);
    pad_n_threads = scheduler_utils::safeDiv(128, bdimx * bdimy);
    after_pad = get_threads_count_in_dim(pt) + pad_n_threads;
    NVF_ERROR(
        (after_pad * bdimx * bdimy) % 128 == 0,
        "Illegal register sharing on TIDz, bdimx = ",
        bdimx,
        ", bdimy = ",
        bdimy,
        ", bdimz = ",
        after_pad);
  } else {
    NVF_THROW("Unsupported parallel type for register sharing: ", pt);
  }

  // Apply the pad
  ws_with_register_sharing_pad_val_ = pad_n_threads;
  auto off_set = IrBuilder::create<Val>(pad_n_threads, DataType::Index);
  auto current_val = dim_it == dim_map_.end()
      ? IrBuilder::create<Val>(1, DataType::Index)
      : dim_it->second;
  dim_map_[pt] = IrBuilder::addExpr(current_val, off_set);
  exact_types_.erase(pt);
}

Val* ParallelDimensionMap::getRaw(ParallelType pt) const {
  NVF_ERROR(isParallelTypeThread(pt), "Invalid ParallelType: ", pt);
  auto it = dim_map_.find(pt);
  if (it == dim_map_.end()) {
    return nullptr;
  } else {
    return it->second;
  }
}

Val* ParallelDimensionMap::get(ParallelType pt) const {
  auto raw = getRaw(pt);
  if (raw != nullptr && !raw->isConstInt()) {
    return NamedScalar::getParallelDim(pt);
  }
  return raw;
}

bool ParallelDimensionMap::isExact(ParallelType pt) const {
  return exact_types_.find(pt) != exact_types_.end();
}

Val* ParallelDimensionMap::getRawCompute(ParallelType pt) const {
  Val* raw = getRaw(pt);
  if (warp_specialized_types_.count(pt)) {
    int64_t padded_val = getWarpSpecializationPaddedVal(pt);
    return SimplifyingIrBuilder::addExpr(raw, -padded_val);
  }
  return raw;
}

Val* ParallelDimensionMap::getRawLoad(ParallelType pt) const {
  if (warp_specialized_types_.count(pt)) {
    return IrBuilder::create<Val>(
        getWarpSpecializationPaddedVal(pt), DataType::Index);
  }
  return getRaw(pt);
}

Val* ParallelDimensionMap::getNumComputeThreadsEachBlock() const {
  Val* num_threads = FusionGuard::getCurFusion()->oneVal();
  for (auto pt : kParallelTypeTIDs) {
    auto dim = getRawCompute(pt);
    if (dim == nullptr) {
      continue;
    }
    num_threads = SimplifyingIrBuilder::mulExpr(num_threads, dim);
  }
  return num_threads;
}

int64_t ParallelDimensionMap::getWarpSpecializationPaddedVal(
    ParallelType pt) const {
  NVF_ERROR(
      warp_specialized_types_.contains(pt), "Can't find ParallelType: ", pt);
  if (!ws_with_register_sharing_pt_.has_value()) {
    return 1;
  }
  NVF_ERROR(
      ws_with_register_sharing_pt_.value() == pt,
      "Can't find padded val for: ",
      pt);
  return ws_with_register_sharing_pad_val_.value();
}

std::string ParallelDimensionMap::toString() const {
  std::stringstream ss;
  for (auto pt : kParallelTypeThreads) {
    ss << pt << ": ";
    auto dim = getRaw(pt);
    if (dim != nullptr) {
      ss << dim->toInlineString();
      if (isExact(pt)) {
        ss << ", exact";
      } else {
        ss << ", non-exact";
      }
    } else {
      ss << "unused";
    }
    ss << "\n";
  }

  return ss.str();
}

} // namespace nvfuser
