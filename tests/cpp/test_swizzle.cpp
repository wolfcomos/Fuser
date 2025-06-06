// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <device_lower/analysis/bank_conflict.h>
#include <ops/all_ops.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/tools/abstract_tensor.h>
#include <scheduler/tools/inlining.h>
#include <swizzle.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <transform_iter.h>

namespace nvfuser {

// TODO: migrate these tests to new swizzle
class LegacySwizzleTest : public NVFuserTest {};
class SwizzleTest : public NVFuserTest {};

// Test a basic swizzle pattern
TEST_F(LegacySwizzleTest, SimpleSwizzle0) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv2);

  // Make a 2x8 Zshape tile
  tv1->split(-1, 16);
  tv1->split(-1, 8);
  // [O, 2, 8]

  tv2->split(-1, 16);
  tv2->split(-1, 4);
  //[O, 4, 4]

  tv1->computeAt(tv2, 1);
  tv1->swizzle(Swizzle2DType::ZShape, -2, -1);

  GpuLower gpulw(&fusion);
  auto exprs = gpulw.run()->topLevelExprs();
  auto str = ir_utils::toString(exprs);
  NVF_CHECK(str.find("where") != std::string::npos);

  KernelExecutor ke;
  ke.compile(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 32}, options);
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Test swizzle inlining
TEST_F(LegacySwizzleTest, SimpleSwizzle1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv3);

  // Make a 2x8 Zshape tile
  tv2->split(-1, 16);
  tv2->split(-1, 8);
  // [O, 2, 8]

  tv3->split(-1, 16);
  tv3->split(-1, 4);
  //[O, 4, 4]

  tv2->computeAt(tv3, 1);
  tv2->swizzle(Swizzle2DType::ZShape, -2, -1);

  // Inlining a producer into a swizzled consumer is ok
  tv1->computeAt(tv2, -1);

  KernelExecutor ke;
  ke.compile(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 32}, options);
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Test sync insertion and memory check in parallelized swizzles.
//  In this test, data is parallel written into smem in zcurve
//   pattern and then read out and output to global mem unswizzled.
TEST_F(LegacySwizzleTest, SimpleSwizzle2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv2);

  tv1->swizzle(Swizzle2DType::ZShape, -2, -1);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDy);

  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDy);

  // Validation should fail since TV1 is not in shared
  //  memory as required by sync info pass.
  ASSERT_ANY_THROW(GpuLower(&fusion).run());

  tv1->setMemoryType(MemoryType::Shared);

  // Make sure that a sync is inserted:
  bool sync_found = false;
  GpuLower gpulw(&fusion);
  auto flattened_exps =
      ir_utils::flattenScopedExprs(gpulw.run()->topLevelExprs());

  for (auto expr : flattened_exps) {
    if (expr->isA<kir::BlockSync>()) {
      sync_found = true;
    }
    // Will require a sync thread before any shared memory read.
    for (auto inp_tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
      if (inp_tv->getMemoryType() == MemoryType::Shared) {
        NVF_ERROR(sync_found, "Block sync required but not inserted");
      }
    }
  }

  KernelExecutor ke;
  ke.compile(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32, 32}, options);
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Test BestEffortReplay behavior with swizzle op
TEST_F(LegacySwizzleTest, SwizzleMapping) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv3);

  // Make a 2x8 Zshape tile
  tv2->split(-1, 16);
  tv2->split(-1, 8);
  // [O, 2, 8]

  tv3->split(-1, 16);
  tv3->split(-1, 4);
  //[O, 4, 4]

  tv2->computeAt(tv3, 1);
  tv2->swizzle(Swizzle2DType::ZShape, -2, -1, SwizzleMode::Loop);

  // Inlining a producer into a swizzled consumer is ok
  tv1->computeAt(tv2, -1);

  // Check BestEffortReplay behavior with skip swizzles option on.
  PairwiseLogicalDomainMap logical_map(tv1, tv2);

  // Check producer to consumer map,
  //  i.e. unswizzled tensor to swizzled tensor map
  //----------------------------------------------------------
  auto p2c_disjoint_id_map =
      BestEffortReplay::replayCasP(tv2, tv1, -1, logical_map)
          .getIterDomainEquivalence();
  // P2C map should exist and both the x and y map should
  //  map to the output of the swizzle op.
  NVF_ERROR(
      p2c_disjoint_id_map.mappingExists(tv1->axis(-2)) &&
      p2c_disjoint_id_map.mappingExists(tv1->axis(-1)));

  NVF_ERROR(
      p2c_disjoint_id_map.strictAreMapped(tv1->axis(-2), tv2->axis(-2)) &&
      p2c_disjoint_id_map.strictAreMapped(tv1->axis(-1), tv2->axis(-1)));

  // Check consumer to producer map,
  //  i.e. swizzled tensor to unswizzled tensor map
  //----------------------------------------------------------
  auto c2p_disjoint_id_map =
      BestEffortReplay::replayPasC(tv1, tv2, -1, logical_map)
          .getIterDomainEquivalence();

  auto swizzle_op = tv2->axis(-1)->definition()->as<Swizzle2D>();

  // Input of swizzle ops will not be mapped to any
  //  by BestEffortReplay, as BestEffortReplay has to be
  //  one to one. IdGraph will further map them together.
  NVF_ERROR(
      !p2c_disjoint_id_map.mappingExists(swizzle_op->inX()) &&
      !p2c_disjoint_id_map.mappingExists(swizzle_op->inY()));

  // Mapping for swizzle outputs should be mapped and should
  //  also map to the corresponding axes on the unswizzled tensor.
  NVF_ERROR(
      p2c_disjoint_id_map.mappingExists(swizzle_op->outX()) &&
      p2c_disjoint_id_map.mappingExists(swizzle_op->outY()));

  NVF_ERROR(
      p2c_disjoint_id_map.strictAreMapped(swizzle_op->outX(), tv1->axis(-2)) &&
      p2c_disjoint_id_map.strictAreMapped(swizzle_op->outY(), tv1->axis(-1)));

  // Check id graph behavior
  //----------------------------------------------------------
  ComputeAtMap ca_map(&fusion);
  // Corresponding inputs and outputs of swizzle ops are
  //  map through by exact and permissive map.
  NVF_ERROR(
      ca_map.areMapped(tv1->axis(-2), swizzle_op->inX(), IdMappingMode::EXACT));
  NVF_ERROR(
      ca_map.areMapped(tv1->axis(-1), swizzle_op->inY(), IdMappingMode::EXACT));
  NVF_ERROR(ca_map.areMapped(
      tv1->axis(-2), swizzle_op->outX(), IdMappingMode::EXACT));
  NVF_ERROR(ca_map.areMapped(
      tv1->axis(-1), swizzle_op->outY(), IdMappingMode::EXACT));

  NVF_ERROR(ca_map.areMapped(
      tv1->axis(-2), swizzle_op->inX(), IdMappingMode::PERMISSIVE));
  NVF_ERROR(ca_map.areMapped(
      tv1->axis(-1), swizzle_op->inY(), IdMappingMode::PERMISSIVE));
  NVF_ERROR(ca_map.areMapped(
      tv1->axis(-2), swizzle_op->outX(), IdMappingMode::PERMISSIVE));
  NVF_ERROR(ca_map.areMapped(
      tv1->axis(-1), swizzle_op->outY(), IdMappingMode::PERMISSIVE));
}

// Test a basic loop swizzle pattern
TEST_F(LegacySwizzleTest, LoopSwizzle0) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv2);

  tv2->split(-1, 16);
  tv2->split(-1, 4);
  //[O, 4, 4]

  tv2->swizzle(Swizzle2DType::ZShape, -2, -1, SwizzleMode::Loop);

  tv0->computeAt(tv2, -1);

  KernelExecutor ke;
  ke.compile(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 32}, options);
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Outer block zshape pattern
TEST_F(LegacySwizzleTest, LoopSwizzle1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv2);

  tv2->split(-2, 8);
  tv2->split(-1, 4);
  //[I0o, I0i, I1o, I1i]
  tv2->reorder({{1, 2}, {2, 1}});
  //[I0o, I1o, I0i, I1i]

  tv2->swizzle(Swizzle2DType::ZShape, 0, 1, SwizzleMode::Loop);
  tv0->computeAt(tv2, -1);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::BIDy);

  KernelExecutor ke;
  ke.compile(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({45, 77}, options);
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Test assertion in unsupported pattern: non-leaf loop swizzle.
TEST_F(LegacySwizzleTest, LoopSwizzleCheck0) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv2);

  tv2->split(-1, 16);
  tv2->split(-1, 4);
  //[O, 4, 4]

  // Swizzle the inner tile.
  tv2->swizzle(Swizzle2DType::ZShape, -2, -1, SwizzleMode::Loop);

  // Make swizzle output not a loop domain.
  tv2->merge(-2);

  tv0->computeAt(tv2, -1);

  KernelExecutor ke;
  ASSERT_ANY_THROW(ke.compile(&fusion));
}

// Test assertion in unsupported pattern: half-inlined loop swizzle.
TEST_F(LegacySwizzleTest, LoopSwizzleCheck1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv3);

  //[O, 4, 4]
  tv2->split(-1, 16);
  tv2->split(-1, 4);

  //[O, 4, 4]
  tv3->split(-1, 16);
  tv3->split(-1, 4);

  // Swizzle inner tile of tv2
  tv2->swizzle(Swizzle2DType::ZShape, -2, -1, SwizzleMode::Loop);

  // Make tv2 swizzled and partially-inlined (unsupported).
  tv0->computeAt(tv3, -2);

  KernelExecutor ke;
  ASSERT_ANY_THROW(ke.compile(&fusion));
}

TEST_F(LegacySwizzleTest, SwizzleVectorize) {
  // When there is a swizzle, non of the involved dimensions are contiguous, so
  // unable to vectorize.
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({4, 4});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->swizzle(Swizzle2DType::XOR, 0, 1);
  tv1->axis(1)->parallelize(ParallelType::Vectorize);

  ASSERT_ANY_THROW(GpuLower(&fusion).run());
}

TEST_F(LegacySwizzleTest, TransposeBankConflictSwizzle1) {
  // Both Xor and CyclicShift swizzling should fully remove bank confliction of
  // a 32x32 non-vectorized transpose.
  std::vector<Swizzle2DType> swizzles{
      Swizzle2DType::XOR, Swizzle2DType::CyclicShift};
  for (auto swizzle_type : swizzles) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeConcreteTensor({32, 32});
    fusion.addInput(tv0);
    auto tv1 = set(tv0);
    auto tv2 = transpose(tv1, 0, 1);
    auto tv3 = set(tv2);
    fusion.addOutput(tv3);

    tv1->setMemoryType(MemoryType::Shared);
    tv1->axis(0)->parallelize(ParallelType::TIDy);
    tv1->axis(1)->parallelize(ParallelType::TIDx);
    tv2->axis(0)->parallelize(ParallelType::TIDy);
    tv2->axis(1)->parallelize(ParallelType::TIDx);
    tv3->axis(0)->parallelize(ParallelType::TIDy);
    tv3->axis(1)->parallelize(ParallelType::TIDx);

    // 32-way bank confliction
    auto bank_conflict_info = fusion.bankConflictInfo();
    ASSERT_EQ(bank_conflict_info.at(tv1).first, std::vector<int64_t>{32});

    // no bank confliction after swizzle
    tv1->swizzle(swizzle_type, 0, 1);
    bank_conflict_info = fusion.bankConflictInfo();
    NVF_CHECK(
        bank_conflict_info.empty(),
        "Expecting no bank conflict after swizzle, but got ",
        bank_conflict_info.size(),
        "bank conflicting expressions.",
        ". Something in our lowering or bank conflict checker must have "
        "changed, ",
        "please update them or this test consistently.");
  }
}

TEST_F(LegacySwizzleTest, TransposeBankConflictSwizzle2) {
  // ZShape should remove half of the bank confliction of a 32x32 non-vectorized
  // transpose.
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 0, 1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(0)->parallelize(ParallelType::TIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDy);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv3->axis(0)->parallelize(ParallelType::TIDy);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  // 32-way bank confliction
  auto bank_conflict_info = fusion.bankConflictInfo();
  ASSERT_EQ(bank_conflict_info.at(tv1).first, std::vector<int64_t>{32});

  // 16-way bank confliction
  tv1->swizzle(Swizzle2DType::ZShape, 0, 1);
  bank_conflict_info = fusion.bankConflictInfo();
  ASSERT_EQ(bank_conflict_info.at(tv1).first, std::vector<int64_t>{16});
}

TEST_F(LegacySwizzleTest, DataSwizzleGlobal) {
  // Data swizzle is ignored in global indexing, so we should just throw an
  // error if someone wants to do so.
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);
  ASSERT_ANY_THROW(tv1->swizzle(Swizzle2DType::XOR, 0, 1));
}

namespace {

// Get the swizzled tensor from input. For example, for ZShape swizzle, if the
// input is
//    1 2 3
//    4 5 6
//    7 8 9
// Then the output will be:
//    1 2 3
//    6 5 4
//    7 8 9
at::Tensor getSwizzledTensor(
    at::Tensor input,
    Swizzle2DType type,
    bool is_unswizzle = false) {
  auto size_x = input.size(0);
  auto size_y = input.size(1);

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  Val* size_x_input = IrBuilder::create<Val>(DataType::Int);
  Val* size_y_input = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(size_x_input);
  fusion.addInput(size_y_input);
  auto x = arange(size_x_input);
  auto xx = broadcast(x, {false, true});
  auto y = arange(size_y_input);
  auto yy = broadcast(y, {true, false});
  std::pair<Val*, Val*> swizzle;
  if (is_unswizzle) {
    swizzle = dispatchUnSwizzle(type, xx, yy, size_x_input, size_y_input);
  } else {
    swizzle = dispatchSwizzle(type, xx, yy, size_x_input, size_y_input);
  }
  fusion.addOutput(swizzle.first);
  fusion.addOutput(swizzle.second);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({size_x, size_y});

  return input.index_put(
      {outputs[0].as<at::Tensor>(), outputs[1].as<at::Tensor>()}, input);
}

} // namespace

TEST_F(LegacySwizzleTest, SwizzleExampleZShape) {
  //    1 2 3      1 2 3
  //    4 5 6  =>  6 5 4
  //    7 8 9      7 8 9
  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto input = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, options);
  auto expect = torch::tensor({{1, 2, 3}, {6, 5, 4}, {7, 8, 9}}, options);
  auto output = getSwizzledTensor(input, Swizzle2DType::ZShape);
  auto unswizzled = getSwizzledTensor(output, Swizzle2DType::ZShape, true);
  NVF_CHECK(at::equal(expect, output));
  NVF_CHECK(at::equal(input, unswizzled));
}

TEST_F(LegacySwizzleTest, SwizzleExampleXor) {
  //    1   2  3  4       1   2   3  4
  //    5   6  7  8       6   5   8  7
  //    9  10 11 12  =>   11  12  9 10
  //    13 14 15 16       16  15 14 13
  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto input = torch::tensor(
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}, options);
  auto expect = torch::tensor(
      {{1, 2, 3, 4}, {6, 5, 8, 7}, {11, 12, 9, 10}, {16, 15, 14, 13}}, options);
  auto output = getSwizzledTensor(input, Swizzle2DType::XOR);
  auto unswizzled = getSwizzledTensor(output, Swizzle2DType::XOR, true);
  NVF_CHECK(at::equal(expect, output));
  NVF_CHECK(at::equal(input, unswizzled));
}

TEST_F(LegacySwizzleTest, SwizzleExampleCyclicShift) {
  //    1   2  3  4       1   2   3   4
  //    5   6  7  8       8   5   6   7
  //    9  10 11 12  =>   11  12  9  10
  //    13 14 15 16       14  15  16 13
  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto input = torch::tensor(
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}, options);
  auto expect = torch::tensor(
      {{1, 2, 3, 4}, {8, 5, 6, 7}, {11, 12, 9, 10}, {14, 15, 16, 13}}, options);
  auto output = getSwizzledTensor(input, Swizzle2DType::CyclicShift);
  auto unswizzled = getSwizzledTensor(output, Swizzle2DType::CyclicShift, true);
  NVF_CHECK(at::equal(expect, output));
  NVF_CHECK(at::equal(input, unswizzled));
}

TEST_F(LegacySwizzleTest, SwizzleIndexing170) {
  // https://github.com/NVIDIA/Fuser/issues/170
  GTEST_SKIP() << "Repro for an unfixed bug";
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({64, 64});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);

  tv1->split(1, 8);
  tv1->split(1, 4);
  tv1->split(0, 8);
  tv1->split(0, 4);
  // [2 4 8 2 4 8]
  tv1->swizzle(Swizzle2DType::XOR, 1, 4);
  tv1->merge(0);
  tv1->merge(0);
  tv1->merge(1);
  tv1->merge(1);

  for (auto tv : {tv1, tv2}) {
    tv->merge(0);
    tv->split(0, 256);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t = at::randn({64, 64}, options);

  KernelExecutor ke;
  ke.compile(&fusion);
  auto outputs = ke.run({t});

  testValidate(&fusion, outputs, {t}, __LINE__, __FILE__);
}

TEST_F(LegacySwizzleTest, TransformPropagatorSkipSwizzleOnTarget) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeConcreteTensor({64, 64});
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion->addInput(tv0);
  fusion->addOutput(tv2);
  tv1->setMemoryType(MemoryType::Shared);

  tv0->split(1, 8);
  tv0->split(0, 8);
  tv0->merge(0);
  tv0->merge(1);

  tv1->split(1, 8);
  tv1->split(0, 8);
  tv1->swizzle(Swizzle2DType::XOR, 0, 2);
  tv1->merge(0);
  tv1->merge(1);

  tv0->merge(0);

  TransformPropagatorWithCheck propagator(tv0);
  MaxLogicalDomainInfoSpanningTree(tv0).traverse(&propagator);

  auto exprs = StmtSort::getExprsBetween(
      {tv1->getLogicalDomain().begin(), tv1->getLogicalDomain().end()},
      {tv1->getLoopDomain().begin(), tv1->getLoopDomain().end()});
  EXPECT_TRUE(std::any_of(exprs.begin(), exprs.end(), [](Expr* expr) {
    return expr->isA<Swizzle2D>();
  }));
}

TEST_F(LegacySwizzleTest, SwizzleInProducerProjection) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  tv1->split(1, 8);
  tv1->split(0, 8);
  tv1->reorder({{2, 1}});
  tv1->swizzle(SwizzleType::XOR, 2, 3);
  tv1->reorder({{2, 1}});
  tv1->merge(0);
  tv1->merge(1);
  tv1->commitLeafToLogical();
  fusion->addOutput(tv1);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t = at::randn({32, 64}, options);

  KernelExecutor ke;
  ke.compile(fusion.get());
  auto outputs = ke.run({t});

  auto expect = at::empty_like(t);
  for (auto i : arange(t.size(0) / 8)) {
    for (auto j : arange(t.size(1) / 8)) {
      for (auto ii : arange(8)) {
        for (auto jj : arange(8)) {
          expect[i * 8 + ii][j * 8 + jj] = t[i * 8 + ii][j * 8 + (ii ^ jj)];
        }
      }
    }
  }
  testValidate(fusion.get(), outputs, {t}, {expect}, __LINE__, __FILE__);
}

TEST_F(SwizzleTest, Transpose1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 0, 1);
  fusion.addOutput(tv2);
  tv1->setMemoryType(MemoryType::Shared);

  std::vector<IterDomain*> dim0{tv1->axis(0), tv2->axis(1)};
  std::vector<IterDomain*> dim1{tv1->axis(1), tv2->axis(0)};
  AbstractTensor loop{dim0, dim1};

  loop.split(1, 32);
  loop.split(0, 32);
  loop.reorder({{1, 2}});
  loop.merge(0);
  loop.parallelize(0, ParallelType::BIDx);
  // BIDx, 32, 32

  auto smem_alloc = loop.unzip()[0];
  smem_alloc.swizzle(SwizzleType::XOR, 1, 2);
  tv1->setAllocationDomain(smem_alloc.as<IterDomain*>(), true);

  std::swap(loop[1][1], loop[2][1]);
  loop.merge(1);
  loop.split(1, 256);
  loop.parallelize(2, ParallelType::TIDx);
  // BIDx, 4, TIDx

  auto uz = loop.unzip();
  tv1->setLoopDomain(uz[0].as<IterDomain*>());
  tv2->setLoopDomain(uz[1].as<IterDomain*>());

  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t = at::randn({10240, 10240}, options);
  KernelExecutor ke;
  ke.compile(&fusion, {t});
  EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto outputs = ke.run({t});
  EXPECT_TRUE(at::equal(t.t(), outputs[0].as<at::Tensor>()));
}

} // namespace nvfuser
