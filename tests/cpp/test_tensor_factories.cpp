// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <codegen.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

class TensorFactoryTest : public NVFuserTest {};

TEST_F(TensorFactoryTest, StandaloneFull) {
  auto sizes = {0, 1, 10, 17, 1024};
  auto dtypes = {
      at::kBool,
      at::kFloat,
      at::kLong,
      at::kDouble,
      at::kHalf,
      at::kBFloat16,
      at::kInt,
      at::kComplexFloat,
      at::kComplexDouble};

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  Val* size = IrBuilder::create<Val>(DataType::Int);
  Val* fill_val1 = IrBuilder::create<Val>(DataType::Int);
  Val* fill_val2 = IrBuilder::create<Val>(DataType::Int);
  Val* fill_val3 = IrBuilder::create<Val>(DataType::Int);
  fusion->addInput(size);
  fusion->addInput(fill_val1);
  fusion->addInput(fill_val2);
  fusion->addInput(fill_val3);
  for (auto dtype : dtypes) {
    if (!isSupportedTypeByDevice(aten_to_data_type(dtype))) {
      continue;
    }
    auto out_tv = full({size}, fill_val1, aten_to_data_type(dtype));
    fusion->addOutput(out_tv);
    out_tv = full({size, size}, fill_val2, aten_to_data_type(dtype));
    fusion->addOutput(out_tv);
    out_tv = full_like(out_tv, fill_val3);
    fusion->addOutput(out_tv);
  }

  FusionExecutorCache executor_cache(std::move(fusion));

  for (auto size : sizes) {
    auto cg_outputs = executor_cache.runFusionWithInputs({size, 11, 12, 13});

    testValidate(
        executor_cache.fusion(),
        cg_outputs,
        {size, 11, 12, 13},
        __LINE__,
        __FILE__);
  }
}

TEST_F(TensorFactoryTest, StandaloneZeros) {
  auto sizes = {0, 1, 10, 17, 1024};
  auto dtypes = {
      at::kBool,
      at::kFloat,
      at::kLong,
      at::kDouble,
      at::kHalf,
      at::kBFloat16,
      at::kInt,
      at::kComplexFloat,
      at::kComplexDouble};

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  Val* size = IrBuilder::create<Val>(DataType::Int);
  fusion->addInput(size);
  for (auto dtype : dtypes) {
    if (!isSupportedTypeByDevice(aten_to_data_type(dtype))) {
      continue;
    }
    auto out_tv = zeros({size}, aten_to_data_type(dtype));
    fusion->addOutput(out_tv);
    out_tv = zeros({size, size}, aten_to_data_type(dtype));
    fusion->addOutput(out_tv);
    out_tv = zeros_like(out_tv);
    fusion->addOutput(out_tv);
  }

  FusionExecutorCache executor_cache(std::move(fusion));

  for (auto size : sizes) {
    auto cg_outputs = executor_cache.runFusionWithInputs({size});

    testValidate(
        executor_cache.fusion(), cg_outputs, {size}, __LINE__, __FILE__);
  }
}

TEST_F(TensorFactoryTest, StandaloneOnes) {
  auto sizes = {0, 1, 10, 17, 1024};
  auto dtypes = {
      at::kBool,
      at::kFloat,
      at::kLong,
      at::kDouble,
      at::kHalf,
      at::kBFloat16,
      at::kInt,
      at::kComplexFloat,
      at::kComplexDouble};

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  Val* size = IrBuilder::create<Val>(DataType::Int);
  fusion->addInput(size);
  for (auto dtype : dtypes) {
    if (!isSupportedTypeByDevice(aten_to_data_type(dtype))) {
      continue;
    }
    auto out_tv = ones({size}, aten_to_data_type(dtype));
    fusion->addOutput(out_tv);
    out_tv = ones({size, size}, aten_to_data_type(dtype));
    fusion->addOutput(out_tv);
    out_tv = ones_like(out_tv);
    fusion->addOutput(out_tv);
  }

  FusionExecutorCache executor_cache(std::move(fusion));

  for (auto size : sizes) {
    auto cg_outputs = executor_cache.runFusionWithInputs({size});

    testValidate(
        executor_cache.fusion(), cg_outputs, {size}, __LINE__, __FILE__);
  }
}

TEST_F(TensorFactoryTest, StandaloneIota) {
  auto starts = {-1., 0., 10.3, 1024. * 256};
  auto steps = {-1.5, 1., 2.};
  auto lengths = {0, 1, 2, 10, 1023, 1024, 1024 * 1024};
  auto dtypes = {at::kInt, at::kLong, at::kFloat, at::kDouble};

  for (auto dtype : dtypes) {
    auto data_type = aten_to_data_type(dtype);
    auto input_type =
        (data_type == DataType::Int32 || data_type == DataType::Int
             ? DataType::Int
             : DataType::Double);

    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    Val* length = IrBuilder::create<Val>(DataType::Int);

    Val* start = IrBuilder::create<Val>(input_type);
    Val* step = IrBuilder::create<Val>(input_type);
    fusion->addInput(length);
    fusion->addInput(start);
    fusion->addInput(step);
    auto tv0 = iota(length, start, step, data_type);
    fusion->addOutput(tv0);

    FusionExecutorCache executor_cache(std::move(fusion));

    switch (dtype) {
      case at::kInt:
      case at::kLong: {
        for (auto length : lengths) {
          for (auto start : starts) {
            for (auto step : steps) {
              int64_t start_ = (int64_t)start;
              int64_t step_ = (int64_t)step;
              auto cg_outputs =
                  executor_cache.runFusionWithInputs({length, start_, step_});

              testValidate(
                  executor_cache.fusion(),
                  cg_outputs,
                  {length, start_, step_},
                  __LINE__,
                  __FILE__);
            }
          }
        }
        break;
      }
      case at::kFloat:
      case at::kDouble: {
        for (auto length : lengths) {
          for (auto start : starts) {
            for (auto step : steps) {
              double start_ = (double)start;
              double step_ = (double)step;
              auto cg_outputs =
                  executor_cache.runFusionWithInputs({length, start_, step_});

              testValidate(
                  executor_cache.fusion(),
                  cg_outputs,
                  {length, start_, step_},
                  __LINE__,
                  __FILE__);
            }
          }
        }
        break;
      }
      default:
        NVF_THROW();
    }
  }
}

TEST_F(TensorFactoryTest, SimpleTriu) {
  std::vector<std::vector<int64_t>> input_sizes_2d = {
      {64, 64}, {4, 16}, {16, 4}};
  std::vector<std::vector<int64_t>> input_sizes_3d = {{16, 8, 32}};
  auto offsets = {0, 1, 2, -1, -2, 200, -200};

  for (auto in : {input_sizes_2d, input_sizes_3d}) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    auto tv_to_triu_on = makeSymbolicTensor(in.at(0).size(), DataType::Half);
    auto input_offset = IrBuilder::create<Val>(DataType::Int);
    auto out = triu(tv_to_triu_on, input_offset);

    fusion->addInput(tv_to_triu_on);
    fusion->addInput(input_offset);
    fusion->addOutput(out);

    FusionExecutorCache executor_cache(std::move(fusion));

    for (auto input_size : in) {
      for (auto offset : offsets) {
        auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
        auto in_tensor = at::randn(input_size, options);

        auto cg_outputs =
            executor_cache.runFusionWithInputs({in_tensor, offset});

        testValidate(
            executor_cache.fusion(),
            cg_outputs,
            {in_tensor, offset},
            {at::triu(in_tensor, offset)},
            __LINE__,
            __FILE__);
      }
    }
  }
}

TEST_F(TensorFactoryTest, StandaloneARange) {
  auto starts_ends = {-1., 0., 10.3, 1024. * 256};
  auto steps = {-1.5, 1., 2.};
  auto dtypes = {at::kFloat, at::kLong, at::kDouble};

  for (auto dtype : dtypes) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    Val* start_int = IrBuilder::create<Val>(DataType::Int);
    Val* end_int = IrBuilder::create<Val>(DataType::Int);
    Val* step_int = IrBuilder::create<Val>(DataType::Int);
    Val* start_double = IrBuilder::create<Val>(DataType::Double);
    Val* end_double = IrBuilder::create<Val>(DataType::Double);
    Val* step_double = IrBuilder::create<Val>(DataType::Double);
    fusion->addInput(start_int);
    fusion->addInput(end_int);
    fusion->addInput(step_int);
    fusion->addInput(start_double);
    fusion->addInput(end_double);
    fusion->addInput(step_double);
    auto tv0 = arange(start_int, end_int, step_int, aten_to_data_type(dtype));
    auto tv1 =
        arange(start_double, end_double, step_double, aten_to_data_type(dtype));
    auto tv2 =
        arange(start_int, end_double, step_double, aten_to_data_type(dtype));
    auto tv3 =
        arange(start_double, end_double, step_int, aten_to_data_type(dtype));
    fusion->addOutput(tv0);
    fusion->addOutput(tv1);
    fusion->addOutput(tv2);
    fusion->addOutput(tv3);

    FusionExecutorCache executor_cache(std::move(fusion));

    for (auto start : starts_ends) {
      for (auto end : starts_ends) {
        for (auto step : steps) {
          if (std::signbit(end - start) != std::signbit(step)) {
            continue;
          }

          auto cg_outputs = executor_cache.runFusionWithInputs(
              {(int64_t)start,
               (int64_t)end,
               (int64_t)step,
               (double)start,
               (double)end,
               (double)step});

          testValidate(
              executor_cache.fusion(),
              cg_outputs,
              {(int64_t)start,
               (int64_t)end,
               (int64_t)step,
               (double)start,
               (double)end,
               (double)step},
              __LINE__,
              __FILE__);
        }
      }
    }
  }
}

TEST_F(TensorFactoryTest, StandaloneEye) {
  auto sizes = {0, 1, 10, 17, 1024};
  auto dtypes = {
      at::kBool,
      at::kFloat,
      at::kLong,
      at::kDouble,
      at::kHalf,
      at::kBFloat16,
      at::kInt,
      at::kComplexFloat,
      at::kComplexDouble};

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  Val* size = IrBuilder::create<Val>(DataType::Int);
  Val* maybe_m = IrBuilder::create<Val>(DataType::Int);
  fusion->addInput(size);
  fusion->addInput(maybe_m);
  for (auto dtype : dtypes) {
    if (!isSupportedTypeByDevice(aten_to_data_type(dtype))) {
      continue;
    }
    auto out_tv1 = eye(size, aten_to_data_type(dtype));
    fusion->addOutput(out_tv1);
    auto out_tv2 = eye(size, maybe_m, aten_to_data_type(dtype));
    fusion->addOutput(out_tv2);
  }

  FusionExecutorCache executor_cache(std::move(fusion));

  for (auto size : sizes) {
    auto cg_outputs = executor_cache.runFusionWithInputs({size, 15});

    testValidate(
        executor_cache.fusion(), cg_outputs, {size, 15}, __LINE__, __FILE__);
  }
}

TEST_F(TensorFactoryTest, TensorConstruct) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  Val* i00 = IrBuilder::create<Val>(DataType::Int);
  Val* i01 = IrBuilder::create<Val>(DataType::Int);
  Val* i10 = IrBuilder::create<Val>(DataType::Int);
  Val* i11 = IrBuilder::create<Val>(DataType::Int);
  fusion->addInput(i00);
  fusion->addInput(i01);
  fusion->addInput(i10);
  fusion->addInput(i11);
  auto output = tensor(std::vector<std::vector<Val*>>{{i00, i01}, {i10, i11}});
  fusion->addOutput(output);

  KernelExecutor ke;
  ke.compile(fusion.get());
  auto cg_outputs = ke.run({00, 01, 10, 11});

  testValidate(fusion.get(), cg_outputs, {00, 01, 10, 11}, __LINE__, __FILE__);
}

TEST_F(TensorFactoryTest, MetadataAsTensor) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeSymbolicTensor(4);
  TensorView* tv1 = makeSymbolicTensor(4);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto meta0 = IrBuilder::metadataExpr(tv0);
  auto meta1 = IrBuilder::metadataExpr(tv1);

  auto meta0_copy0 = set(meta0);
  auto meta1_copy0 = set(meta1);

  // also test unamed structure
  auto unamed_dtype0 = metaDataTypeOf(tv0);
  std::get<StructType>(unamed_dtype0.type).name = "";
  auto unamed_dtype1 = metaDataTypeOf(tv1);
  std::get<StructType>(unamed_dtype1.type).name = "";
  auto meta0_copy1 = IrBuilder::create<Val>(unamed_dtype0);
  auto meta1_copy1 = IrBuilder::create<Val>(unamed_dtype1);
  IrBuilder::create<LoadStoreOp>(
      LoadStoreOpType::Set, meta0_copy1, meta0_copy0);
  IrBuilder::create<LoadStoreOp>(
      LoadStoreOpType::Set, meta1_copy1, meta1_copy0);

  auto meta0_copy2 = set(meta0_copy1);
  auto meta1_copy2 = set(meta1_copy1);

  auto size0 = IrBuilder::getAttrExpr(meta0_copy2, "logical_size");
  auto stride0 = IrBuilder::getAttrExpr(meta0_copy2, "alloc_stride");
  auto size1 = IrBuilder::getAttrExpr(meta1_copy2, "logical_size");
  auto stride1 = IrBuilder::getAttrExpr(meta1_copy2, "alloc_stride");

  auto output = tensor(std::vector<Val*>{size0, stride0, size1, stride1});
  fusion->addOutput(output);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto input0 = at::randn({2, 3, 4, 5}, options);
  auto input1 = at::randn({6, 7, 8, 9}, options);

  KernelExecutor ke;
  ke.compile(fusion.get());
  auto cg_outputs = ke.run({input0, input1});

  testValidate(fusion.get(), cg_outputs, {input0, input1}, __LINE__, __FILE__);
}

TEST_F(TensorFactoryTest, NoInputs) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  Val* size = IrBuilder::create<Val>(16);
  Val* fill_value = IrBuilder::create<Val>(1.0);
  TensorView* out = full({size}, fill_value, DataType::Float);
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto out_tensors = executor_cache.runFusionWithInputs({});
  testValidate(executor_cache.fusion(), out_tensors, {}, __LINE__, __FILE__);
}

TEST_F(TensorFactoryTest, FactoryBroadcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  Val* i = IrBuilder::create<Val>(DataType::Int);
  fusion->addInput(i);
  TensorView* tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);

  auto tv1 = arange(i);
  auto tv2 = full({i}, fusion->oneVal(DataType::Float), DataType::Float);
  // Random factory ops.
  // NOTE: we fix seeds and offsets here so that we can easily validate
  auto tv3 = normal(
      {i},
      fusion->zeroVal(DataType::Float),
      fusion->oneVal(DataType::Float),
      DataType::Float,
      /*Val* philox_seed=*/fusion->zeroVal(),
      /*Val* philox_offset=*/fusion->zeroVal());
  auto tv4 = randn(
      {i},
      DataType::Float,
      /*Val* philox_seed=*/fusion->zeroVal(),
      /*Val* philox_offset=*/fusion->zeroVal());
  auto tv5 = uniform(
      {i},
      fusion->zeroVal(DataType::Float),
      fusion->oneVal(DataType::Float),
      DataType::Float,
      /*Val* philox_seed=*/fusion->zeroVal(),
      /*Val* philox_offset=*/fusion->zeroVal());
  auto tv6 = rand(
      {i},
      DataType::Float,
      /*Val* philox_seed=*/fusion->zeroVal(),
      /*Val* philox_offset=*/fusion->zeroVal());

  auto tv7 = add(tv1, tv0);
  auto tv8 = add(tv2, tv0);
  auto tv9 = add(tv3, tv0);
  auto tv10 = add(tv4, tv0);
  auto tv11 = add(tv5, tv0);
  auto tv12 = add(tv6, tv0);

  fusion->addOutput(tv7);
  fusion->addOutput(tv8);
  fusion->addOutput(tv9);
  fusion->addOutput(tv10);
  fusion->addOutput(tv11);
  fusion->addOutput(tv12);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({100}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({1, t0});

  at::manual_seed(0);
  at::Tensor randn_sample = at::randn({1}, options);
  at::manual_seed(0);
  at::Tensor rand_sample = at::rand({1}, options);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {1, t0},
      {t0,
       t0 + 1,
       t0 + randn_sample,
       t0 + randn_sample,
       t0 + rand_sample,
       t0 + rand_sample},
      __LINE__,
      __FILE__);
}

} // namespace nvfuser
