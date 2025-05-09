// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <ops/all_ops.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

class LoopRotationTest : public NVFuserTest {};

TEST_F(LoopRotationTest, RotateInner) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, 3});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  inlineMost();
  scheduler_utils::rotateLoop(tv4, -1, {tv1, tv2});

  const std::string expected_kernel = R"(
// Codegen generated code
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO;
  #pragma unroll 1
  for(nvfuser_index_t i0 = 0LL; i0 < T0.logical_size[0LL]; ++i0) {
    nvfuser_index_t i1;
    i1 = T0.alloc_stride[0LL] * i0;
    nvfuser_index_t i2;
    i2 = 3LL * i0;
    Array<float, 1LL, 1> T1;
    Array<float, 1LL, 1> T2;
    T1[0LL] = 0LL;
    T1[0LL]
       = T0[i1];
    T2[0LL]
       = T1[0LL];
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i3 = 0LL; i3 < 3LL; ++i3) {
      nvfuser_index_t i4;
      i4 = (1LL + i3) + nvfuser_zero;
      Array<float, 1LL, 1> T3;
      T3[0LL]
         = T2[0LL];
      T4[(i2 + (i3 + nvfuser_zero))]
         = T3[0LL];
      T1[0LL] = 0LL;
      if ((i4 < 3LL)) {
        T1[0LL]
           = T0[(i1 + (T0.alloc_stride[1LL] * i4))];
      }
      T2[0LL]
         = T1[0LL];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
  }
}
)";

  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    KernelExecutor ke;
    ke.compile(&fusion, {t0});
    auto cg_outputs = ke.run({t0});
    testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(LoopRotationTest, RotateOuter) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, 3});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  inlineAllAt(tv4, 1);
  scheduler_utils::rotateLoop(tv4, 0, {tv1, tv2});

  const std::string expected_kernel = R"(
// Codegen generated code
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO;
  Array<float, 3LL, 1> T1;
  Array<float, 3LL, 1> T2;
  #pragma unroll
  for(nvfuser_index_t i0 = 0LL; i0 < 3LL; ++i0) {
    T1[i0] = 0LL;
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i0 = 0LL; i0 < 3LL; ++i0) {
    T1[i0]
       = T0[(T0.alloc_stride[1LL] * (i0 + nvfuser_zero))];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i1 = 0LL; i1 < 3LL; ++i1) {
    T2[i1]
       = T1[i1];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll 1
  for(nvfuser_index_t i2 = 0LL; i2 < T0.logical_size[0LL]; ++i2) {
    nvfuser_index_t i3;
    i3 = 3LL * i2;
    nvfuser_index_t i4;
    i4 = T0.alloc_stride[0LL] + (T0.alloc_stride[0LL] * i2);
    bool b5;
    b5 = (1LL + i2) < T0.logical_size[0LL];
    // Alias Allocation - register
    auto& T3 = T1;
    #pragma unroll
    for(nvfuser_index_t i6 = 0LL; i6 < 3LL; ++i6) {
      T3[i6]
         = T2[i6];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i7 = 0LL; i7 < 3LL; ++i7) {
      T4[(i3 + (i7 + nvfuser_zero))]
         = T3[i7];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i0 = 0LL; i0 < 3LL; ++i0) {
      T1[i0] = 0LL;
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i0 = 0LL; i0 < 3LL; ++i0) {
      if (b5) {
        T1[i0]
           = T0[(i4 + (T0.alloc_stride[1LL] * (i0 + nvfuser_zero)))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i1 = 0LL; i1 < 3LL; ++i1) {
      T2[i1]
         = T1[i1];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    KernelExecutor ke;
    ke.compile(&fusion, {t0});
    auto cg_outputs = ke.run({t0});
    testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(LoopRotationTest, NonDivisibleSplit) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, -1});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  for (auto tv : {tv0, tv1, tv2, tv3, tv4}) {
    tv->merge(0);
    tv->split(0, 5);
  }
  inlineAllAt(tv4, 1);
  scheduler_utils::rotateLoop(tv4, 0, {tv1, tv2});

  const std::string expected_kernel = R"(
// Codegen generated code
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO;
  nvfuser_index_t i0;
  i0 = T0.logical_size[0LL] * T0.logical_size[1LL];
  nvfuser_index_t i1;
  i1 = ceilDiv(i0, 5LL);
  Array<float, 5LL, 1> T1;
  Array<float, 5LL, 1> T2;
  #pragma unroll
  for(nvfuser_index_t i2 = 0LL; i2 < 5LL; ++i2) {
    T1[i2] = 0LL;
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i2 = 0LL; i2 < 5LL; ++i2) {
    nvfuser_index_t i3;
    i3 = i2 + nvfuser_zero;
    if ((i3 < i0)) {
      T1[i2]
         = T0[((T0.alloc_stride[0LL] * (i3 / T0.logical_size[1LL])) + (T0.alloc_stride[1LL] * (i3 % T0.logical_size[1LL])))];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i4 = 0LL; i4 < 5LL; ++i4) {
    T2[i4]
       = T1[i4];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll 1
  for(nvfuser_index_t i5 = 0LL; i5 < i1; ++i5) {
    nvfuser_index_t i6;
    i6 = 5LL * i5;
    nvfuser_index_t i7;
    i7 = 5LL + i6;
    // Alias Allocation - register
    auto& T3 = T1;
    #pragma unroll
    for(nvfuser_index_t i8 = 0LL; i8 < 5LL; ++i8) {
      T3[i8]
         = T2[i8];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i9 = 0LL; i9 < 5LL; ++i9) {
      nvfuser_index_t i10;
      i10 = i6 + (i9 + nvfuser_zero);
      if ((i10 < i0)) {
        T4[i10]
           = T3[i9];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i2 = 0LL; i2 < 5LL; ++i2) {
      T1[i2] = 0LL;
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i2 = 0LL; i2 < 5LL; ++i2) {
      nvfuser_index_t i11;
      i11 = i7 + (i2 + nvfuser_zero);
      if ((i11 < i0)) {
        T1[i2]
           = T0[((T0.alloc_stride[0LL] * (i11 / T0.logical_size[1LL])) + (T0.alloc_stride[1LL] * (i11 % T0.logical_size[1LL])))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i4 = 0LL; i4 < 5LL; ++i4) {
      T2[i4]
         = T1[i4];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    KernelExecutor ke;
    ke.compile(&fusion, {t0});
    auto cg_outputs = ke.run({t0});
    testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(LoopRotationTest, CircularBuffered) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, 3});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  inlineAllAt(tv4, 1);
  tv1->circularBuffer(/*number_of_stages=*/5);
  scheduler_utils::rotateLoop(tv4, 0, {tv2});

  const std::string expected_kernel = R"(
// Codegen generated code
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO;
  nvfuser_index_t i0;
  i0 = 4LL * T0.alloc_stride[0LL];
  Array<float, 15LL, 1> T1;
  #pragma unroll 4
  for(nvfuser_index_t i1 = 0LL; i1 < 4LL; ++i1) {
    nvfuser_index_t i2;
    i2 = 3LL * i1;
    nvfuser_index_t i3;
    i3 = T0.alloc_stride[0LL] * i1;
    bool b4;
    b4 = (i1 + nvfuser_zero) < T0.logical_size[0LL];
    #pragma unroll
    for(nvfuser_index_t i5 = 0LL; i5 < 3LL; ++i5) {
      T1[(i2 + i5)] = 0LL;
    }
    #pragma unroll
    for(nvfuser_index_t i5 = 0LL; i5 < 3LL; ++i5) {
      if (b4) {
        T1[(i2 + i5)]
           = T0[(i3 + (T0.alloc_stride[1LL] * (i5 + nvfuser_zero)))];
      }
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  Array<float, 3LL, 1> T2;
  #pragma unroll
  for(nvfuser_index_t i6 = 0LL; i6 < 3LL; ++i6) {
    T2[i6]
       = T1[i6];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll 4
  for(nvfuser_index_t i7 = 0LL; i7 < T0.logical_size[0LL]; ++i7) {
    nvfuser_index_t i8;
    i8 = 4LL + i7;
    nvfuser_index_t i9;
    i9 = 3LL * (i8 % 5LL);
    nvfuser_index_t i10;
    i10 = i0 + (T0.alloc_stride[0LL] * i7);
    nvfuser_index_t i11;
    i11 = 3LL * i7;
    nvfuser_index_t i12;
    i12 = 3LL * ((1LL + i7) % 5LL);
    bool b13;
    b13 = i8 < T0.logical_size[0LL];
    #pragma unroll
    for(nvfuser_index_t i5 = 0LL; i5 < 3LL; ++i5) {
      T1[(i9 + i5)] = 0LL;
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i5 = 0LL; i5 < 3LL; ++i5) {
      if (b13) {
        T1[(i9 + i5)]
           = T0[(i10 + (T0.alloc_stride[1LL] * (i5 + nvfuser_zero)))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    Array<float, 3LL, 1> T3;
    #pragma unroll
    for(nvfuser_index_t i14 = 0LL; i14 < 3LL; ++i14) {
      T3[i14]
         = T2[i14];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i15 = 0LL; i15 < 3LL; ++i15) {
      T4[(i11 + (i15 + nvfuser_zero))]
         = T3[i15];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i6 = 0LL; i6 < 3LL; ++i6) {
      T2[i6]
         = T1[(i12 + i6)];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {5, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    KernelExecutor ke;
    ke.compile(&fusion, {t0});
    auto cg_outputs = ke.run({t0});
    testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(LoopRotationTest, SelectCircularBufferLoad) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, 3});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  inlineAllAt(tv4, 1);
  tv1->circularBuffer(/*number_of_stages=*/5);
  scheduler_utils::rotateLoop(tv4, 0, {tv1, tv2});

  const std::string expected_kernel = R"(
// Codegen generated code
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO;
  nvfuser_index_t i0;
  i0 = 4LL * T0.alloc_stride[0LL];
  nvfuser_index_t i1;
  i1 = 5LL * T0.alloc_stride[0LL];
  bool b2;
  b2 = 4LL < T0.logical_size[0LL];
  Array<float, 15LL, 1> T1;
  #pragma unroll
  for(nvfuser_index_t i3 = 0LL; i3 < 3LL; ++i3) {
    T1[i3] = 0LL;
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i3 = 0LL; i3 < 3LL; ++i3) {
    T1[i3]
       = T0[(T0.alloc_stride[1LL] * (i3 + nvfuser_zero))];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll 4
  for(nvfuser_index_t i4 = 0LL; i4 < 4LL; ++i4) {
    nvfuser_index_t i5;
    i5 = 3LL + (3LL * i4);
    nvfuser_index_t i6;
    i6 = T0.alloc_stride[0LL] + (T0.alloc_stride[0LL] * i4);
    bool b7;
    b7 = ((1LL + i4) + nvfuser_zero) < T0.logical_size[0LL];
    #pragma unroll
    for(nvfuser_index_t i3 = 0LL; i3 < 3LL; ++i3) {
      T1[(i5 + i3)] = 0LL;
    }
    #pragma unroll
    for(nvfuser_index_t i3 = 0LL; i3 < 3LL; ++i3) {
      if (b7) {
        T1[(i5 + i3)]
           = T0[(i6 + (T0.alloc_stride[1LL] * (i3 + nvfuser_zero)))];
      }
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  Array<float, 3LL, 1> T2;
  #pragma unroll
  for(nvfuser_index_t i3 = 0LL; i3 < 3LL; ++i3) {
    T1[(12LL + i3)] = 0LL;
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i3 = 0LL; i3 < 3LL; ++i3) {
    if (b2) {
      T1[(12LL + i3)]
         = T0[(i0 + (T0.alloc_stride[1LL] * (i3 + nvfuser_zero)))];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll
  for(nvfuser_index_t i8 = 0LL; i8 < 3LL; ++i8) {
    T2[i8]
       = T1[i8];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  #pragma unroll 4
  for(nvfuser_index_t i9 = 0LL; i9 < T0.logical_size[0LL]; ++i9) {
    nvfuser_index_t i10;
    i10 = 3LL * i9;
    nvfuser_index_t i11;
    i11 = 3LL * (i9 % 5LL);
    nvfuser_index_t i12;
    i12 = i1 + (T0.alloc_stride[0LL] * i9);
    nvfuser_index_t i13;
    i13 = 3LL * ((1LL + i9) % 5LL);
    bool b14;
    b14 = (5LL + i9) < T0.logical_size[0LL];
    Array<float, 3LL, 1> T3;
    #pragma unroll
    for(nvfuser_index_t i15 = 0LL; i15 < 3LL; ++i15) {
      T3[i15]
         = T2[i15];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i16 = 0LL; i16 < 3LL; ++i16) {
      T4[(i10 + (i16 + nvfuser_zero))]
         = T3[i16];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i3 = 0LL; i3 < 3LL; ++i3) {
      T1[(i11 + i3)] = 0LL;
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i3 = 0LL; i3 < 3LL; ++i3) {
      if (b14) {
        T1[(i11 + i3)]
           = T0[(i12 + (T0.alloc_stride[1LL] * (i3 + nvfuser_zero)))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i8 = 0LL; i8 < 3LL; ++i8) {
      T2[i8]
         = T1[(i13 + i8)];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {5, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    KernelExecutor ke;
    ke.compile(&fusion, {t0});
    auto cg_outputs = ke.run({t0});
    testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }
}

// This is a case similar to matmul, where we have
// tv4 = set(tv0) // cp.async for matmul
// tv1 = set(tv4) // ld.matrix for matmul
// and both are circular buffered
TEST_F(LoopRotationTest, MultipleCircularBuffer) {
  if (!deviceMajorMinorCheck(8)) {
    GTEST_SKIP() << "skipping tests on pre-Ampere GPUs";
    return;
  }
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, 3});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  auto tv4 = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  tv4->setMemoryType(MemoryType::Shared);

  inlineAllAt(tv3, 1);
  inlineSelectedAt({tv1, tv2, tv3}, tv3, 2);

  tv4->circularBuffer(/*number_of_stages=*/5);
  tv1->circularBuffer(/*number_of_stages=*/2);
  scheduler_utils::rotateLoop(tv3, 0, {tv1});

  const std::string expected_kernel = R"(
// Codegen generated code

__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T3) {
  alignas(16) extern __shared__ char array[];
  const unsigned smem_offset = 0;
  NVFUSER_DEFINE_MAGIC_ZERO;
  float* T4 = reinterpret_cast<float*>(array + smem_offset + 0LL);
  uint32_t i0;
  i0 = toSmem(T4);
  float* ptr1;
  ptr1 = T0.data + (4LL * T0.alloc_stride[0LL]);
  #pragma unroll 4
  for(nvfuser_index_t i2 = 0LL; i2 < 4LL; ++i2) {
    float* ptr3;
    ptr3 = T0.data + (T0.alloc_stride[0LL] * i2);
    uint32_t i4;
    i4 = i0 + (12LL * i2);
    bool b5;
    b5 = (i2 + nvfuser_zero) < T0.logical_size[0LL];
    #pragma unroll
    for(nvfuser_index_t i6 = 0LL; i6 < 3LL; ++i6) {
      asm volatile(
        "{\n"
        "  .reg .pred p0; \n"
        "  setp.ne.b32 p0, %3, 0;\n"
        "  cp.async.ca.shared.global [%0], [%1], %2, p0;\n"
        "}\n"
        :
        :"r"((uint32_t)((i4 + (4LL * i6)))),
         "l"((ptr3 + (T0.alloc_stride[1LL] * (i6 + nvfuser_zero)))),
         "n"(4LL),
         "r"((uint32_t)((!b5)))
      );
    }
    asm volatile("cp.async.commit_group;\n");
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  asm volatile("cp.async.wait_group %0;\n"::"n"(3LL));
  Array<float, 2LL, 1> T1;
  T1[0LL]
     = T4[0LL];
  #pragma unroll 4
  for(nvfuser_index_t i7 = 0LL; i7 < T0.logical_size[0LL]; ++i7) {
    float* ptr8;
    ptr8 = ptr1 + (T0.alloc_stride[0LL] * i7);
    nvfuser_index_t i9;
    i9 = 4LL + i7;
    uint32_t i10;
    i10 = i0 + (12LL * (i9 % 5LL));
    nvfuser_index_t i11;
    i11 = 1LL + (3LL * (i7 % 5LL));
    nvfuser_index_t i12;
    i12 = 3LL * i7;
    bool b13;
    b13 = i9 < T0.logical_size[0LL];
    #pragma unroll
    for(nvfuser_index_t i6 = 0LL; i6 < 3LL; ++i6) {
      asm volatile(
        "{\n"
        "  .reg .pred p0; \n"
        "  setp.ne.b32 p0, %3, 0;\n"
        "  cp.async.ca.shared.global [%0], [%1], %2, p0;\n"
        "}\n"
        :
        :"r"((uint32_t)((i10 + (4LL * i6)))),
         "l"((ptr8 + (T0.alloc_stride[1LL] * (i6 + nvfuser_zero)))),
         "n"(4LL),
         "r"((uint32_t)((!b13)))
      );
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    asm volatile("cp.async.commit_group;\n");
    #pragma unroll 1
    for(nvfuser_index_t i14 = 0LL; i14 < 2LL; ++i14) {
      T1[((1LL + i14) % 2LL)]
         = T4[(i11 + i14)];
      Array<float, 1LL, 1> T2;
      T2[0LL]
         = T1[i14];
      T3[(i12 + (i14 + nvfuser_zero))]
         = T2[0LL];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    Array<float, 1LL, 1> T2;
    T2[0LL]
       = T1[0LL];
    T3[(2LL + i12)]
       = T2[0LL];
    NVFUSER_UPDATE_MAGIC_ZERO;
    asm volatile("cp.async.wait_group %0;\n"::"n"(3LL));
    T1[0LL]
       = T4[(3LL * ((1LL + i7) % 5LL))];
  }
  asm volatile("cp.async.wait_all;\n");
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {5, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    KernelExecutor ke;
    ke.compile(&fusion, {t0});
    auto cg_outputs = ke.run({t0});
    testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }
}
} // namespace nvfuser
