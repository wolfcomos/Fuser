// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <fusion.h> // For TensorView and at::Tensor
#include <memory>  // For std::unique_ptr

namespace nvfuser {

class HostIrLlvmJit {
 public:
  // Constructor initializes the JIT
  explicit HostIrLlvmJit(int num_threads = 0);
  // Destructor is required for PIMPL with std::unique_ptr
  ~HostIrLlvmJit();

  // Enable move semantics
  HostIrLlvmJit(HostIrLlvmJit&&) noexcept;
  HostIrLlvmJit& operator=(HostIrLlvmJit&&) noexcept;

  // Disable copy
  HostIrLlvmJit(const HostIrLlvmJit&) = delete;
  HostIrLlvmJit& operator=(const HostIrLlvmJit&) = delete;

  // Compile a fusion associated with the given output TensorView.
  void compile(const TensorView* output_tv);

  // Allocate an output tensor with the given input tensors
  at::Tensor allocateOutputTensor(const std::vector<at::Tensor>& input_tensors);

  // Infer the shape and stride of the output tensor
  void inferShapeAndStride(std::vector<int64_t>& result_shape, std::vector<int64_t>& result_stride);

  // Set the input tensors
  void setInputTensor(const at::Tensor& input_tensor);

 private:
  struct LlvmJitImpl; // The PIMPL forward declaration
  std::unique_ptr<LlvmJitImpl> pimpl_;
  static std::vector<at::Tensor> input_tensors;
};

} // namespace nvfuser