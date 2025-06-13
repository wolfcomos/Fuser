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
  // Get singleton instance
  static HostIrLlvmJit& getInstance(int num_threads = 4);

  // Delete copy constructor and assignment operator
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
  // Private constructor
  explicit HostIrLlvmJit(int num_threads = 4);
  
  // Destructor is required for PIMPL with std::unique_ptr
  ~HostIrLlvmJit();

  // Enable move semantics
  HostIrLlvmJit(HostIrLlvmJit&&) noexcept;
  HostIrLlvmJit& operator=(HostIrLlvmJit&&) noexcept;

  struct LlvmJitImpl; // The PIMPL forward declaration
  std::unique_ptr<LlvmJitImpl> pimpl_;
  std::vector<at::Tensor> input_tensors_; // Changed from static to member variable
};

} // namespace nvfuser