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

// Combined inference function type that handles both shape and stride inference
using InferenceFunc = void (*)(
    int64_t* input_shape,    // Input shape array
    int64_t input_size,      // Size of input shape array
    int64_t* output_shape,   // Output shape array
    int64_t output_size,     // Size of output shape array
    int64_t* output_stride,  // Output stride array
    int64_t stride_size      // Size of output stride array
);

class HostIrLlvmJit {
 public:
  // Get singleton instance
  static HostIrLlvmJit& getInstance(int num_threads = 4);

  // Delete copy constructor and assignment operator
  HostIrLlvmJit(const HostIrLlvmJit&) = delete;
  HostIrLlvmJit& operator=(const HostIrLlvmJit&) = delete;

  // Compile a fusion associated with the given output TensorView.
  void compile(const hir::HostIrContainer* container);

  // Set the input tensors
  void setInputTensor(const at::Tensor& input_tensor);

  // If input tensor is set, return true
  bool isInputTensorSet() const;

  // If compiled, return true
  bool isCompiled(const TensorView* output_tv) const;

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