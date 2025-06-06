// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <torch/extension.h>

PYBIND11_MODULE(PYTHON_DIRECT_EXTENSION, m) {
  m.doc() = "Python bindings for NvFuser Direct CPP API";
  nvfuser::python::initNvFuserPythonBindings(m.ptr());
}
