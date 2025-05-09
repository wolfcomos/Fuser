// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <utils.h>

#include <nvtx3/nvToolsExt.h>

// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <stdio.h>
#include <chrono>
#include <cstdio>

namespace nvfuser {
namespace inst {

//! An optional record of selected timestamped operations, events and counters
//!
//! This class is not intended to be used directly. Instead, the operations
//! to be traced are marked (for example using the FUSER_PERF_SCOPE macro)
//!
//! In order to enable tracing, the `NVFUSER_TRACE` environment
//! variable is set to point to a trace file (ex `test.trace`). The file name
//! may be a relative or an absolute path.
//!
//! The trace uses the Chrome Tracing (Catapult) format, which is a well
//! documented JSON based format supported by multiple tools:
//! https://chromium.googlesource.com/catapult/+/HEAD/tracing/README.md
//!
//! An easy way to view traces is to type `about://tracing` in Chrome or
//! Chromium.
//!
class Trace : public NonCopyable {
 public:
  using Clock = std::chrono::steady_clock;

 public:
  NVF_API static Trace* instance() {
    static Trace trace;
    return &trace;
  }

  void beginEvent(const char* name) {
    if (log_file_ != nullptr) {
      logEvent('B', name);
    }
    if (record_nvtx_range_) {
      nvtxRangePushA(name);
    }
  }

  void endEvent(const char* name) {
    if (record_nvtx_range_) {
      nvtxRangePop();
    }
    if (log_file_ != nullptr) {
      logEvent('E', name);
    }
  }

 private:
  NVF_API Trace();
  NVF_API ~Trace();

  NVF_API void logEvent(char ph, const char* name, char sep = ',');

 private:
  FILE* log_file_ = nullptr;
  Clock::time_point start_timestamp_;
  bool record_nvtx_range_ = true;
};

//! \internal Automatic scope for a perf marker
//!   (normally used through the FUSER_PERF_SCOPE macro)
class TraceScope : public NonCopyable {
 public:
  explicit TraceScope(const char* event_name) : event_name_(event_name) {
    Trace::instance()->beginEvent(event_name_);
  }

  ~TraceScope() {
    Trace::instance()->endEvent(event_name_);
  }

 private:
  const char* event_name_ = nullptr;
};

#define FUSER_MACRO_CONCAT2(a, b) a##b
#define FUSER_MACRO_CONCAT(a, b) FUSER_MACRO_CONCAT2(a, b)
#define FUSER_ANONYMOUS(prefix) FUSER_MACRO_CONCAT(prefix, __COUNTER__)

//! Defines a scope we want to measure and record in a perf trace
//!
//! \param name The name of the scope, normally a simple string literal
//!
#define FUSER_PERF_SCOPE(name) \
  nvfuser::inst::TraceScope FUSER_ANONYMOUS(_perf_scope_)(name)

} // namespace inst
} // namespace nvfuser
