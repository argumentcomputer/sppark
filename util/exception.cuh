// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_UTIL_EXCEPTION_CUH__
#define __SPPARK_UTIL_EXCEPTION_CUH__

#include "exception.hpp"

using cuda_error = sppark_error;

#ifdef SPPARK_NO_CXX_RUNTIME
// A failing CUDA call is recorded per thread instead of thrown, and the
// caller of the device-pointer interfaces reads it back with
// sppark_take_cuda_error(). Nothing in the transform paths then needs the
// C++ runtime library (std::string, std::runtime_error, std::thread), so
// the archive links into programs built on another C++ runtime.
extern "C" void sppark_record_cuda_error(int code);
extern "C" int sppark_take_cuda_error();

#define CUDA_OK(expr) do {                                  \
    cudaError_t code = expr;                                \
    if (code != cudaSuccess)                                \
        sppark_record_cuda_error(static_cast<int>(code));   \
} while(0)
#else
#define CUDA_OK(expr) do {                                  \
    cudaError_t code = expr;                                \
    if (code != cudaSuccess) {                              \
        auto file = std::strstr(__FILE__, "sppark");        \
        auto str = fmt("%s@%s:%d failed: \"%s\"", #expr,    \
                       file ? file : __FILE__, __LINE__,    \
                       cudaGetErrorString(code));           \
        throw cuda_error{-code, str};                       \
    }                                                       \
} while(0)
#endif

#endif
