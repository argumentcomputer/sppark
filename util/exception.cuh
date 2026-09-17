// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_UTIL_EXCEPTION_CUH__
#define __SPPARK_UTIL_EXCEPTION_CUH__

#include "exception.hpp"

using cuda_error = sppark_error;

#ifdef SPPARK_NO_CXX_RUNTIME
// A failing CUDA call ends the process with a message instead of throwing.
// The code after a CUDA_OK assumes the call succeeded, so without
// exceptions the only safe reaction to a failure is to stop before a bad
// handle or table can be used or cached. Nothing in the transform paths
// then needs the C++ runtime library (std::string, std::runtime_error,
// std::thread), so the archive links into programs built on another one.
extern "C" [[noreturn]] void sppark_cuda_fail(const char* expr, const char* file,
                                              int line, int code);

#define CUDA_OK(expr) do {                                  \
    cudaError_t code = expr;                                \
    if (code != cudaSuccess)                                \
        sppark_cuda_fail(#expr, __FILE__, __LINE__,         \
                         static_cast<int>(code));           \
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
