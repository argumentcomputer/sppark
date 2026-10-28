// Copyright Argument Computer Corporation
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
#ifndef SPPARK_MSM_SPLIT_BUCKETS_CUH
#define SPPARK_MSM_SPLIT_BUCKETS_CUH

#include <limits>

constexpr uint32_t MSM_SPLIT_CHUNK = 256;
constexpr uint32_t MSM_SPLIT_REDUCE_THREADS = 256;

struct msm_split_statistics {
    uint64_t max_bucket[32]{}, chunk_count[32]{};
    uint64_t nonzero_digits[32]{}, zero_digits[32]{};
    uint64_t unit_points = 0;
    double classify_ms = 0, chunks_ms = 0, reduce_ms = 0;
};

struct msm_split_header {
    uint32_t tasks, heavy, overflow, reserved;
    uint32_t max_bucket[32], chunk_count[32];
    uint32_t nonzero_digits[32], zero_digits[32];
    uint32_t unit_points;
};

struct msm_split_geometry {
    size_t ranges_offset, tasks_offset, heavy_offset, partials_offset, bytes;
    uint32_t capacity;

    msm_split_geometry() : ranges_offset(0), tasks_offset(0), heavy_offset(0),
                           partials_offset(0), bytes(0), capacity(0) {}
    msm_split_geometry(size_t count, uint32_t window, uint32_t windows,
                       size_t bucket_bytes) {
        if (!count || count > (size_t(1) << 31) || window < 10 || window > 18 || !windows || windows > 32)
            CUDA_OK(cudaErrorInvalidValue);
        uint32_t bits = 0;
        for (size_t n = count + count / 2; n >>= 1;) ++bits;
        size_t batch = std::max(size_t(1), (size_t(1) << (std::max(bits, window) - window)) >> 6);
        size_t stride = ((count + batch - 1) / batch + 31) & ~size_t(31);
        size_t tasks = windows * ((stride + MSM_SPLIT_CHUNK - 1) / MSM_SPLIT_CHUNK);
        if (tasks > std::numeric_limits<uint32_t>::max()) CUDA_OK(cudaErrorInvalidValue);
        capacity = static_cast<uint32_t>(tasks);
        ranges_offset = (sizeof(msm_split_header) + 31) & ~size_t(31);
        tasks_offset = ranges_offset + size_t(windows) * (size_t(1) << (window-1)) * sizeof(uint2);
        heavy_offset = tasks_offset + tasks * sizeof(uint4);
        partials_offset = (heavy_offset + tasks * sizeof(uint32_t) + 31) & ~size_t(31);
        if (tasks > (std::numeric_limits<size_t>::max() - partials_offset) / bucket_bytes)
            CUDA_OK(cudaErrorInvalidValue);
        bytes = partials_offset + tasks * bucket_bytes;
    }
};

// Cross-stream consumers must finish before asynchronous scratch frees unwind.
class msm_unwind_completion {
    int device_;
    bool complete_ = false;
  public:
    explicit msm_unwind_completion(int device) noexcept : device_(device) {}
    msm_unwind_completion(const msm_unwind_completion&) = delete;
    msm_unwind_completion& operator=(const msm_unwind_completion&) = delete;
    void complete() noexcept { complete_ = true; }
    ~msm_unwind_completion() noexcept {
        if (!complete_) {
            (void)cudaSetDevice(device_);
            (void)cudaDeviceSynchronize();
        }
    }
};

class msm_split_timer {
    cudaEvent_t begin_ = nullptr, end_ = nullptr;
  public:
    explicit msm_split_timer(bool enabled) {
        if (!enabled) return;
        CUDA_OK(cudaEventCreate(&begin_));
        try { CUDA_OK(cudaEventCreate(&end_)); }
        catch (...) { cudaEventDestroy(begin_); throw; }
    }
    msm_split_timer(const msm_split_timer&) = delete;
    msm_split_timer& operator=(const msm_split_timer&) = delete;
    ~msm_split_timer() {
        if (begin_) (void)cudaEventDestroy(begin_);
        if (end_) (void)cudaEventDestroy(end_);
    }
    void start(cudaStream_t stream) { if (begin_) CUDA_OK(cudaEventRecord(begin_, stream)); }
    void stop(cudaStream_t stream) { if (end_) CUDA_OK(cudaEventRecord(end_, stream)); }
    double elapsed() {
        if (!end_) return 0;
        float ms = 0;
        CUDA_OK(cudaEventElapsedTime(&ms, begin_, end_));
        return ms;
    }
};

__device__ inline uint32_t msm_split_reserve(uint32_t* next, uint32_t count,
                                            uint32_t capacity, uint32_t* overflow) {
    uint32_t current = atomicAdd(next, 0);
    for (;;) {
        if (current > capacity || count > capacity - current) {
            atomicExch(overflow, 1); return 0xffffffffu;
        }
        uint32_t observed = atomicCAS(next, current, current + count);
        if (observed == current) return current;
        current = observed;
    }
}

static __global__ void msm_split_classify(const vec2d_t<uint32_t> histogram,
    uint32_t windows, uint32_t window, uint32_t num, uint32_t capacity,
    msm_split_header* header, uint2* ranges, uint4* tasks, uint32_t* heavy) {
    const uint32_t row = 1u << (window-1);
    for (uint32_t id = blockIdx.x*blockDim.x + threadIdx.x;
         id < windows*row; id += gridDim.x*blockDim.x) {
        uint32_t w = id / row, bucket = id % row;
        uint32_t end = histogram[w][bucket];
        uint32_t start = bucket ? histogram[w][bucket-1] : 0;
        ranges[id] = uint2{0, 0};
        if (end < start || end > num) { atomicExch(&header->overflow, 1); continue; }
        uint32_t length = end - start;
        if (bucket == 0) {
            uint32_t nonzero = histogram[w][row-1];
            if (nonzero > num) { atomicExch(&header->overflow, 1); continue; }
            header->nonzero_digits[w] = nonzero;
            header->zero_digits[w] = num - nonzero;
        }
        // The lowest-window unit bucket is handled by batch_addition.
        if (id == 0) { header->unit_points = length; continue; }
        atomicMax(&header->max_bucket[w], length);
        if (length <= MSM_SPLIT_CHUNK) continue;
        uint32_t extra = (length - 1) / MSM_SPLIT_CHUNK;
        uint32_t base = msm_split_reserve(&header->tasks, extra, capacity, &header->overflow);
        if (base == 0xffffffffu) continue;
        uint32_t heavy_id = msm_split_reserve(&header->heavy, 1, capacity, &header->overflow);
        if (heavy_id == 0xffffffffu) continue;
        ranges[id] = uint2{base, extra};
        heavy[heavy_id] = id;
        atomicAdd(&header->chunk_count[w], extra);
        for (uint32_t part = 0; part < extra; ++part) {
            uint32_t first = start + (part + 1) * MSM_SPLIT_CHUNK;
            tasks[base + part] = uint4{id, first, min(end-first, MSM_SPLIT_CHUNK), 0};
        }
    }
}

template<class bucket_t, class affine_h, class bucket_h = typename bucket_t::mem_t>
__launch_bounds__(128) __global__
void msm_split_accumulate(bucket_h* partials, const affine_h* points,
    const vec2d_t<uint32_t> digits, uint32_t window, const uint4* tasks,
    const msm_split_header* header, uint32_t capacity) {
    static_assert(bucket_t::degree == 1, "split buckets require a base-field G1");
    if (header->overflow) return;
    uint32_t count = min(header->tasks, capacity);
    for (uint32_t id = blockIdx.x*blockDim.x + threadIdx.x;
         id < count; id += gridDim.x*blockDim.x) {
        uint4 task = tasks[id];
        uint32_t w = task.x >> (window-1);
        uint32_t digit = digits[w][task.y];
        typename bucket_t::affine_t point = points[digit & 0x7fffffff];
        bucket_t sum = point;
        sum.cneg(digit >> 31);
        for (uint32_t i = 1; i < task.z; ++i) {
            digit = digits[w][task.y+i];
            point = points[digit & 0x7fffffff];
            sum.add(point, digit >> 31);
        }
        partials[id] = sum;
    }
}

template<class bucket_t, class bucket_h = typename bucket_t::mem_t>
__launch_bounds__(MSM_SPLIT_REDUCE_THREADS) __global__
void msm_split_reduce(bucket_h* buckets, const bucket_h* partials,
    const uint2* ranges, const uint32_t* heavy, const msm_split_header* header,
    uint32_t capacity) {
    static_assert(alignof(bucket_h) <= 32, "split shared-memory alignment");
    extern __shared__ __align__(32) unsigned char scratch_bytes[];
    auto* scratch = reinterpret_cast<bucket_h*>(scratch_bytes);
    if (header->overflow) return;
    uint32_t count = min(header->heavy, capacity);
    for (uint32_t id = blockIdx.x; id < count; id += gridDim.x) {
        uint32_t bucket = heavy[id];
        uint2 range = ranges[bucket];
        bucket_t sum;
        memset(&sum, 0, sizeof(sum));
        if (threadIdx.x == 0) sum = buckets[bucket];
        for (uint32_t i = threadIdx.x; i < range.y; i += MSM_SPLIT_REDUCE_THREADS) {
            bucket_t part = partials[range.x+i];
            sum.add(part);
        }
        scratch[threadIdx.x] = sum;
        __syncthreads();
        for (uint32_t offset = MSM_SPLIT_REDUCE_THREADS/2; offset; offset >>= 1) {
            if (threadIdx.x < offset) {
                bucket_t other = scratch[threadIdx.x+offset];
                sum.add(other);
                scratch[threadIdx.x] = sum;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) buckets[bucket] = sum;
        __syncthreads();
    }
}

// Extension-field MSMs retain their existing kernels without instantiating the
// single-lane G1 split kernels.
template<class bucket_t, class affine_h, unsigned degree = bucket_t::degree>
struct msm_split_dispatch {
    template<typename... Args>
    static void accumulate(const Args&...) { CUDA_OK(cudaErrorNotSupported); }
    template<typename... Args>
    static void reduce(const Args&...) { CUDA_OK(cudaErrorNotSupported); }
};

template<class bucket_t, class affine_h>
struct msm_split_dispatch<bucket_t, affine_h, 1> {
    using bucket_h = typename bucket_t::mem_t;
    static void accumulate(const stream_t& stream, unsigned blocks,
                           bucket_h* partials, const affine_h* points,
                           const vec2d_t<uint32_t> digits, uint32_t window,
                           const uint4* tasks, const msm_split_header* header,
                           uint32_t capacity) {
        msm_split_accumulate<bucket_t, affine_h><<<blocks, 128, 0, stream>>>(
            partials, points, digits, window, tasks, header, capacity);
        CUDA_OK(cudaGetLastError());
    }
    static void reduce(const stream_t& stream, unsigned blocks,
                       bucket_h* buckets, const bucket_h* partials,
                       const uint2* ranges, const uint32_t* heavy,
                       const msm_split_header* header, uint32_t capacity) {
        msm_split_reduce<bucket_t><<<blocks, MSM_SPLIT_REDUCE_THREADS,
            sizeof(bucket_h)*MSM_SPLIT_REDUCE_THREADS, stream>>>(
                buckets, partials, ranges, heavy, header, capacity);
        CUDA_OK(cudaGetLastError());
    }
};
#endif
