// Copyright Argument Computer Corporation
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0

#include <ff/alt_bn128.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
#define SPPARK_DONT_INSTANTIATE_TEMPLATES
#define TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
#include <msm/pippenger.cuh>
#include <polynomial/div_by_x_minus_z.cuh>
#include <util/all_gpus.cpp>

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <thread>

using point_t = jacobian_t<fp_t>;
using bucket_t = xyzz_t<fp_t>;
using affine_t = bucket_t::affine_t;
using msm_impl = msm_t<bucket_t, point_t, affine_t, fr_t>;

static RustError invoke_msm(msm_impl& msm, point_t& result,
                            const affine_t* points, size_t count,
                            const fr_t* scalars, bool resident, bool mont)
{
    if (resident) {
        dev_ptr_t<affine_t> point_view(const_cast<affine_t*>(points), count);
        dev_ptr_t<fr_t> scalar_view(const_cast<fr_t*>(scalars), count);
        return msm.invoke(result, point_view, count, scalar_view, mont);
    }
    return msm.invoke(result, points, count, scalars, mont);
}

static void invoke_division(bool rotate, fr_t* values, size_t count,
                             const fr_t& z, const stream_t& stream, int grid)
{
    if (rotate) div_by_x_minus_z<true>(values, count, z, stream, grid);
    else div_by_x_minus_z<false>(values, count, z, stream, grid);
}

#ifndef __CUDA_ARCH__
static void check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

static point_t generator_multiple(const fr_t& scalar)
{
    fr_t::pow_t bits;
    scalar.to_scalar(bits);
    point_t result{fp_t(0), fp_t(1), fp_t(0)};
    point_t power = affine_t{fp_t(1), fp_t(2)};
    for (unsigned bit = 0; bit < 254; ++bit) {
        if ((bits[bit / 8] >> (bit % 8)) & 1) result.add(power);
        power.dbl();
    }
    return result;
}
static void check_msm(const gpu_t& gpu, size_t count, unsigned pattern,
                      bool resident, bool mont)
{
    std::vector<affine_t> basis(263);
    std::vector<fr_t> weights(basis.size());
    for (size_t i = 0; i < basis.size(); ++i) {
        weights[i] = fr_t(uint64_t(i % 17 ? (i + 1) * (i + 1) + 11 : 0));
        weights[i].cneg(i % 7 == 0);
        basis[i] = generator_multiple(weights[i]);
    }
    std::vector<affine_t> points(count);
    std::vector<fr_t> scalars(count);
    fr_t total(0), high = fr_t(2)^252;
    for (size_t i = 0; i < count; ++i) {
        fr_t weight = weights[i % basis.size()];
        if (i < 768) {
            weight = fr_t(i < 512 ? 1 : 0);
            weight.cneg(i >= 256 && i < 512);
            points[i] = generator_multiple(weight);
        } else {
            points[i] = basis[i % basis.size()];
        }
        switch (pattern) {
        case 0: scalars[i] = fr_t(2); break;
        case 1: scalars[i] = -fr_t(2); break;
        case 2: scalars[i] = high; break;
        case 3: scalars[i] = fr_t(0); break;
        case 4: scalars[i] = fr_t(1); break;
        case 5: scalars[i] = -fr_t(1); break;
        default:
            scalars[i] = fr_t(uint64_t(i % 31));
            scalars[i].cneg(i % 2);
            break;
        }
        total += weight * scalars[i];
    }
    const auto expected = generator_multiple(total);
    if (!mont) for (auto& scalar : scalars) scalar.from();
    const auto original = scalars;
    msm_impl msm(nullptr, count, sizeof(affine_t), gpu.id());
    dev_ptr_t<affine_t> device_points(resident ? count : 0);
    dev_ptr_t<fr_t> device_scalars(resident ? count : 0);
    if (resident) {
        gpu.HtoD(device_points.data(), points.data(), count);
        gpu.HtoD(device_scalars.data(), scalars.data(), count);
        gpu.sync();
    }
    for (bool split : {false, true, false}) {
        msm.set_split_buckets(split);
        point_t result;
        RustError status = invoke_msm(msm, result,
            resident ? device_points.data() : points.data(), count,
            resident ? device_scalars.data() : scalars.data(), resident, mont);
        std::string message = status.message ? status.message : "MSM failed";
        std::free(status.message);
        check(status.code == 0, message.c_str());
        check(result == expected, "MSM differs from CPU scalar multiplication");
        check(scalars == original, "MSM changed host scalars");
        const auto& stats = msm.split_statistics();
        uint64_t chunks = 0;
        for (auto value : stats.chunk_count) chunks += value;
        if (split && count > 256 && pattern < 2 && chunks == 0)
            throw std::runtime_error("concentrated MSM did not split a bucket: count="
                + std::to_string(count) + " pattern=" + std::to_string(pattern));
        for (unsigned w = 0; w < 32; ++w)
            if (split && stats.max_bucket[w] > 256)
                check(stats.chunk_count[w] > 0, "heavy bucket has no split chunks");
        if (!split) check(chunks == 0, "disabled split retained statistics");
    }
    if (resident) {
        gpu.DtoH(scalars, device_scalars.data());
        gpu.sync();
        check(scalars == original, "MSM changed device scalars");
    }
}

template<bool rotate>
static void check_division(const stream_t& stream, size_t count,
                           const fr_t& z, int grid = 0)
{
    std::vector<fr_t> input(count), expected(count), output(count);
    for (size_t i = 0; i < count; ++i) input[i] = fr_t(uint64_t(i * i + 17));
    fr_t carry(0);
    for (size_t i = count; i-- > 0;) {
        carry = input[i] + carry * z;
        expected[rotate ? (i ? i - 1 : count - 1) : i] = carry;
    }
    dev_ptr_t<fr_t> device(count, stream);
    if (count) stream.HtoD(device.data(), input);
    invoke_division(rotate, device.data(), count, z, stream, grid);
    if (count) stream.DtoH(output, device.data());
    stream.sync();
    check(output == expected, "division differs from CPU synthetic division");
}

static void division_tests(const gpu_t& gpu)
{
    for (size_t count : {0, 1, 2, 31, 32, 33, 1023, 1024, 1025,
                         4096, 6144, 65535, 65536, 65537}) {
        for (const fr_t& z : {fr_t(0), fr_t(1), fr_t(-fr_t(1)), fr_t(918273)}) {
            check_division<false>(gpu, count, z);
            check_division<true>(gpu, count, z);
        }
    }
    for (int grid : {1, 3, 4, 8}) {
        check_division<false>(gpu, 65537, fr_t(37), grid);
        check_division<true>(gpu, 65536, fr_t(37), grid);
    }
    std::atomic<unsigned> ready{0};
    std::exception_ptr errors[2];
    std::thread threads[2];
    for (unsigned i = 0; i < 2; ++i) {
        threads[i] = std::thread([&, i] {
            ++ready;
            while (ready.load() != 2) std::this_thread::yield();
            try {
                const auto& device = select_gpu(i % ngpus());
                stream_t stream(device.id());
                for (size_t count : {1025, 65536, 65537})
                    check_division<true>(stream, count, fr_t(uint64_t(19 + i)));
            } catch (...) { errors[i] = std::current_exception(); }
        });
    }
    for (auto& thread : threads) thread.join();
    for (auto& error : errors) if (error) std::rethrow_exception(error);
    gpu.select();
}

int main(int argc, char** argv)
{
    try {
        check(ngpus() > 0, "no supported CUDA device");
        const auto& gpu = select_gpu(0);
        division_tests(gpu);
        std::cout << "Division CPU parity and concurrent launches passed" << std::endl;
        const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
        for (size_t count : {255, 256, 257, 511, 512, 513, 65535, 65536, 65537, 131073}) {
            if (quick && count > 513) continue;
            for (unsigned pattern = 0; pattern < 3; ++pattern)
                check_msm(gpu, count, pattern, false, true);
        }
        for (unsigned pattern = 3; pattern < 7; ++pattern)
            check_msm(gpu, 4099, pattern, false, false);
        check_msm(gpu, quick ? 513 : 131073, 0, true, true);
        std::cout << "MSM CPU parity, split boundaries, G1 exceptional points, "
                     "host/device inputs and strategy reuse passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
