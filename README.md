# sppark

sppark (pronounced 'spark') is **S**upranational's **p**erformance **p**rimitives for **ar**guments of **k**nowledge such as SNARKs and STARKs. The library focuses on accelerating the most computationally expensive pieces of zero-knowledge proofs generation such as multi-scalar multiplication (MSM), number theoretic transform (NTT), arithmetic hashes, and more. The library is a collection of CUDA/C++ templates that can be instantiated for a range of finite fields and elliptic curves.

## Table of Contents

  * [Status](#status)
  * [General notes on implementation](#general-notes-on-implementation)
  * [Platform and Language Compatibility](#platform-and-language-compatibility)
  * [Introductory Integration Tutorial](#introductory-integration-tutorial)
    + [Multi-scalar Multiplication (MSM)](#multi-scalar-multiplication-msm)
  * [Repository Structure](#repository-structure)
  * [Performance](#performance)
  * [License](#license)

## Status

**This library is under active development [with a list of planned significant improvements]**

## Changes in this fork

This fork, branch `dev`, carries reusable acceleration primitives and
integration hooks for downstream provers.

- `SPPARK_NO_CXX_RUNTIME` (`util/exception.cuh`, `util/gpu_t.cuh`,
  `util/all_gpus.cpp`): a build mode in which `CUDA_OK` ends the process
  with the failing expression, location and CUDA error instead of throwing,
  `gpu_t` carries no thread pool, and the three container error hooks
  libstdc++'s headers call are defined weakly to abort. The archive then
  references nothing from libstdc++ and links into a libc++ program.
- `NTT::Base_dev_ptr_batch` (`ntt/ntt.cuh`, `ntt/kernels/*.cu`): a batch of
  vectors laid out a stride apart transformed in one launch sequence, the
  mixed-radix kernels taking the vector from the grid's second dimension
  and the permutation and coset passes looping over the batch. A batch of
  one is the unchanged single-vector path.

- `gpu_t::sync` (`util/gpu_t.cuh`) selects its device before synchronizing
  its streams, so a host thread that synchronizes every device in turn, as
  NTT parameter setup does, waits in each stream's own context.
- `SPPARK_MSM_PROFILE_START` and `SPPARK_MSM_PROFILE_END`
  (`msm/pippenger.cuh`): optional hooks around the digit breakdown and each
  bucket pass. Both default to nothing; an integrator defines them before
  including the header to time those kernels on their streams.
- `msm_t::set_split_buckets(true)` (`msm/pippenger.cuh`,
  `msm/split_buckets.cuh`): opt-in G1 bucket splitting for concentrated scalar
  distributions. Buckets longer than 256 points are accumulated as independent
  chunks and reduced before integration; zero digits and the lowest-window
  unit bucket retain their existing paths. Splitting defaults to off, and G2
  MSMs retain their existing implementation. Enabling splitting for an
  extension-field bucket reports `cudaErrorNotSupported`.
  `split_scratch_bytes(count, window, windows)` reports additional device
  scratch, and `split_statistics()` exposes counts and timings from the most
  recent invocation. The split path supports the existing 10–18-bit windows,
  up to 32 windows and up to `2^31` points, matching the signed digit index
  limit. Splitting does not transform caller scalars. Point-upload streams
  wait for the shared scratch allocation before using it.
- `div_by_x_minus_z` (`polynomial/div_by_x_minus_z.cuh`): reserves shared
  storage for every warp lane, queries kernel limits for the current device,
  and selects an exact cooperative tile or a single block to avoid a
  partial-tile carry race. Empty inputs are a no-op.

The runtime also supports `stream_t(id, cudaStream_t)` as a non-owning
wrapper. Its destruction leaves the caller's stream alive; the caller must
keep it alive through every operation enqueued through the wrapper. The id
is sppark's logical device index, which can differ from the CUDA ordinal.

### Primitive regression tests

`tests/cuda/run.py` builds a standalone BN254 MSM/division regression executable
and compile-checks the existing BN254 and BLS12-381 G1/G2 wrappers. It requires
CUDA, a compatible GPU, and an explicit BLST source root:

```sh
python3 tests/cuda/run.py --blst /path/to/blst --arch 120 --out /tmp/sppark-tests
```

Use `--compile-only` on a build host without a GPU. The tests compare MSMs
against CPU scalar multiplication and polynomial division against CPU synthetic
division. They cover split boundaries, multi-batch tails, signed and zero/unit
scalars, equal/inverse/infinity points, host and device operands, Montgomery and
canonical scalars, both division layouts, and concurrent stream launches. A
shorter run is available as `/tmp/sppark-tests/primitives --quick`, including for
`compute-sanitizer --tool memcheck` or `--tool racecheck`.

## General notes on implementation

The goal of the sppark library is to provide foundational components for applications and other libraries that require high-performance operations for zero-knowledge proofs generation.

## Platform and Language Compatibility

This library primarily supports x86_64 with Nvidia's Volta+ GPU hardware platforms on Linux and Windows operating systems. A limited support for AMD's RDNA and CDNA GPUs is provided. Non-GPU portions can be utilized even on ARM64, and additionally on Mac.

We show how to interface with Rust and Go. Caveat lector. Achieving highest possible GPU performance requires interfacing with target language memory management, possibly its async facilities, and might even require changes to object's data layout. These are hard to generalize and consequently are also a matter of discussion, likely on a case-by-case basis.

## Introductory Integration Tutorial

[TBD]

### Multi-scalar Multiplication (MSM)

[TBD]

## Repository Structure

**Root** - Contains various configuration files, documentation, licensing.
* **conversion** -
* **ec** - Contains templates for elliptic curve operations such as addition and doubling for different point representations.
* **ff** - Contains CUDA template[s] for finite field operations and instantiations of a variety of fields.
* **hash** -
* **memory** -
* **merkle** -
* **msm** - Contains multi-scalar multiplication template[s] that can be instantiated for a variety of elliptic curves such as BLS12-381 and the Pasta curves
* **ntt** - Contains NTT CUDA kernels.
* **poc** - Proof-of-concept implementations, including benchmarking.
* **rust** - Houses Rust crate definition.
* **util** - General-purpose helper classes.

## Performance

Simplified benchmark results can be collected by end users by exercising proof-of-concept applications. "Simplified" refers to the fact that there is always room for application-specific tuning. Intention is to give a general "taste." Just in case, benchmarks are likely to require high-end GPUs and one can't expect that they will execute on a laptop unmodified.

Caveat lector. As you compile PoC applications you might get warnings about not FFI-safe types. Alarming as they are, the fact that tests pass means that it works out nevertheless. However, this is **not** to say that they should be ignored and that one can proceed to build production code upon it. We intend to work with external software maintainers to resolve these warnings.

## License

The sppark library is licensed under the [Apache License Version 2.0](LICENSE) software license.
