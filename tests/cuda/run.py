#!/usr/bin/env python3
"""Build and run focused CUDA MSM and polynomial-division regressions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shlex
import subprocess


def run(command):
    print(shlex.join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blst", required=True, type=Path,
                        help="BLST source root containing src/server.c and build/assembly.S")
    parser.add_argument("--arch", required=True, help="CUDA architecture, e.g. 80 or 120")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    blst = args.blst.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    nvcc = os.environ.get("NVCC", "nvcc")
    cc = os.environ.get("CC", "cc")
    flags = [nvcc, "--std=c++17", "-O2", "-lineinfo", f"-arch=sm_{args.arch}",
             "-D__BLST_PORTABLE__", f"-I{root}", f"-I{blst / 'src'}", "-diag-suppress=177"]
    commands = [
        [*flags, "-DFEATURE_BN254", "-c", root / "tests/cuda/primitives.cu",
         "-o", output / "primitives.o"],
        [*flags, "-DFEATURE_BN254", "-c", root / "poc/msm-cuda/cuda/pippenger_inf.cu",
         "-o", output / "bn254-g1-g2.o"],
        [*flags, "-DFEATURE_BLS12_381", "-c", root / "poc/msm-cuda/cuda/pippenger_inf.cu",
         "-o", output / "bls12-381-g1-g2.o"],
        [cc, "-O2", "-D__BLST_PORTABLE__", "-c", blst / "src/server.c",
         "-o", output / "blst-server.o"],
        [cc, "-D__BLST_PORTABLE__", "-c", blst / "build/assembly.S",
         "-o", output / "blst-assembly.o"],
    ]
    with ThreadPoolExecutor() as pool:
        list(pool.map(run, commands))
    executable = output / "primitives"
    run([nvcc, output / "primitives.o", output / "blst-server.o",
         output / "blst-assembly.o", "-o", executable, "-lpthread"])
    if not args.compile_only:
        run([executable])


if __name__ == "__main__":
    main()
