#!/usr/bin/env python3
"""Build LLVM 20 as PIC archives, with the zlib and zstd it compresses with.

Everything lands in one prefix: what libddx links where no package supplies
the archives.  Does nothing when the prefix already holds LLVMConfig.cmake,
which is what lets a CI cache of the prefix stand in for the hour the build
takes.

Usage:
    python3 build_llvm.py <prefix>                # the machine's own target
    python3 build_llvm.py <prefix> --jobs 8
    python3 build_llvm.py <prefix> --targets "X86;AArch64"

Standard library only, deliberately: this runs before any environment exists,
inside a manylinux container or on a bare runner, with whatever python3 is to
hand.  CMake and Ninja are installed into that interpreter if the PATH lacks
them.
"""

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import urllib.request
from pathlib import Path

LLVM = "20.1.8"
ZLIB = "1.3.1"
ZSTD = "1.5.7"

_LLVM_BASE = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{LLVM}"
SOURCES = {
    f"zlib-{ZLIB}.tar.gz": (
        f"https://github.com/madler/zlib/releases/download/v{ZLIB}/zlib-{ZLIB}.tar.gz",
        "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23",
    ),
    f"zstd-{ZSTD}.tar.gz": (
        f"https://github.com/facebook/zstd/releases/download/v{ZSTD}/zstd-{ZSTD}.tar.gz",
        "eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3",
    ),
    f"llvm-{LLVM}.src.tar.xz": (
        f"{_LLVM_BASE}/llvm-{LLVM}.src.tar.xz",
        "e1363888216b455184dbb8a74a347bf5612f56a3f982369e1cba6c7e0726cde1",
    ),
    f"cmake-{LLVM}.src.tar.xz": (
        f"{_LLVM_BASE}/cmake-{LLVM}.src.tar.xz",
        "3319203cfd1172bbac50f06fa68e318af84dcb5d65353310c0586354069d6634",
    ),
    f"third-party-{LLVM}.src.tar.xz": (
        f"{_LLVM_BASE}/third-party-{LLVM}.src.tar.xz",
        "9a4e452a8163732d417db067a89190fcda823cb3aa33199e834ac7c028923f4b",
    ),
}

# The native target only: a wheel serves one machine, and every other target
# is code nobody loads.
LLVM_DEFINES = (
    "-DLLVM_ENABLE_PROJECTS=",
    "-DLLVM_BUILD_LLVM_DYLIB=OFF",
    "-DLLVM_LINK_LLVM_DYLIB=OFF",
    "-DLLVM_ENABLE_ZLIB=FORCE_ON",
    "-DZLIB_USE_STATIC_LIBS=ON",
    "-DLLVM_ENABLE_ZSTD=FORCE_ON",
    "-DLLVM_USE_STATIC_ZSTD=ON",
    "-DLLVM_ENABLE_FFI=OFF",
    "-DLLVM_ENABLE_LIBEDIT=OFF",
    "-DLLVM_ENABLE_LIBXML2=OFF",
    "-DLLVM_ENABLE_CURL=OFF",
    "-DLLVM_ENABLE_LIBPFM=OFF",
    "-DLLVM_ENABLE_BINDINGS=OFF",
    "-DLLVM_BUILD_TOOLS=OFF",
    "-DLLVM_INSTALL_UTILS=OFF",
    "-DLLVM_INCLUDE_TESTS=OFF",
    "-DLLVM_INCLUDE_EXAMPLES=OFF",
    "-DLLVM_INCLUDE_BENCHMARKS=OFF",
    "-DLLVM_INCLUDE_DOCS=OFF",
    "-DLLVM_ENABLE_ASSERTIONS=OFF",
)


def native_target() -> str:
    """Name the LLVM target that codegens for this machine."""
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "X86"
    if machine in {"arm64", "aarch64"}:
        return "AArch64"
    sys.exit(f"build_llvm: no LLVM target known for {machine}; pass --targets")


def run(*args: str) -> None:
    """Echo and run, failing loudly."""
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def ensure_tools() -> None:
    """CMake and Ninja from PyPI when the PATH has neither, onto this interpreter."""
    missing = [tool for tool in ("cmake", "ninja") if shutil.which(tool) is None]
    if not missing:
        return
    run(sys.executable, "-m", "pip", "install", "--quiet", *missing)
    scripts = sysconfig.get_path("scripts")
    os.environ["PATH"] = scripts + os.pathsep + os.environ["PATH"]


def fetch(work: Path, name: str) -> None:
    """Download one pinned source, check its digest, unpack it under work."""
    url, sha256 = SOURCES[name]
    archive = work / name
    print(f"fetching {name}", flush=True)
    with urllib.request.urlopen(url) as response, archive.open("wb") as out:
        shutil.copyfileobj(response, out)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != sha256:
        sys.exit(f"build_llvm: {name}: sha256 {digest}, expected {sha256}")
    with tarfile.open(archive) as tar:
        tar.extractall(work)


def cmake_install(
    source: Path, build: Path, prefix: Path, jobs: int, *defines: str
) -> None:
    """Configure, build and install one CMake project: Release, PIC."""
    run(
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        *defines,
    )
    run("cmake", "--build", str(build), "-j", str(jobs), "--target", "install")


def build(prefix: Path, jobs: int, targets: str) -> None:
    """Build zlib, then zstd, then LLVM against both, into prefix."""
    work = Path(tempfile.mkdtemp(prefix="build_llvm."))
    try:
        for name in SOURCES:
            fetch(work, name)
        # A standalone llvm/ looks for its siblings under their unversioned names.
        (work / f"cmake-{LLVM}.src").rename(work / "cmake")
        (work / f"third-party-{LLVM}.src").rename(work / "third-party")

        # zlib's CMake builds the shared library unconditionally; it is removed
        # afterwards so that only the archive is there to be found.
        cmake_install(
            work / f"zlib-{ZLIB}",
            work / "zlib-build",
            prefix,
            jobs,
            "-DZLIB_BUILD_EXAMPLES=OFF",
        )
        shared = ("lib/libz.so*", "lib/libz*.dylib*", "bin/*zlib*.dll", "lib/zlib.lib")
        for pattern in shared:
            for path in prefix.glob(pattern):
                path.unlink()

        cmake_install(
            work / f"zstd-{ZSTD}" / "build" / "cmake",
            work / "zstd-build",
            prefix,
            jobs,
            "-DZSTD_BUILD_STATIC=ON",
            "-DZSTD_BUILD_SHARED=OFF",
            "-DZSTD_BUILD_PROGRAMS=OFF",
            "-DZSTD_BUILD_TESTS=OFF",
        )

        # The archives end up inside libddx, so PIC throughout.  LLVMConfig
        # records the zlib and zstd it found, so the prefix on CMAKE_PREFIX_PATH
        # is what makes an LLVM consumer find the same static pair.
        cmake_install(
            work / f"llvm-{LLVM}.src",
            work / "llvm-build",
            prefix,
            jobs,
            f"-DCMAKE_PREFIX_PATH={prefix}",
            f"-DLLVM_TARGETS_TO_BUILD={targets}",
            *LLVM_DEFINES,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    """Parse the prefix and options; skip the build when it is already there."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "prefix", type=Path, help="where the archives and LLVMConfig.cmake go"
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--targets",
        default=native_target(),
        help="LLVM_TARGETS_TO_BUILD (default: this machine's)",
    )
    args = parser.parse_args()
    prefix = args.prefix.resolve()

    if (prefix / "lib" / "cmake" / "llvm" / "LLVMConfig.cmake").is_file():
        print(f"build_llvm: {prefix} already holds LLVM {LLVM}")
        return
    ensure_tools()
    build(prefix, args.jobs, args.targets)
    print(f"build_llvm: LLVM {LLVM}, zlib {ZLIB}, zstd {ZSTD} under {prefix}")


if __name__ == "__main__":
    main()
