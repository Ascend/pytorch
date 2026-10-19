#!/usr/bin/env python3
"""Repair a freshly built torch wheel so it is self-contained outside the
build image.

Mirrors upstream pytorch's .ci/wheel/linux/repair_wheel.py for the
aarch64 CPU case. A build image may provide shared libraries on private
paths (e.g. /opt/OpenBLAS in the torch-npu builder image); the torch
build links them dynamically into libtorch_cpu.so, but
`pip wheel --no-build-isolation` does not vendor them into the wheel, so
`import torch` fails in any environment that lacks those paths (run
38041281352: every test file collected 0 cases with
"ImportError: libopenblas.so.0").

The repair:

  1. discover builder-only libs: every ldd resolution of the wheel's .so
     files landing under a builder-only prefix (/opt/OpenBLAS and /acl on
     the aarch64 builder, /opt/intel MKL on the x86_64 builder), plus the
     always-vendor set (libgfortran.so.5, libgomp.so.1) when linked, plus
     sonames ldd reports as "not found" but that exist in the
     builder-only lib dirs (linked without an RPATH)
  2. copy them into torch/lib/
  3. set $ORIGIN RPATHs on every shipped .so (patchelf), same scheme as
     upstream: torch/*.so* -> "$ORIGIN:$ORIGIN/lib", torch/lib/*.so* ->
     "$ORIGIN"
  4. regenerate RECORD
  5. repack under the same filename (platform tag unchanged)
  6. reinstall the repaired wheel (pip --force-reinstall --no-deps) so
     downstream builds (torchvision, torch_npu) link against the exact
     artifact the test image will receive
  7. verify: import torch + one matmul with the builder-only prefixes
     stripped from LD_LIBRARY_PATH (simulates the test image). This
     closes the "build-time self-check passes inside the builder" blind
     spot — if the repair failed to vendor a needed lib, the job stops
     here.

A no-op for build images without builder-only prefixes (e.g. test-image
builds): nothing is vendored and the RPATH/RECORD pass is behaviour-
neutral. Validated end-to-end by manual run 38890857638.

Usage (from the CI workflow, after the torch wheel build):
    python3 repair_torch_wheel.py --wheels-dir /tmp/wheels
"""

import argparse
import base64
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DEFAULT_BUILDER_PREFIXES = ("/opt/OpenBLAS", "/acl", "/opt/intel")
DEFAULT_ALWAYS_VENDOR = ("libgfortran.so.5", "libgomp.so.1")

# Kept in one place so the verify subprocess runs the exact same probe as
# the validated inline version (prints are greppable in CI logs).
VERIFY_CODE = (
    "import torch\n"
    "print(f'clean-env torch import ok: {torch.__version__}')\n"
    "x = torch.randn(8, 8)\n"
    "print(f'clean-env matmul ok: {float((x @ x).sum())}')\n"
)


def ensure_patchelf() -> None:
    """Use the image's patchelf; fall back to the PyPI prebuilt binary."""
    if shutil.which("patchelf"):
        return
    print("repair: patchelf not found on PATH, installing from PyPI")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "patchelf"], check=True
    )
    if not shutil.which("patchelf"):
        sys.exit("repair: patchelf unavailable even after pip install")


def ldd(so: Path) -> str:
    """Run ldd on a shared object and return its stdout."""
    result = subprocess.run(["ldd", str(so)], capture_output=True, text=True)
    return result.stdout


def parse_ldd(output: str) -> Iterable[Tuple[str, str]]:
    """Yield (soname, resolved_path) pairs from ldd output."""
    for line in output.splitlines():
        if "=>" not in line:
            continue
        left, right = line.split("=>", 1)
        soname = left.strip()
        resolved = right.split("(", 1)[0].strip()
        yield soname, resolved


def _builder_lib_dirs(builder_prefixes: Tuple[str, ...]) -> List[Path]:
    """Candidate library directories under the builder-only prefixes."""
    dirs: List[Path] = []
    for prefix in builder_prefixes:
        root = Path(prefix)
        dirs.extend(root / sub for sub in ("lib", "lib64", "build"))
        dirs.append(root)
    return dirs


def _recover_not_found(
    soname: str, builder_prefixes: Tuple[str, ...]
) -> Path | None:
    """
    Look a soname up in the builder-only lib dirs when ldd cannot resolve
    it (the build linked the lib by path without embedding an RPATH, and
    the builder image does not put it on LD_LIBRARY_PATH either — e.g.
    MKL under /opt/intel on the x86_64 builder). The vendored copy in
    torch/lib is found at runtime through the $ORIGIN RPATHs set below,
    so recovery makes the wheel self-contained where ldd alone would
    have failed the clean-env verify.
    """
    for d in _builder_lib_dirs(builder_prefixes):
        candidate = d / soname
        if candidate.is_file():
            return candidate
    return None


def discover_libs_to_vendor(
    sos: List[Path],
    builder_prefixes: Tuple[str, ...],
    always_vendor: Tuple[str, ...],
) -> Dict[str, str]:
    """
    Map soname -> source path for every library that must be vendored.

    Three discovery rules, mirroring upstream repair_wheel.py:
      - any ldd resolution landing under a builder-only prefix
        (/opt/OpenBLAS, /acl, /opt/intel) — the test image cannot
        provide these
      - the always-vendor sonames (libgfortran.so.5, libgomp.so.1) when
        anything links them — upstream bundles these for aarch64 CPU
        builds even though they resolve in system dirs
      - sonames ldd reports as "not found" are looked up in the
        builder-only lib dirs and recovered from there when present

    Follows one extra level so vendored libs can pull in their own deps
    (OpenBLAS -> libgfortran/libgomp).
    """
    to_vendor: Dict[str, str] = {}

    for so in sos:
        for soname, resolved in parse_ldd(ldd(so)):
            if resolved == "not found":
                recovered = _recover_not_found(soname, builder_prefixes)
                if recovered is not None and soname not in to_vendor:
                    to_vendor[soname] = str(recovered)
                    print(f"  recovered: {soname} <- {recovered} (ldd could not resolve it)")
                else:
                    print(f"  warning: {soname} not found for {so.name}")
                continue
            if not resolved:
                continue
            if resolved.startswith(builder_prefixes) and soname not in to_vendor:
                to_vendor[soname] = resolved
            elif soname in always_vendor and soname not in to_vendor:
                to_vendor[soname] = resolved

    for src in list(to_vendor.values()):
        for soname, resolved in parse_ldd(ldd(Path(src))):
            if resolved == "not found":
                recovered = _recover_not_found(soname, builder_prefixes)
                if recovered is not None and soname not in to_vendor:
                    to_vendor[soname] = str(recovered)
                    print(f"  recovered: {soname} <- {recovered} (ldd could not resolve it)")
                else:
                    print(f"  warning: {soname} not found for {src}")
                continue
            if not resolved or soname in to_vendor:
                continue
            if resolved.startswith(builder_prefixes) or soname in always_vendor:
                to_vendor[soname] = resolved

    return to_vendor


def repair_wheel(
    wheel: Path,
    builder_prefixes: Tuple[str, ...],
    always_vendor: Tuple[str, ...],
) -> None:
    """Unpack, vendor, set RPATHs, regenerate RECORD, repack in place."""
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(td)
        unpacked = Path(td)
        torch_dir = unpacked / "torch"
        torch_lib = torch_dir / "lib"

        # ---- 1. discover + vendor ---------------------------------
        sos = [
            p
            for p in list(torch_dir.glob("*.so*")) + list(torch_lib.glob("*.so*"))
            if p.is_file()
        ]
        to_vendor = discover_libs_to_vendor(sos, builder_prefixes, always_vendor)

        if not to_vendor:
            print(
                "repair: no builder-only dependencies found, "
                "wheel already self-contained"
            )
        else:
            print("repair: vendoring into torch/lib:")
            for soname, src in sorted(to_vendor.items()):
                dst = torch_lib / soname
                if Path(src).resolve() == dst.resolve():
                    # Idempotency: re-running on an already-repaired wheel
                    # makes ldd resolve the soname from the in-wheel copy
                    # in the unpack dir (src == dst). Keep it as is.
                    print(f"  {soname}  <-  already vendored (in-wheel copy kept)")
                    continue
                print(f"  {soname}  <-  {src}")
                shutil.copy2(src, dst)

        # ---- 2. $ORIGIN RPATHs (upstream repair_wheel.py scheme) --
        for so in torch_dir.glob("*.so*"):
            if so.is_file():
                subprocess.run(
                    ["patchelf", "--set-rpath", "$ORIGIN:$ORIGIN/lib", str(so)],
                    check=True,
                )
        for so in torch_lib.glob("*.so*"):
            if so.is_file():
                subprocess.run(
                    ["patchelf", "--set-rpath", "$ORIGIN", str(so)], check=True
                )

        # ---- 3. regenerate RECORD ---------------------------------
        dist_infos = list(unpacked.glob("*.dist-info"))
        if len(dist_infos) != 1:
            sys.exit(
                "repair: expected one .dist-info in wheel, found "
                f"{[p.name for p in dist_infos]}"
            )
        record = dist_infos[0] / "RECORD"
        rows = []
        for f in sorted(unpacked.rglob("*")):
            if not f.is_file() or f == record:
                continue
            rel = f.relative_to(unpacked).as_posix()
            data = f.read_bytes()
            digest = base64.urlsafe_b64encode(
                hashlib.sha256(data).digest()
            ).rstrip(b"=").decode()
            rows.append((rel, f"sha256={digest}", str(len(data))))
        rows.append((record.relative_to(unpacked).as_posix(), "", ""))
        with record.open("w", newline="") as fh:
            csv.writer(fh).writerows(rows)

        # ---- 4. repack under the same filename --------------------
        new_wheel = wheel.with_name(wheel.name + ".repaired")
        with zipfile.ZipFile(
            new_wheel, "w", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zf:
            for f in sorted(unpacked.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(unpacked).as_posix())
        new_wheel.replace(wheel)
        print(f"repair: repacked {wheel.name} ({wheel.stat().st_size / 1e6:.1f} MB)")


def reinstall(wheel: Path) -> None:
    """pip-install the repaired wheel so downstream builds (torchvision,
    torch_npu) link against the exact artifact the test image receives."""
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--force-reinstall", "--no-deps", str(wheel),
        ],
        check=True,
    )


def verify(builder_prefixes: Tuple[str, ...]) -> None:
    """Import torch + one matmul with the builder-only prefixes stripped
    from LD_LIBRARY_PATH, simulating the test image."""
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    clean = ":".join(
        p
        for p in ld_path.split(":")
        if p and not p.startswith(builder_prefixes)
    )
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = clean
    # cwd=/tmp: must not run from the pytorch source dir, where a local
    # torch/ package would shadow the installed one.
    subprocess.run(
        [sys.executable, "-c", VERIFY_CODE], env=env, cwd="/tmp", check=True
    )


def main() -> None:
    args = parse_args()
    builder_prefixes = tuple(args.builder_prefix)
    always_vendor = tuple(args.always_vendor)

    ensure_patchelf()
    subprocess.run(["patchelf", "--version"], check=True)

    wheels = sorted(Path(args.wheels_dir).glob("torch-*.whl"))
    if len(wheels) != 1:
        sys.exit(
            f"repair: expected exactly one torch wheel in {args.wheels_dir}, "
            f"found {[w.name for w in wheels]}"
        )
    wheel = wheels[0]
    print(f"repair: {wheel.name}")

    repair_wheel(wheel, builder_prefixes, always_vendor)

    if not args.skip_reinstall:
        reinstall(wheel)
    if not args.skip_verify:
        verify(builder_prefixes)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--wheels-dir",
        default="/tmp/wheels",
        help="Directory holding the freshly built torch-*.whl (default: /tmp/wheels)",
    )
    parser.add_argument(
        "--builder-prefix",
        action="append",
        default=None,
        metavar="PATH",
        help="Build-image-only library prefix to vendor; repeatable. "
        "Default: /opt/OpenBLAS /acl /opt/intel",
    )
    parser.add_argument(
        "--always-vendor",
        action="append",
        default=None,
        metavar="SONAME",
        help="Soname always vendored when linked; repeatable. "
        "Default: libgfortran.so.5 libgomp.so.1",
    )
    parser.add_argument(
        "--skip-reinstall",
        action="store_true",
        help="Skip pip-installing the repaired wheel",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the clean-env import check",
    )
    args = parser.parse_args()
    if args.builder_prefix is None:
        args.builder_prefix = list(DEFAULT_BUILDER_PREFIXES)
    if args.always_vendor is None:
        args.always_vendor = list(DEFAULT_ALWAYS_VENDOR)
    return args


if __name__ == "__main__":
    main()
