#!/usr/bin/env python3
"""Repair a freshly built torchvision wheel so its compiled extensions
are self-contained outside the build image.

The torchvision build links the build image's system image libraries
into image_stable.so — libjpeg.so.62 + libpng16.so.16 on the manylinux
builders (RHEL-family sonames). `pip wheel --no-build-isolation
--no-deps` does not vendor them, and unlike upstream — whose wheels are
auditwheel-repaired into torchvision.libs/ with hashed sonames — our
wheel ships bare NEEDED entries with no RPATH. On systems that do not
provide the linked soname the image extension then fails to load:

    image_stable.so: libjpeg.so.62: cannot open shared object file

observed when installing our x86_64 wheels on a vanilla Ubuntu machine
(Ubuntu ships libjpeg.so.8, a different soname). aarch64 users on
openEuler never hit it — it is RHEL-family and provides libjpeg.so.62.
Companion of repair_torch_wheel.py; the mechanism is identical, only
the discovery rules differ:

  1. discover: every ldd resolution of the wheel's .so files that
     - lands under a builder-only prefix (/opt/OpenBLAS, /acl,
       /opt/intel), or
     - matches a vendor-when-linked soname prefix (libjpeg.so.* — any
       soname, so both the RHEL libjpeg.so.62 and Ubuntu libjpeg.so.8
       flavors are covered — plus libpng16.so.16 and libz.so.1)
     is vendored; one follow-up ldd level pulls the vendored libs' own
     vendorable deps (libpng16 -> libz.so.1)
  2. copy them into torchvision.libs/ (same layout as upstream's
     packaging/wheel/relocate.py, keeps the package dir clean)
  3. set $ORIGIN RPATHs (patchelf) on every shipped .so
  4. verify every vendored soname now resolves from INSIDE the unpacked
     wheel via the $ORIGIN RPATH (this is what makes the wheel work on
     systems that lack the linked soname), then regenerate RECORD and
     repack under the same filename (platform tag unchanged)
  5. reinstall the repaired wheel
  6. verify: CDLL-load every shipped .so from the installed package
     with the builder-only prefixes stripped from LD_LIBRARY_PATH

Usage (from the CI workflow, after the torchvision wheel build):
    python3 repair_torchvision_wheel.py --wheels-dir /tmp/wheels
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
# Vendored when linked, by soname prefix. libjpeg matches any soname so
# both RHEL-family (libjpeg.so.62) and Ubuntu (libjpeg.so.8) builds are
# covered: the wheel carries the exact flavor it was compiled against.
DEFAULT_VENDOR_SONAME_PREFIXES = ("libjpeg.so", "libpng16.so", "libz.so.")

# Kept in one place so the verify subprocess runs the exact same probe as
# the validated inline version (prints are greppable in CI logs).
VERIFY_CODE = (
    "import ctypes, glob, os\n"
    "import torchvision\n"
    "pkg = os.path.dirname(torchvision.__file__)\n"
    "sos = [p for p in glob.glob(os.path.join(pkg, '*.so*'))]\n"
    "assert sos, 'no compiled extensions in installed torchvision'\n"
    "for p in sorted(sos):\n"
    "    ctypes.CDLL(p)\n"
    "    print(f'vision so load ok: {os.path.basename(p)}')\n"
    "print(f'clean-env torchvision ok: {torchvision.__version__}')\n"
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


def ldd(so: Path, clean_env: bool = False) -> str:
    """Run ldd on a shared object and return its stdout.

    clean_env=True strips LD_LIBRARY_PATH, simulating a target system:
    LD_LIBRARY_PATH takes precedence over DT_RUNPATH, so with it set the
    vendored in-wheel copies could be shadowed by build-environment
    paths and the structural check below would test the wrong thing.
    """
    env = None
    if clean_env:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ""
    result = subprocess.run(
        ["ldd", str(so)], capture_output=True, text=True, env=env
    )
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
    """Look a soname up in the builder-only lib dirs when ldd cannot
    resolve it (the build linked the lib by path without embedding an
    RPATH)."""
    for d in _builder_lib_dirs(builder_prefixes):
        candidate = d / soname
        if candidate.is_file():
            return candidate
    return None


def discover_libs_to_vendor(
    sos: List[Path],
    builder_prefixes: Tuple[str, ...],
    vendor_soname_prefixes: Tuple[str, ...],
) -> Dict[str, str]:
    """Map soname -> source path for every library that must be vendored.

    Two discovery rules, mirroring repair_torch_wheel.py:
      - any ldd resolution landing under a builder-only prefix
        (/opt/OpenBLAS, /acl, /opt/intel) — the target system cannot be
        assumed to provide these
      - sonames matching the vendor-when-linked prefixes (libjpeg.so.*,
        libpng16.so.16, libz.so.1) — the image libraries torchvision's
        extension links; the soname (not the path) decides, because the
        failure mode is a target system missing that soname entirely

    Follows one extra level so vendored libs can pull in their own
    vendorable deps (libpng16 -> libz.so.1).
    """
    to_vendor: Dict[str, str] = {}

    def consider(so: Path) -> None:
        for soname, resolved in parse_ldd(ldd(so)):
            if resolved == "not found":
                recovered = _recover_not_found(soname, builder_prefixes)
                if recovered is not None and soname not in to_vendor:
                    to_vendor[soname] = str(recovered)
                    print(f"  recovered: {soname} <- {recovered} (ldd could not resolve it)")
                else:
                    print(f"  warning: {soname} not found for {so.name}")
                continue
            if not resolved or soname in to_vendor:
                continue
            if resolved.startswith(builder_prefixes) or soname.startswith(
                vendor_soname_prefixes
            ):
                to_vendor[soname] = resolved

    for so in sos:
        consider(so)
    for src in list(to_vendor.values()):
        consider(Path(src))

    return to_vendor


def repair_wheel(
    wheel: Path,
    builder_prefixes: Tuple[str, ...],
    vendor_soname_prefixes: Tuple[str, ...],
) -> None:
    """Unpack, vendor, set RPATHs, verify resolution, RECORD, repack."""
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(td)
        unpacked = Path(td)
        pkg_dir = unpacked / "torchvision"
        libs_dir = unpacked / "torchvision.libs"

        # ---- 1. discover + vendor ---------------------------------
        sos = [p for p in pkg_dir.glob("*.so*") if p.is_file()]
        to_vendor = discover_libs_to_vendor(sos, builder_prefixes, vendor_soname_prefixes)

        if not to_vendor:
            print(
                "repair: no image libraries to vendor, "
                "wheel already self-contained"
            )
        else:
            print("repair: vendoring into torchvision.libs/:")
            libs_dir.mkdir(exist_ok=True)
            for soname, src in sorted(to_vendor.items()):
                dst = libs_dir / soname
                if Path(src).resolve() == dst.resolve():
                    # Idempotency: re-running on an already-repaired wheel
                    # makes ldd resolve the soname from the in-wheel copy
                    # in the unpack dir (src == dst). Keep it as is.
                    print(f"  {soname}  <-  already vendored (in-wheel copy kept)")
                    continue
                print(f"  {soname}  <-  {src}")
                shutil.copy2(src, dst)

        # ---- 2. $ORIGIN RPATHs -------------------------------------
        # The extensions live in torchvision/, the vendored libs in the
        # sibling torchvision.libs/ — same scheme as upstream's
        # relocate.py: extensions get $ORIGIN:$ORIGIN/../torchvision.libs,
        # and each vendored lib gets $ORIGIN so its own deps (libpng16 ->
        # libz.so.1) resolve from the same directory.
        for so in pkg_dir.glob("*.so*"):
            if so.is_file():
                subprocess.run(
                    [
                        "patchelf", "--set-rpath",
                        "$ORIGIN:$ORIGIN/../torchvision.libs", str(so),
                    ],
                    check=True,
                )
        if libs_dir.is_dir():
            for lib in libs_dir.glob("*.so*"):
                if lib.is_file():
                    subprocess.run(
                        ["patchelf", "--set-rpath", "$ORIGIN", str(lib)],
                        check=True,
                    )

        # ---- 3. structural verification ----------------------------
        # With the RPATH set, every vendored soname must now resolve from
        # inside the unpacked wheel. This is the property that makes the
        # wheel work on systems that lack the linked soname (Ubuntu
        # ships libjpeg.so.8, not .62) — check it here, fail the build
        # otherwise.
        for so in sos:
            for soname, resolved in parse_ldd(ldd(so, clean_env=True)):
                if soname in to_vendor and not resolved.startswith(str(unpacked)):
                    sys.exit(
                        f"repair: {soname} still resolves outside the wheel "
                        f"({resolved}) after vendoring — RPATH not effective?"
                    )
        if to_vendor:
            print("repair: vendored sonames resolve from inside the wheel")

        # ---- 4. regenerate RECORD ----------------------------------
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

        # ---- 5. repack under the same filename ---------------------
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
    """pip-install the repaired wheel so the environment holds the exact
    artifact the test jobs will receive."""
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--force-reinstall", "--no-deps", str(wheel),
        ],
        check=True,
    )


def verify(builder_prefixes: Tuple[str, ...]) -> None:
    """CDLL-load every shipped .so from the installed package with the
    builder-only prefixes stripped from LD_LIBRARY_PATH."""
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    clean = ":".join(
        p
        for p in ld_path.split(":")
        if p and not p.startswith(builder_prefixes)
    )
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = clean
    # cwd=/tmp: must not run from a source dir, where a local
    # torchvision/ package would shadow the installed one.
    subprocess.run(
        [sys.executable, "-c", VERIFY_CODE], env=env, cwd="/tmp", check=True
    )


def main() -> None:
    args = parse_args()
    builder_prefixes = tuple(args.builder_prefix)
    vendor_soname_prefixes = tuple(args.vendor_soname_prefix)

    ensure_patchelf()
    subprocess.run(["patchelf", "--version"], check=True)

    wheels = sorted(Path(args.wheels_dir).glob("torchvision-*.whl"))
    if len(wheels) != 1:
        sys.exit(
            f"repair: expected exactly one torchvision wheel in {args.wheels_dir}, "
            f"found {[w.name for w in wheels]}"
        )
    wheel = wheels[0]
    print(f"repair: {wheel.name}")

    repair_wheel(wheel, builder_prefixes, vendor_soname_prefixes)

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
        help="Directory holding the freshly built torchvision-*.whl (default: /tmp/wheels)",
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
        "--vendor-soname-prefix",
        action="append",
        default=None,
        metavar="SONAME",
        help="Soname prefix vendored when linked; repeatable. "
        "Default: libjpeg.so libpng16.so libz.so.",
    )
    parser.add_argument(
        "--skip-reinstall",
        action="store_true",
        help="Skip pip-installing the repaired wheel",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the clean-env load check",
    )
    args = parser.parse_args()
    if args.builder_prefix is None:
        args.builder_prefix = list(DEFAULT_BUILDER_PREFIXES)
    if args.vendor_soname_prefix is None:
        args.vendor_soname_prefix = list(DEFAULT_VENDOR_SONAME_PREFIXES)
    return args


if __name__ == "__main__":
    main()
