#!/usr/bin/env python3
"""Check converter imports without loading artifacts or requiring installed CANN."""

import argparse
import ctypes
from pathlib import Path
import subprocess


def symbols(nm, path, defined):
    output = subprocess.check_output(
        [nm, "-D", "--format=posix", "--defined-only" if defined else "--undefined-only", str(path)],
        text=True,
    )
    # Undefined weak symbols are optional; strong imports have type U.
    return {parts[0].split("@")[0] for row in output.splitlines()
            if len(parts := row.split()) >= 2 and (defined or parts[1] == "U")}


def python_provides(name):
    # CMake runs this check with the build's Python. Its C API can come from
    # the executable itself, so do not require a separate libpython file.
    return name.startswith(("Py", "_Py")) and hasattr(ctypes.pythonapi, name)


def check(args):
    providers = set(args.provider)
    for root in args.provider_dir:
        providers.update(path.resolve() for path in root.glob("*.so*") if path.is_file())
    # Query runtime SONAMEs: GCC toolsets may make libstdc++.so a GNU ld
    # script (with nonshared archives), which nm -D cannot read.
    for name in ("libstdc++.so.6", "libgcc_s.so.1", "libc.so.6", "libm.so.6", "libdl.so.2",
                 "libpthread.so.0", "librt.so.1", "ld-linux-x86-64.so.2", "ld-linux-aarch64.so.1"):
        found = subprocess.check_output([args.compiler, "-print-file-name=" + name], text=True).strip()
        if Path(found).is_file():
            providers.add(Path(found).resolve())
    provided = set()
    for provider in providers:
        provided.update(symbols(args.nm, provider, True))
    for artifact in args.artifact:
        exported = symbols(args.nm, artifact, True)
        if "fxrt_cann_stub_marker" in exported:
            raise RuntimeError(f"CANN stub implementation leaked into {artifact}")
        dynamic = subprocess.check_output(["readelf", "-d", str(artifact)], text=True)
        for line in dynamic.splitlines():
            if ("(RPATH)" in line or "(RUNPATH)" in line) and (
                "cann_stub" in line or "/third_party/cann" in line
            ):
                raise RuntimeError(f"Build-only CANN path leaked into {artifact}: {line}")
        missing = {name for name in symbols(args.nm, artifact, False) - provided
                   if not python_provides(name)}
        if missing:
            names = subprocess.check_output(["c++filt"], input="\n".join(sorted(missing)), text=True)
            raise RuntimeError(f"Unprovided imports in {artifact}; update CANN stubs if adding APIs:\n{names}")
    print("[CANN stub check] All imports provided; no stub implementation or runtime search path leaked")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--provider", type=Path, action="append", default=[])
    parser.add_argument("--provider-dir", type=Path, action="append", default=[])
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--nm", default="nm")
    check(parser.parse_args())
