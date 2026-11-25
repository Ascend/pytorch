#!/usr/bin/env python3
"""Prepare pinned converter headers without CANN or a GE build tree."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
MANIFEST = ROOT / "converter_deps.json"
HEADER_SUFFIXES = {".h", ".hpp", ".hh", ".hxx", ".inc", ".inl", ".def"}


def fingerprint(revisions):
    digest = hashlib.sha256()
    for path in (Path(__file__), MANIFEST):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    for name, revision in sorted(revisions.items()):
        digest.update(name.encode())
        digest.update(revision.encode())
    manifest = json.loads(MANIFEST.read_text())
    metadef = ROOT / manifest["headers"]["metadef"]["path"]
    for path in sorted(path for path in metadef.rglob("*") if path.is_file()):
        digest.update(path.relative_to(metadef).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def tracked_submodule_revision(spec):
    path = spec["submodule"]
    entry = subprocess.check_output(
        ["git", "ls-files", "-s", "--", path], cwd=REPO_ROOT, text=True
    ).strip()
    fields = entry.split()
    if len(fields) < 3 or fields[0] != "160000":
        raise RuntimeError(f"Expected {path} to be a tracked submodule")
    return fields[1]


def checked_out_submodule(spec, name, revision):
    source = REPO_ROOT / spec["submodule"]
    if not source.is_dir():
        raise RuntimeError(
            f"{name} submodule is missing at {source}; run "
            f"git submodule update --init --recursive {spec['submodule']}"
        )
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"{name} submodule is not initialized at {source}") from error
    if actual != revision:
        raise RuntimeError(
            f"{name} submodule revision mismatch: expected {revision}, found {actual}"
        )
    return source


def copy_headers(source, destination):
    for path in source.rglob("*"):
        if path.is_file() and path.suffix in HEADER_SUFFIXES:
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def copy_licenses(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for pattern in ("LICENSE*", "COPYING*", "NOTICE*"):
        for path in source.glob(pattern):
            if path.is_file():
                shutil.copyfile(path, destination / path.name)


def prepare_ge(spec, stage, revision):
    source = checked_out_submodule(spec, "GE", revision)
    destination = stage / "ge"
    # Preserve the GE header layout for the converter and link stubs.
    for directory in ("inc", "base", "graph_metadef", "runtime/v2"):
        if not (source / directory).is_dir():
            raise RuntimeError(f"GE submodule is missing required header directory {directory}")
        copy_headers(source / directory, destination / "include" / directory)
    # These declarations only use LoweringGlobalData by pointer or reference.
    # Avoid pulling the full task protobuf into every tiling translation unit.
    for header, include in (
        ("runtime/v2/kernel/tiling_cache.h", "lowering/lowering_global_data.h"),
        ("inc/graph_metadef/register/node_converter_registry.h", "exe_graph/lowering/lowering_global_data.h"),
        ("inc/graph_metadef/register/ffts_node_converter_registry.h", "exe_graph/lowering/lowering_global_data.h"),
    ):
        path = destination / "include" / header
        text = path.read_text()
        directive = f'#include "{include}"'
        if text.count(directive) != 1 or text.count("namespace gert {") != 1:
            raise RuntimeError(f"Unexpected LoweringGlobalData declaration in {header}")
        text = text.replace(directive, "", 1)
        text = text.replace("namespace gert {", "namespace gert {\nclass LoweringGlobalData;", 1)
        path.write_text(text)
    copy_licenses(source, destination)
    (destination / "REVISION").write_text(revision + "\n")
    return source


def prepare_cann_headers(specs, stage, ge, revisions):
    """Assemble GE/runtime headers from submodules and pinned metadef headers."""
    destination = stage / "cann"
    include = destination / "include"
    runtime = checked_out_submodule(specs["runtime"], "runtime", revisions["runtime"])
    metadef = ROOT / specs["metadef"]["path"]
    revision_file = metadef / "REVISION"
    if not revision_file.is_file():
        raise RuntimeError(f"Vendored metadef revision file is missing: {revision_file}")
    actual_revision = revision_file.read_text().strip()
    if actual_revision != revisions["metadef"]:
        raise RuntimeError(
            f"Vendored metadef revision mismatch: expected {revisions['metadef']}, found {actual_revision}"
        )
    for directory in (runtime / "include/external", runtime / "include/dfx", runtime / "pkg_inc"):
        if not directory.is_dir():
            raise RuntimeError(f"runtime submodule is missing required header directory {directory}")
    copy_headers(runtime / "include" / "external", include)
    copy_headers(runtime / "include" / "dfx", include)
    copy_headers(runtime / "pkg_inc", destination / "pkg_inc")
    for relative in specs["metadef"]["headers"]:
        if relative.startswith("include/external/ge_common/"):
            suffix = relative[len("include/external/ge_common/"):]
            source = metadef / "inc" / "external" / "ge_common" / suffix
            target = include / "external" / "ge_common" / suffix
        elif relative.startswith("include/"):
            source = metadef / "inc" / "external" / relative[len("include/"):]
            target = include / relative[len("include/"):]
        elif relative.startswith("pkg_inc/"):
            source = metadef / relative
            target = destination / relative
        else:
            raise RuntimeError(f"Unsupported metadef header path {relative}")
        if not source.is_file():
            raise RuntimeError(
                f"metadef {specs['metadef']['revision']} is missing required header {relative}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # GE owns Graph/GNode/ES and ACL compiler headers. Like torch-npu, copy its
    # external headers last so GE's ACL extensions take precedence over runtime.
    copy_headers(ge / "inc" / "graph_metadef" / "external", include)
    copy_headers(ge / "inc" / "graph_metadef" / "external" / "ge_common", include / "external" / "ge_common")
    copy_headers(ge / "inc" / "external", include)
    copy_headers(ge / "inc" / "external" / "ge" / "eager_style_graph_builder" / "cpp", include / "ge")
    copy_headers(runtime / "pkg_inc" / "mmpa", include / "mmpa")
    copy_headers(runtime / "pkg_inc" / "platform", include / "platform")
    copy_licenses(runtime, destination / "licenses" / "runtime")
    copy_licenses(metadef, destination / "licenses" / "metadef")


def prepare():
    specs = json.loads(MANIFEST.read_text())
    revisions = {
        "ge": tracked_submodule_revision(specs["ge"]),
        "runtime": tracked_submodule_revision(specs["headers"]["runtime"]),
        "metadef": specs["headers"]["metadef"]["revision"],
    }
    stamp = ROOT / ".converter-deps.json"
    signature = fingerprint(revisions)
    if stamp.exists():
        previous = json.loads(stamp.read_text())
        if previous.get("fingerprint") == signature and previous.get("revisions") == revisions and all(
            (ROOT / name).is_file() for name in previous.get("files", [])
        ):
            print("[converter deps] Using prepared third_party/ge and cann headers", flush=True)
            return
    for program in ("git",):
        if shutil.which(program) is None:
            raise RuntimeError(f"Preparing converter dependencies requires {program}")
    with tempfile.TemporaryDirectory(prefix=".converter-deps-", dir=ROOT) as temporary:
        work = Path(temporary)
        stage = work / "prepared"
        stage.mkdir()
        ge = prepare_ge(specs["ge"], stage, revisions["ge"])
        prepare_cann_headers(specs["headers"], stage, ge, revisions)
        # Commit only complete dependency trees; failed validation leaves the previous
        # preparation intact and does not create a valid cache stamp.
        stamp.unlink(missing_ok=True)
        for name in ("ge", "cann"):
            target = ROOT / name
            if target.exists():
                shutil.rmtree(target)
            (stage / name).replace(target)
        files = sorted(str(path.relative_to(ROOT)) for name in ("ge", "cann")
                       for path in (ROOT / name).rglob("*") if path.is_file())
        stamp.write_text(
            json.dumps({"fingerprint": fingerprint(revisions), "revisions": revisions, "files": files}, indent=2)
            + "\n"
        )
    print("[converter deps] Pinned converter headers ready", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    # CMake and wheel builds may configure concurrently in the same checkout.
    with (ROOT / ".converter-deps.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepare()
