#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check torch_npu's COMPAT (multi-version compatibility) implementation.

Checks:
  R1. Version consistency: MIN_SUPPORTED_VERSION must be identical across
      torch_npu/_compat/version.py, torch_npu/csrc/_compat/version.h, and the
      minimum major.minor in version.txt (the supported build matrix); no
      COMPAT(>= X.Y) threshold may exceed the maximum version in version.txt.
  R2. No stale compat blocks: any version branch (if CURRENT_VERSION >= X.Y /
      #if TORCH_NPU_VERSION_GE(X, Y)) whose threshold is <= the effective
      supported floor (version.txt minimum) is dead code and should already
      have been removed.
  R3. Every version branch must carry a COMPAT(>= X.Y) comment, and the branch
      must match it exactly: same threshold number, and a switching boundary of
      >= X.Y (operators '>' / '<=' shift the boundary by one and are rejected).
      A COMPAT(>= X.Y) comment with no matching version branch is reported too.

Report is printed to stdout; a txt file is written only when --report <path> is provided.
Findings are reported as warnings; exit code is always 0 (non-blocking).
"""


import argparse
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PY_VERSION_PATH = REPO_ROOT / "torch_npu" / "_compat" / "version.py"
CPP_VERSION_PATH = REPO_ROOT / "torch_npu" / "csrc" / "_compat" / "version.h"
PY_COMPAT_DIR = REPO_ROOT / "torch_npu" / "_compat"
CPP_COMPAT_DIR = REPO_ROOT / "torch_npu" / "csrc" / "_compat"
VERSION_TXT_PATH = REPO_ROOT / "version.txt"

PY_MIN_RE = re.compile(r"MIN_SUPPORTED_VERSION\s*[:=].*?\(\s*(\d+)\s*,\s*(\d+)\s*\)")
CPP_MAJOR_RE = re.compile(r"TORCH_NPU_MIN_SUPPORTED_MAJOR\s+(\d+)")
CPP_MINOR_RE = re.compile(r"TORCH_NPU_MIN_SUPPORTED_MINOR\s+(\d+)")

COMPAT_RE = re.compile(r"COMPAT\(>=\s*(\d+)\.(\d+)\)")

PY_IF_RE = re.compile(
    r"CURRENT_VERSION\s*(?P<op>>=|>|<|<=)\s*\(\s*(?P<maj>\d+)\s*,\s*(?P<min>\d+)\s*\)"
)
CPP_IF_RE = re.compile(
    r"#if\s+TORCH_NPU_VERSION_GE\(\s*(?P<maj>\d+)\s*,\s*(?P<min>\d+)\s*\)"
)


def read(path):
    return path.read_text(encoding="utf-8").splitlines()


def parse_py_min(path):
    for line in read(path):
        m = PY_MIN_RE.search(line)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    return None


def parse_cpp_min(path):
    maj = None
    min_ = None
    for line in read(path):
        m = CPP_MAJOR_RE.search(line)
        if m:
            maj = int(m.group(1))
        m = CPP_MINOR_RE.search(line)
        if m:
            min_ = int(m.group(1))
    if maj is not None and min_ is not None:
        return (maj, min_)
    return None


def parse_version_txt(path):
    versions = []
    for line in read(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(".")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            versions.append((int(parts[0]), int(parts[1])))
    return sorted(set(versions))


def scan_blocks(lines, if_re, min_supported, max_supported, issues, path):
    rel = path.relative_to(REPO_ROOT)
    pending = None  # (threshold, comment_line_no)
    for i, line in enumerate(lines):
        cm = COMPAT_RE.search(line)
        if cm:
            pending = ((int(cm.group(1)), int(cm.group(2))), i + 1)
        m = if_re.search(line)
        if not m:
            continue
        gd = m.groupdict()
        op = gd.get("op", ">=")
        code_thr = (int(gd["maj"]), int(gd["min"]))
        if pending is None:
            issues.append((rel, i + 1, "warning",
                           f"version branch {op} {code_thr[0]}.{code_thr[1]} "
                           f"has no COMPAT(>= X.Y) comment"))
            continue
        compat_thr, compat_line = pending
        if compat_thr != code_thr:
            issues.append((rel, i + 1, "warning",
                           f"COMPAT comment (line {compat_line}) says >= "
                           f"{compat_thr[0]}.{compat_thr[1]} but code checks "
                           f"{op} {code_thr[0]}.{code_thr[1]}"))
        elif op in (">", "<="):
            issues.append((rel, i + 1, "warning",
                           f"branch uses {op} {code_thr[0]}.{code_thr[1]} but "
                           f"COMPAT(>= {compat_thr[0]}.{compat_thr[1]}) implies >= "
                           f"(off-by-one at the version boundary)"))
        # staleness / unsupported are judged on the CODE threshold, not the comment
        if min_supported is not None and min_supported >= code_thr:
            issues.append((rel, i + 1, "warning",
                           f"stale compat block: supported floor "
                           f"{min_supported[0]}.{min_supported[1]} >= COMPAT threshold "
                           f"{code_thr[0]}.{code_thr[1]}; remove it"))
        if max_supported is not None and code_thr > max_supported:
            issues.append((rel, i + 1, "warning",
                           f"COMPAT threshold {code_thr[0]}.{code_thr[1]} exceeds "
                           f"version.txt max {max_supported[0]}.{max_supported[1]} "
                           f"(compat for an unsupported version)"))
        pending = None
    if pending is not None:
        pthr, pline = pending
        issues.append((rel, pline, "warning",
                       f"COMPAT(>= {pthr[0]}.{pthr[1]}) comment has no matching "
                       f"version branch"))


def main():
    ap = argparse.ArgumentParser(description="Check torch_npu COMPAT implementation")
    ap.add_argument("--report", default=None,
                    help="path to write a txt report (only written when provided)")
    args = ap.parse_args()

    issues = []

    # R1: version consistency across version.py / version.h / version.txt
    py_min = parse_py_min(PY_VERSION_PATH)
    cpp_min = parse_cpp_min(CPP_VERSION_PATH)
    if py_min is None or cpp_min is None:
        issues.append((PY_VERSION_PATH.relative_to(REPO_ROOT), 0, "warning",
                       f"cannot parse MIN_SUPPORTED_VERSION from "
                       f"{PY_VERSION_PATH} / {CPP_VERSION_PATH}"))
    elif py_min != cpp_min:
        issues.append((PY_VERSION_PATH.relative_to(REPO_ROOT), 0, "warning",
                       f"Python MIN_SUPPORTED_VERSION {py_min} != C++ "
                       f"TORCH_NPU_MIN_SUPPORTED {cpp_min}"))

    vtxt = parse_version_txt(VERSION_TXT_PATH)
    if not vtxt:
        issues.append((VERSION_TXT_PATH.relative_to(REPO_ROOT), 0, "warning",
                       "cannot parse any version from version.txt"))
    elif py_min is not None and py_min != vtxt[0]:
        issues.append((PY_VERSION_PATH.relative_to(REPO_ROOT), 0, "warning",
                       f"Python MIN_SUPPORTED_VERSION {py_min} != version.txt "
                       f"minimum {vtxt[0]}"))

    # Effective floor/ceiling come from the declared build matrix (version.txt),
    # which is what actually gets built and exercised.
    min_supported = vtxt[0] if vtxt else py_min
    max_supported = vtxt[-1] if vtxt else None

    if min_supported is not None:
        for path in sorted(PY_COMPAT_DIR.glob("*.py")):
            scan_blocks(read(path), PY_IF_RE, min_supported, max_supported, issues, path)
        for path in sorted(CPP_COMPAT_DIR.glob("*")):
            if path.suffix not in (".h", ".hpp", ".cpp"):
                continue
            scan_blocks(read(path), CPP_IF_RE, min_supported, max_supported, issues, path)

    # build report lines
    body = []
    for rel, lineno, sev, msg in issues:
        body.append(f"{rel}:{lineno}: {sev}: {msg}")
    summary = f"{len(issues)} issue(s) found" if issues else "no issues found"
    body.append("")
    body.append(f"summary: {summary}")

    for line in body:
        print(line)

    if args.report:
        min_str = f"{min_supported[0]}.{min_supported[1]}" if min_supported else "unknown"
        report_path = Path(args.report)
        report_path.write_text(
            "\n".join([
                f"# torch_npu check_compat report @ {datetime.now().isoformat()}",
                f"# repo: {REPO_ROOT}",
                f"# MIN_SUPPORTED_VERSION: {min_str}",
                "",
                *body,
            ]),
            encoding="utf-8",
        )
        print(f"report written to {report_path}")

    return 0


if __name__ == "__main__":
    main()
