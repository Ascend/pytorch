#!/usr/bin/env bash
set -euo pipefail

# run_pytorch_coverage.sh - PyTorch source coverage collection for the sharded
# coverage pipeline (resolve -> build -> collect -> coverage -> merge).
#
#   shard mode: run ONE collected case shard under coverage. It reuses
#   run_npu_test_shard.py, so the case set, the per-case process isolation, the
#   worker count and the NPU card binding match the PR trigger pipeline
#   (receive-trigger.yml). The shard directory is already laid out exactly like
#   the merged output, so merging is a union of the shard directories. Every
#   group is named after its test file (the path the collected case list uses,
#   e.g. test/nn/test_linear.py), so nothing is flattened:
#
#     <shard>/                          one category shard, e.g. core-1
#     ├── test/nn/test_linear.py/covdata/coverage   per test file data
#     ├── test/nn/test_linear.py/covdata/FAILED     that file had failed case(s)
#     ├── logs/test/nn/test_linear.py.log           aggregated per-case logs
#     ├── logs/failed_cases.json                    this shard's failures
#     ├── _unmapped/covdata/coverage                data of no reported case
#     ├── reports/ ...                              internal runner material
#     ├── covdata-raw/                              per-case data, grouped here
#     │                                             and then deleted
#     └── run.log, .coveragerc, cov-plugin/         runner internals
#
#   merge mode: union every shard's <test file>/covdata + logs into the fixed
#   pytorch@latest/ directory, render coverage.xml and promote it atomically
#   (staging first, atomic promotion last). The caller adds the source snapshot
#   and publishes, so the final artifact is:
#
#     <artifact>/
#     ├── convstub/pytorch/
#     │   ├── pytorch/                  source snapshot (coverage sources)
#     │   └── test/                     test files of that snapshot
#     └── pytorch@latest/
#         ├── test/nn/test_linear.py/covdata/coverage   per test file data
#         ├── test/nn/test_linear.py/covdata/FAILED     that file had failures
#         ├── logs/test/nn/test_linear.py.log           per-case logs
#         ├── logs/failed_cases.json                    merged shard failures
#         ├── .coverage                 all shards combined
#         └── coverage.xml              torch-only report
#
# Usage:
#   ./run_pytorch_coverage.sh --cases-json <cases-shards/<cat>_cases_shard_<n>.json> \
#                            --test-dir <pytorch>/test --out-dir <dir> \
#                            [--max-workers 32] [--timeout 1200]
#   ./run_pytorch_coverage.sh --merge <dir1,dir2,...>
# Env:
#   OUT_ROOT    output root for merged artifacts (default <repo root>/outputs)
#   PYTHON_BIN  python interpreter (default python)
#
# Coverage measurement: this script writes a .coveragerc (branch/source=torch)
# and loads a pytest plugin (PYTEST_PLUGINS) in every worker process. The plugin
# starts coverage for each case's pytest session and, at session end, writes
# that case's own data file (covdata-raw/<nodeid>.coverage). Nothing depends on
# atexit or a process-level startup hook, which matters because the shard
# workers exit through os._exit(0).

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

test_dir="${GITHUB_WORKSPACE:-${repo_root}}/pytorch/test"
source_pkg="torch"
out_name="pytorch"
device_env="privateuse1"
timeout_seconds=1200
python_bin="${PYTHON_BIN:-python}"
cases_json=""
merge_dirs=""
out_dir=""
max_workers=32

while [ "$#" -gt 0 ]; do
  case "$1" in
    --cases-json)  cases_json="$2";      shift 2 ;;
    --merge)       merge_dirs="$2";      shift 2 ;;
    --test-dir)    test_dir="$2";        shift 2 ;;
    --out-dir)     out_dir="$2";         shift 2 ;;
    --max-workers) max_workers="$2";     shift 2 ;;
    --timeout)     timeout_seconds="$2"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *)  echo "Unexpected positional argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "${cases_json}" ] && [ -z "${merge_dirs}" ]; then
  echo "ERROR: use --cases-json (shard mode) or --merge <dirs> (merge mode)" >&2
  exit 2
fi
if [ -n "${cases_json}" ] && [ ! -d "${test_dir}" ]; then
  echo "ERROR: pytorch test dir not found: ${test_dir} (use --test-dir)" >&2
  exit 1
fi
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "ERROR: ${python_bin} not found in PATH" >&2
  exit 1
fi

out_root="${OUT_ROOT:-${repo_root}/outputs}"
overall_status=0

# Aggregate the per-case failures from the runner's own result file
# (report_dir/shard_<prefix>-<shard>_cases.json), no junit parsing needed.
write_failed_cases() {
  local report_dir="$1" out_file="$2"
  python3 - "$report_dir" "$out_file" <<'PYEOF'
import json
import sys
from pathlib import Path

report_dir = Path(sys.argv[1])
out_file = Path(sys.argv[2])
failed = []
total = 0
for results_file in sorted(report_dir.glob("shard_*_cases.json")):
    data = json.loads(results_file.read_text(encoding="utf-8"))
    total += data.get("total_cases", 0)
    for case in data.get("cases", []):
        status = case.get("status", "")
        if status in ("failed", "error", "timeout"):
            failed.append({"nodeid": case.get("nodeid", ""), "status": status})
out_file.write_text(
    json.dumps(
        {"total_cases": total, "total_failed": len(failed), "failed_cases": failed},
        indent=2,
    ),
    encoding="utf-8",
)
print(f"  failed cases: {len(failed)} of {total} -> {out_file}")
PYEOF
}

# Aggregate the runner's per-case logs (report_dir/cases_logs) into one log per
# test file, named after that test file, so the artifact keeps the file-level
# view: logs/<test file path>.log, e.g. logs/test/nn/test_linear.py.log.
write_file_logs() {
  local report_dir="$1" out_dir="$2"
  python3 - "$report_dir" "$out_dir" <<'PYEOF'
import json
import sys
from pathlib import Path

report_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

cases = []
for results_file in sorted(report_dir.glob("shard_*_cases.json")):
    data = json.loads(results_file.read_text(encoding="utf-8"))
    cases.extend(data.get("cases", []))


def sanitize(nodeid: str) -> str:
    safe = nodeid.replace("::", "_").replace("/", "_").replace("\\", "_")
    safe = safe.replace("(", "_").replace(")", "_").replace("[", "_").replace("]", "_")
    safe = safe.replace("<", "_lt_").replace(">", "_gt_")
    safe = safe.replace('"', "_quot_").replace("|", "_pipe_")
    safe = safe.replace("*", "_star_").replace("?", "_q_")
    safe = safe.replace(":", "_colon_")
    safe = safe.replace(" ", "_")
    safe = safe.replace(".", "_")
    while safe.startswith("_"):
        safe = safe[1:]
    while "__" in safe:
        safe = safe.replace("__", "_")
    if len(safe) > 200:
        safe = safe[:200]
    return safe or "unknown_case"


groups = {}
for case in cases:
    groups.setdefault(case.get("file") or "unknown", []).append(case)

logs_dir = report_dir / "cases_logs"
for file_path, file_cases in sorted(groups.items()):
    log_path = out_dir / f"{file_path}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"=== {file_path}: {len(file_cases)} case(s) ===\n")
        for case in file_cases:
            nodeid = case.get("nodeid", "")
            fh.write(f"\n----- {case.get('status', '?')}  {nodeid} -----\n")
            case_logs = sorted(logs_dir.glob(f"*_{sanitize(nodeid)}.log"))
            if case_logs:
                fh.write(case_logs[0].read_text(encoding="utf-8", errors="replace"))
            else:
                fh.write("(case log not found)\n")
print(f"  file logs: {len(groups)} -> {out_dir}")
PYEOF
}

# Group the per-case data files by test file, writing the layout the merged
# output needs (<test file path>/covdata/coverage) directly into the shard
# directory. The shard tarball then carries hundreds of files instead of tens of
# thousands, and the merge job only has to union them.
group_case_covdata() {
  local report_dir="$1" raw_dir="$2" shard_dir="$3"
  python3 - "$report_dir" "$raw_dir" "$shard_dir" <<'PYEOF'
import json
import sys
from pathlib import Path

import coverage

report_dir = Path(sys.argv[1])
raw_dir = Path(sys.argv[2])
shard_dir = Path(sys.argv[3])

UNMAPPED = "_unmapped"


def case_data_name(nodeid: str) -> str:
    safe = nodeid.replace("::", "_").replace("/", "_").replace("\\", "_")[:180]
    return f"{safe}.coverage"


def case_data_file(nodeid: str):
    # The plugin names the file after pytest's collected nodeid, which may or
    # may not carry the leading "test/" that the collected case list uses.
    alternatives = [nodeid]
    alternatives.append(nodeid[5:] if nodeid.startswith("test/") else f"test/{nodeid}")
    for candidate in alternatives:
        path = raw_dir / case_data_name(candidate)
        if path.is_file():
            return path
    return None


cases = []
for results_file in sorted(report_dir.glob("shard_*_cases.json")):
    data = json.loads(results_file.read_text(encoding="utf-8"))
    cases.extend(data.get("cases", []))

# Every group is named after its test file, exactly as the collected case list
# spells it (e.g. test/nn/test_linear.py), so the artifact mirrors the test tree.
groups = {}
failed_files = set()
claimed = set()
for case in cases:
    group = case.get("file") or "unknown"
    if case.get("status", "") in ("failed", "error", "timeout"):
        failed_files.add(group)
    data_file = case_data_file(case.get("nodeid", ""))
    if data_file is None:
        continue
    groups.setdefault(group, []).append(data_file)
    claimed.add(data_file)

unmapped = 0
for data_file in sorted(raw_dir.glob("*.coverage")):
    if data_file in claimed:
        continue
    groups.setdefault(UNMAPPED, []).append(data_file)
    unmapped += 1

written = 0
for group, files in sorted(groups.items()):
    group_dir = shard_dir / group / "covdata"
    group_dir.mkdir(parents=True, exist_ok=True)
    group_data = coverage.CoverageData(basename=str(group_dir / "coverage"))
    for path in files:
        src_data = coverage.CoverageData(basename=str(path))
        src_data.read()
        group_data.update(src_data)
    group_data.write()
    written += len(files)
    if group in failed_files:
        (group_dir / "FAILED").write_text("1\n", encoding="utf-8")

for group in sorted(failed_files - set(groups)):
    group_dir = shard_dir / group / "covdata"
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "FAILED").write_text("1\n", encoding="utf-8")

if unmapped:
    print(f"WARNING: {unmapped} coverage data file(s) not matched to any reported case", file=sys.stderr)
print(f"  grouped {written} case data file(s) into {len(groups)} test group(s)")
PYEOF
}

# Shard mode: one collected case shard through run_npu_test_shard.py (same case
# isolation, worker count and NPU card binding as the PR trigger pipeline) with
# coverage started in every worker/pytest subprocess.
run_case_shard() {
  if [ ! -f "${cases_json}" ]; then
    echo "ERROR: cases json not found: ${cases_json}" >&2
    exit 1
  fi

  local shard_dir="${out_dir:-${out_root}/shard}"
  local raw_cov_dir="${shard_dir}/covdata-raw"
  local report_dir="${shard_dir}/reports"
  local logs_dir="${shard_dir}/logs"
  rm -rf "${shard_dir}"
  mkdir -p "${raw_cov_dir}" "${report_dir}" "${logs_dir}"

  local rc_file="${shard_dir}/.coveragerc"
  cat > "${rc_file}" <<EOF
[run]
branch = True
source = ${source_pkg}
EOF

  # Coverage is measured per case: the plugin starts coverage when a case's
  # pytest session is configured and, at session end, writes that case's data to
  # its own file (covdata-raw/<nodeid>.coverage) before the worker exits through
  # os._exit(0) — so no atexit/process-level hook is needed. The per-case files
  # are grouped by test file once the shard finished (see group_case_covdata).
  local plugin_dir="${shard_dir}/cov-plugin"
  mkdir -p "${plugin_dir}"
  cat > "${plugin_dir}/zz_cov_plugin.py" <<'PYEOF'
import os
from pathlib import Path

_cov = None
_nodeid = ""


def pytest_configure(config):
    global _cov, _nodeid
    _nodeid = ""
    if not os.environ.get("COVERAGE_PROCESS_START"):
        return
    import coverage

    _cov = coverage.Coverage(config_file=os.environ["COVERAGE_PROCESS_START"])
    _cov.start()


def pytest_collection_modifyitems(session, config, items):
    global _nodeid
    # The runner's pytest command line puts flags before the nodeid
    # (--color=no -ra --tb=short <nodeid> ...), so the collected item is the
    # only reliable source of the case's own nodeid.
    if not _nodeid and items:
        _nodeid = items[0].nodeid


def pytest_sessionfinish(session, exitstatus):
    global _cov, _nodeid
    if _cov is None:
        return

    case_dir = os.environ.get("COVERAGE_CASE_DIR", "")
    items = list(getattr(session, "items", ()) or ())
    nodeid = _nodeid or (items[0].nodeid if items else "unknown")
    safe = nodeid.replace("::", "_").replace("/", "_").replace("\\", "_")[:180]

    _cov.stop()
    case = _cov
    _cov = None
    _nodeid = ""
    if not case_dir:
        return

    import coverage

    case_data = coverage.CoverageData(basename=str(Path(case_dir) / f"{safe}.coverage"))
    case_data.update(case.get_data())
    case_data.write()
PYEOF

  export PYTHONPATH="${plugin_dir}${PYTHONPATH:+:${PYTHONPATH}}"
  export PYTEST_PLUGINS="zz_cov_plugin${PYTEST_PLUGINS:+,${PYTEST_PLUGINS}}"
  export COVERAGE_PROCESS_START="${rc_file}"
  export COVERAGE_CASE_DIR="${raw_cov_dir}"

  echo "=== Cases: ${cases_json}  workers: ${max_workers}  timeout: ${timeout_seconds}s  device-env: ${device_env} ==="
  echo "=== Coverage rc: ${rc_file}  per-case data dir: ${raw_cov_dir}  pytest plugin: zz_cov_plugin ==="

  local status=0
  set +e
  "${python_bin}" -u "${repo_root}/.github/scripts/run_npu_test_shard.py" \
    --cases-json "${cases_json}" \
    --test-dir "${test_dir}" \
    --report-dir "${report_dir}" \
    --max-workers "${max_workers}" \
    --timeout "${timeout_seconds}" \
    --device-env "${device_env}" \
    --verbose 2>&1 | tee "${shard_dir}/run.log"
  status=${PIPESTATUS[0]}
  set -e

  write_failed_cases "${report_dir}" "${logs_dir}/failed_cases.json"
  write_file_logs "${report_dir}" "${logs_dir}"

  local case_files
  case_files="$(find "${raw_cov_dir}" -name '*.coverage' -type f | wc -l)"
  echo "=== Shard run done: exit=${status}, per-case coverage file(s)=${case_files} ==="
  group_case_covdata "${report_dir}" "${raw_cov_dir}" "${shard_dir}"
  rm -rf "${raw_cov_dir}"

  overall_status="${status}"
}

# Merge mode: combine every shard's coverage data + failure list into the fixed
# @latest dir (pytorch@latest/). Zero data files is a hard error so an empty run
# can never overwrite the previous @latest / OBS copy.
merge_coverage() {
  local staging="${out_root}/${out_name}@staging"
  local merged="${out_root}/${out_name}@latest"
  local -a dirs=()
  IFS=',' read -r -a dirs <<< "$1"

  rm -rf "${staging}"
  mkdir -p "${staging}/logs"

  local src
  for src in "${dirs[@]}"; do
    if [ ! -d "${src}" ]; then
      echo "WARNING: merge input not found: ${src}" >&2
      continue
    fi
    echo "=== Merge input: ${src} ==="
    # The shard already wrote the final layout (<test file>/covdata +
    # logs/<test file>.log + logs/failed_cases.json), so the merge is a union of
    # the shard dirs. A test file can span two shards (sharding slices cases, not
    # files), so the per-file logs append; failed_cases.json is regenerated below.
    local log_file log_name target
    while IFS= read -r log_file; do
      log_name="${log_file#"${src}/logs/"}"
      target="${staging}/logs/${log_name}"
      mkdir -p "$(dirname "${target}")"
      if [ -e "${target}" ]; then
        cat "${log_file}" >> "${target}"
      else
        cp -a "${log_file}" "${target}"
      fi
    done < <(find "${src}/logs" -type f -name '*.log' 2>/dev/null | sort)
  done

  local no_data=0
  if ! "${python_bin}" - "${staging}" "${dirs[@]}" <<'PYEOF'
import json
import sys
from pathlib import Path

import coverage

staging = Path(sys.argv[1])
shards = [Path(p) for p in sys.argv[2:]]

groups = {}
all_failed = []
total = 0
sources = []
failed_files = set()

for shard in shards:
    if not shard.is_dir():
        continue
    sources.append(shard.name)

    # Every shard already grouped its per-case data by test file and wrote the
    # final layout (<test file>/covdata/{coverage,FAILED}), so here it is only a
    # union across shards.
    for group_dir in sorted(shard.rglob("covdata")):
        if not group_dir.is_dir():
            continue
        group = group_dir.parent.relative_to(shard).as_posix()
        data_file = group_dir / "coverage"
        if data_file.is_file():
            groups.setdefault(group, []).append(data_file)
        if (group_dir / "FAILED").is_file():
            failed_files.add(group)

    shard_file = shard / "logs" / "failed_cases.json"
    if shard_file.is_file():
        data = json.loads(shard_file.read_text(encoding="utf-8"))
        total += data.get("total_cases", 0)
        all_failed.extend(data.get("failed_cases", []))

mapped_groups = [group for group in groups if not group.startswith("_unmapped")]
if groups and not mapped_groups:
    print(f"ERROR: {len(groups)} data group(s) exist but none maps to a reported test file", file=sys.stderr)
    sys.exit(1)
if len(mapped_groups) < len(groups):
    print(f"WARNING: {len(groups) - len(mapped_groups)} group(s) carry unmatched data", file=sys.stderr)

combined = coverage.CoverageData(basename=str(staging / ".coverage"))
written = 0
for group, files in sorted(groups.items()):
    group_dir = staging / group / "covdata"
    group_dir.mkdir(parents=True, exist_ok=True)
    group_data = coverage.CoverageData(basename=str(group_dir / "coverage"))
    for path in files:
        src_data = coverage.CoverageData(basename=str(path))
        src_data.read()
        group_data.update(src_data)
        combined.update(src_data)
    group_data.write()
    written += len(files)
    if group in failed_files:
        (group_dir / "FAILED").write_text("1\n", encoding="utf-8")

for group in sorted(failed_files - set(groups)):
    group_dir = staging / group / "covdata"
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "FAILED").write_text("1\n", encoding="utf-8")

out = staging / "logs" / "failed_cases.json"
out.write_text(
    json.dumps(
        {
            "total_cases": total,
            "total_failed": len(all_failed),
            "shards": sources,
            "failed_cases": all_failed,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"  merged failed cases: {len(all_failed)} of {total} -> {out}")

if written == 0:
    print(f"ERROR: no coverage data files found in: {[str(s) for s in shards]}", file=sys.stderr)
    sys.exit(1)

combined.write()
print(f"  per-file coverage: {written} group file(s) into {len(groups)} test group(s)")
PYEOF
  then
    no_data=1
  fi

  # Two patterns: the installed path as recorded during collection
  # (*/site-packages/torch/*) and the mapped source snapshot
  # (*/convstub/pytorch/pytorch/*) when the [paths] map is in effect.
  local include_patterns="*/${source_pkg}/*,*/${out_name}/*"
  if [ "${no_data}" -eq 0 ]; then
    if ! COVERAGE_FILE="${staging}/.coverage" "${python_bin}" -m coverage xml \
      --include="${include_patterns}" -o "${staging}/coverage.xml"; then
      no_data=1
      echo "ERROR: coverage xml failed (no ${source_pkg} data in the merged data file?)" >&2
    else
      echo "=== Combined coverage: ${staging}/.coverage, report: ${staging}/coverage.xml ==="
      COVERAGE_FILE="${staging}/.coverage" "${python_bin}" -m coverage report \
        --include="${include_patterns}" | tail -n 5 || true
    fi
  fi

  if [ "${no_data}" -eq 1 ]; then
    echo "ERROR: nothing to promote (no usable coverage data); previous ${merged} left untouched" >&2
    exit 3
  fi

  rm -rf "${merged}"
  mv "${staging}" "${merged}"
  echo "=== Promoted: ${merged} ==="
  overall_status=0
}

if [ -n "${merge_dirs}" ]; then
  merge_coverage "${merge_dirs}"
  exit "${overall_status}"
fi

run_case_shard
exit "${overall_status}"

