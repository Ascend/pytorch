#!/usr/bin/env bash
set -euo pipefail

# run_pytorch_coverage.sh - PyTorch source coverage collection for the sharded
# coverage pipeline (resolve -> build -> collect -> coverage -> merge).
#
#   shard mode: run ONE collected case shard under coverage. It reuses
#   run_npu_test_shard.py, so the case set, the per-case process isolation, the
#   worker count and the NPU card binding match the PR trigger pipeline
#   (receive-trigger.yml).
#
#   merge mode: combine every shard's coverage data files into the fixed
#   torch@latest/ directory (staging first, atomic promotion last).
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
# that case's own data file (covdata/<nodeid>.coverage). Nothing depends on
# atexit or a process-level startup hook, which matters because the shard
# workers exit through os._exit(0).

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

test_dir="${GITHUB_WORKSPACE:-${repo_root}}/pytorch/test"
source_pkg="torch"
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
# test file, named <n>-<flat>.log, so the artifact keeps the file-level view.
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
shard_type = "regular"
for results_file in sorted(report_dir.glob("shard_*_cases.json")):
    data = json.loads(results_file.read_text(encoding="utf-8"))
    shard_type = data.get("shard_type", shard_type)
    cases.extend(data.get("cases", []))


def flat_name(file_path: str) -> str:
    name = file_path[5:] if file_path.startswith("test/") else file_path
    if name.endswith(".py"):
        name = name[:-3]
    return f"{shard_type}__{name.replace('/', '__')}"


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
    groups.setdefault(case.get("file", "unknown"), []).append(case)

logs_dir = report_dir / "cases_logs"
for index, (file_path, file_cases) in enumerate(sorted(groups.items()), 1):
    log_path = out_dir / f"{index}-{flat_name(file_path)}.log"
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

# Shard mode: one collected case shard through run_npu_test_shard.py (same case
# isolation, worker count and NPU card binding as the PR trigger pipeline) with
# coverage started in every worker/pytest subprocess.
run_case_shard() {
  if [ ! -f "${cases_json}" ]; then
    echo "ERROR: cases json not found: ${cases_json}" >&2
    exit 1
  fi

  local shard_dir="${out_dir:-${out_root}/shard}"
  local cov_dir="${shard_dir}/covdata"
  local report_dir="${shard_dir}/reports"
  rm -rf "${shard_dir}"
  mkdir -p "${cov_dir}" "${report_dir}"

  local rc_file="${shard_dir}/.coveragerc"
  cat > "${rc_file}" <<EOF
[run]
branch = True
source = ${source_pkg}
EOF

  # Coverage is measured per case: the plugin starts coverage when a case's
  # pytest session is configured and, at session end, writes that case's data to
  # its own file (covdata/<nodeid>.coverage) before the worker exits through
  # os._exit(0) — so no atexit/process-level hook is needed.
  local plugin_dir="${shard_dir}/cov-plugin"
  mkdir -p "${plugin_dir}"
  cat > "${plugin_dir}/zz_cov_plugin.py" <<'PYEOF'
import os
from pathlib import Path

_cov = None


def pytest_configure(config):
    global _cov
    if not os.environ.get("COVERAGE_PROCESS_START"):
        return
    import coverage

    _cov = coverage.Coverage(config_file=os.environ["COVERAGE_PROCESS_START"])
    _cov.start()


def pytest_sessionfinish(session, exitstatus):
    global _cov
    if _cov is None:
        return

    case_dir = os.environ.get("COVERAGE_CASE_DIR", "")
    invocation = getattr(getattr(session, "config", None), "invocation_params", None)
    args = list(getattr(invocation, "args", ()) or ())
    nodeid = args[0] if args else "unknown"
    safe = nodeid.replace("::", "_").replace("/", "_").replace("\\", "_")[:180]

    _cov.stop()
    case = _cov
    _cov = None
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
  export COVERAGE_CASE_DIR="${cov_dir}"

  echo "=== Cases: ${cases_json}  workers: ${max_workers}  timeout: ${timeout_seconds}s  device-env: ${device_env} ==="
  echo "=== Coverage rc: ${rc_file}  per-case data dir: ${cov_dir}  pytest plugin: zz_cov_plugin ==="

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

  write_failed_cases "${report_dir}" "${shard_dir}/failed_cases.json"
  write_file_logs "${report_dir}" "${shard_dir}/file_logs"

  echo "=== Shard done: exit=${status}, per-case coverage file(s)=$(find "${cov_dir}" -name '*.coverage' -type f | wc -l) ==="
  overall_status="${status}"
}

# Merge mode: combine every shard's coverage data + failure list into the fixed
# @latest dir. Zero data files is a hard error so an empty run can never
# overwrite the previous @latest / OBS copy.
merge_coverage() {
  local staging="${out_root}/${source_pkg}@staging"
  local merged="${out_root}/${source_pkg}@latest"
  local -a dirs=()
  IFS=',' read -r -a dirs <<< "$1"

  rm -rf "${staging}"
  mkdir -p "${staging}/logs"

  local src shard_name
  for src in "${dirs[@]}"; do
    if [ ! -d "${src}" ]; then
      echo "WARNING: merge input not found: ${src}" >&2
      continue
    fi
    shard_name="$(basename "${src}")"
    echo "=== Merge input: ${src} ==="
    if [ -f "${src}/run.log" ]; then
      cp "${src}/run.log" "${staging}/logs/${shard_name}.log"
    fi
    if [ -d "${src}/file_logs" ]; then
      cp -a "${src}/file_logs/." "${staging}/logs/"
    fi
  done

  local no_data=0
  if ! "${python_bin}" - "${staging}" "${dirs[@]}" <<'PYEOF'
import json
import sys
from pathlib import Path

import coverage

staging = Path(sys.argv[1])
shards = [Path(p) for p in sys.argv[2:]]


def flat_name(shard_type: str, file_path: str) -> str:
    name = file_path[5:] if file_path.startswith("test/") else file_path
    if name.endswith(".py"):
        name = name[:-3]
    return f"{shard_type}__{name.replace('/', '__')}"


def case_data_name(nodeid: str) -> str:
    safe = nodeid.replace("::", "_").replace("/", "_").replace("\\", "_")[:180]
    return f"{safe}.coverage"


groups = {}
all_failed = []
total = 0
sources = []
failed_files = set()
unmapped = 0

for shard in shards:
    if not shard.is_dir():
        continue
    sources.append(shard.name)
    shard_type = "regular"
    cases = []
    for results_file in sorted(shard.glob("reports/shard_*_cases.json")):
        data = json.loads(results_file.read_text(encoding="utf-8"))
        shard_type = data.get("shard_type", shard_type)
        cases.extend(data.get("cases", []))

    claimed = set()
    for case in cases:
        flat = flat_name(shard_type, case.get("file", "unknown"))
        if case.get("status", "") in ("failed", "error", "timeout"):
            failed_files.add(flat)
        data_file = shard / "covdata" / case_data_name(case.get("nodeid", ""))
        if not data_file.is_file():
            continue
        groups.setdefault(flat, []).append(data_file)
        claimed.add(data_file)

    cov_dir = shard / "covdata"
    if cov_dir.is_dir():
        for data_file in sorted(cov_dir.glob("*.coverage")):
            if data_file not in claimed:
                groups.setdefault(f"zz_unmapped__{shard.name}", []).append(data_file)
                unmapped += 1

    shard_file = shard / "failed_cases.json"
    if shard_file.is_file():
        data = json.loads(shard_file.read_text(encoding="utf-8"))
        total += data.get("total_cases", 0)
        all_failed.extend(data.get("failed_cases", []))

combined = coverage.CoverageData(basename=str(staging / ".coverage"))
written = 0
for flat, files in sorted(groups.items()):
    group_dir = staging / flat / "covdata"
    group_dir.mkdir(parents=True, exist_ok=True)
    group_data = coverage.CoverageData(basename=str(group_dir / "coverage"))
    for path in files:
        src_data = coverage.CoverageData(basename=str(path))
        src_data.read()
        group_data.update(src_data)
        combined.update(src_data)
    group_data.write()
    written += len(files)
    if flat in failed_files:
        (group_dir / "FAILED").write_text("1\n", encoding="utf-8")

for flat in sorted(failed_files - set(groups)):
    group_dir = staging / flat / "covdata"
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "FAILED").write_text("1\n", encoding="utf-8")

if unmapped:
    print(f"WARNING: {unmapped} coverage data file(s) not matched to any reported case", file=sys.stderr)

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
print(f"  per-file coverage: {written} data file(s) into {len(groups)} test group(s)")
PYEOF
  then
    no_data=1
  fi

  if [ "${no_data}" -eq 0 ]; then
    if ! COVERAGE_FILE="${staging}/.coverage" "${python_bin}" -m coverage xml \
      --include="*/${source_pkg}/*" -o "${staging}/coverage.xml"; then
      no_data=1
      echo "ERROR: coverage xml failed (no ${source_pkg} data in the merged data file?)" >&2
    else
      echo "=== Combined coverage: ${staging}/.coverage, report: ${staging}/coverage.xml ==="
      COVERAGE_FILE="${staging}/.coverage" "${python_bin}" -m coverage report \
        --include="*/${source_pkg}/*" | tail -n 5 || true
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

