#!/usr/bin/env bash
set -euo pipefail

# run_pytorch_coverage.sh - PyTorch source coverage collection.
# Called only by .github/workflows/pytorch_coverage_nightly.yml. Runs the PR
# trigger case set (receive-trigger.yml): pr_test_whitelist.yml files, filtered
# by hw_classification and minus the pr_skip_list.jsonl nodeids, and collects
# coverage of the installed torch python sources.
#
# Usage:
#   # shard mode: run one pre-collected case shard (receive-trigger style jobs)
#   ./run_pytorch_coverage.sh --cases-json <cases-shards/<cat>_cases_shard_<n>.json> \
#                            --test-dir <pytorch>/test --out-dir <dir> \
#                            [--max-workers 32] [--timeout 1200]
#   # file mode: run the pr_test_whitelist.yml files directly (local/debug)
#   ./run_pytorch_coverage.sh --test-dir <pytorch>/test \
#                            [--category c1,c2] [--timeout <seconds>] \
#                            [--skip-list <path>] [--no-promote] \
#                            [--hw-classification ACCELERATOR]
#   # merge mode: combine every shard's coverage data into torch@latest/
#   ./run_pytorch_coverage.sh --merge <dir1,dir2,...>
# Env:
#   OUT_ROOT    output root for all artifacts (default <repo root>/outputs)
#   PYTHON_BIN  python interpreter (default python)
#
# Shard mode reuses run_npu_test_shard.py (same case isolation, worker count and
# NPU card binding as the PR trigger pipeline) and measures coverage of the
# installed torch package through COVERAGE_PROCESS_START + a site-packages
# startup hook, so every worker/pytest subprocess writes its own data file.
# Merge mode combines those files, writes coverage.xml and aggregates failures.
#
# Outputs land under OUT_ROOT (single copy, crash-safe):
#   convstub/torch/            source snapshot (matches the run env)
#   torch@latest/              per-test covdata dirs (+ FAILED markers), logs,
#                              failed_cases.json, combined .coverage/coverage.xml
# Each run writes to torch@staging/ and promotes it to torch@latest/ only
# after the full flow completed — a crashed run keeps the previous data.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

config_file="${repo_root}/.github/config/pr_test_whitelist.yml"
skip_list_file="${repo_root}/.github/config/pr_skip_list.jsonl"
category_filter=""
test_dir="${GITHUB_WORKSPACE:-${repo_root}}/pytorch/test"
source_pkg="torch"
device_env="privateuse1"
hw_classification="ACCELERATOR"
timeout_seconds=600
python_bin="${PYTHON_BIN:-python}"
cases_json=""
merge_dirs=""
out_dir=""
max_workers=32
promote=1

while [ "$#" -gt 0 ]; do
  case "$1" in
    --category) category_filter="$2"; shift 2 ;;
    --test-dir) test_dir="$2";        shift 2 ;;
    --timeout)  timeout_seconds="$2"; shift 2 ;;
    --skip-list) skip_list_file="$2"; shift 2 ;;
    --hw-classification) hw_classification="$2"; shift 2 ;;
    --cases-json) cases_json="$2";    shift 2 ;;
    --merge)    merge_dirs="$2";      shift 2 ;;
    --out-dir)  out_dir="$2";         shift 2 ;;
    --max-workers) max_workers="$2";  shift 2 ;;
    --no-promote) promote=0;          shift ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *)  echo "Unexpected positional argument: $1 (targets come from ${config_file})" >&2; exit 2 ;;
  esac
done

if [ -z "${cases_json}" ] && [ -z "${merge_dirs}" ] && [ ! -f "${config_file}" ]; then
  echo "ERROR: whitelist config not found: ${config_file}" >&2
  exit 1
fi
if [ ! -d "${test_dir}" ]; then
  echo "ERROR: pytorch test dir not found: ${test_dir} (use --test-dir)" >&2
  exit 1
fi
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "ERROR: ${python_bin} not found in PATH" >&2
  exit 1
fi

out_root="${OUT_ROOT:-${repo_root}/outputs}"
src_snapshot_dir="${out_root}/convstub/${source_pkg}"
# Single-copy output with crash safety: all artifacts are written to a
# staging dir first and promoted (atomic mv) to the fixed @latest dir only
# after the run completed end-to-end — a crashed/killed run leaves the
# previous @latest data intact.
staging_dir="${out_root}/${source_pkg}@staging"
covdata_root="${out_root}/${source_pkg}@latest"
rm -rf "${staging_dir}"
log_dir="${staging_dir}/logs"
mkdir -p "${log_dir}"

# Emit "category<TAB>file" lines from the whitelist. Legacy flat
# "whitelist:" format is grouped under "regular".
read_targets() {
  python3 - "$1" "$2" <<'PYEOF'
import sys
try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")
data = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
cat_filter = set(sys.argv[2].split(",")) if sys.argv[2] else None
cats = data.get("categories")
if cats:
    for name, cfg in cats.items():
        if cat_filter and name not in cat_filter:
            continue
        for f in (cfg or {}).get("files", []):
            print(f"{name}\t{f}")
elif "whitelist" in data:
    for f in data["whitelist"]:
        print(f"regular\t{f}")
PYEOF
}

targets=()
while IFS=$'\t' read -r cat file; do
  [ -n "${cat:-}" ] && targets+=("${cat}|${file}")
done < <(read_targets "${config_file}" "${category_filter}")

if [ "${#targets[@]}" -eq 0 ] && [ -z "${cases_json}" ] && [ -z "${merge_dirs}" ]; then
  echo "ERROR: no test targets resolved from ${config_file} (category filter: '${category_filter}')" >&2
  exit 1
fi

echo "=== Targets: ${#targets[@]} file(s) from ${config_file} ==="
echo "=== test-dir: ${test_dir}  source: ${source_pkg}  device-env: ${device_env}  hw-classification: ${hw_classification} ==="

skips_tsv=""
if [ -n "${skip_list_file}" ] && [ -f "${skip_list_file}" ]; then
  skips_tsv="$(mktemp)"
  python3 - "${skip_list_file}" > "${skips_tsv}" <<'PYEOF'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        nodeid = obj.get("nodeid", "")
        if not nodeid:
            continue
        if nodeid.startswith("test/"):
            nodeid = nodeid[5:]
        print(f"{nodeid.split('::', 1)[0]}\t{nodeid}")
PYEOF
  echo "=== Skip list: $(wc -l < "${skips_tsv}") nodeid(s) from ${skip_list_file} ==="
else
  echo "WARNING: skip list not found: ${skip_list_file}, no case filtering applied" >&2
fi

test_results=()
failed_logs=()
test_index=0
overall_status=0

# Snapshot the installed torch sources + the test dir so report-time sources
# match the coverage data exactly.
dump_source_snapshot() {
  mkdir -p "${src_snapshot_dir}"

  local torch_path
  torch_path="$("${python_bin}" -c 'import torch, os; print(os.path.dirname(torch.__file__))' 2>/dev/null || true)"
  if [ -n "${torch_path}" ] && [ -d "${torch_path}" ]; then
    echo "=== Dumping torch source: ${torch_path} -> ${src_snapshot_dir}/torch ==="
    rm -rf "${src_snapshot_dir}/torch"
    cp -a "${torch_path}" "${src_snapshot_dir}/torch"
  else
    echo "WARNING: torch install path not found, skip source snapshot" >&2
  fi

  echo "=== Dumping test source: ${test_dir} -> ${src_snapshot_dir}/test ==="
  rm -rf "${src_snapshot_dir}/test"
  cp -a "${test_dir}" "${src_snapshot_dir}/test"
}

# Flatten "category + test file" into a valid name: strip .py, / -> __.
# Example: core|test/nn/test_dropout.py -> core__test__nn__test_dropout
flatten_name() {
  local n="$2"; n="${n%.py}"; n="${n//\//__}"
  printf '%s__%s' "$1" "$n"
}

setup_coverage() {
  local covdata_dir="${covdata_root}/$1/covdata"
  mkdir -p "${covdata_dir}"
  export COVERAGE_FILE="${covdata_dir}/coverage"
  echo "  COVERAGE_FILE: ${COVERAGE_FILE}"
}

# Per-test env: autoload the torch_npu backend (registers privateuse1),
# instantiate device-parameterized classes for the NPU device only, and make
# sibling module imports resolve.
setup_env() {
  local td; td="$(dirname "${1#test/}")"
  export TORCH_DEVICE_BACKEND_AUTOLOAD=1
  export PYTORCH_TESTING_DEVICE_ONLY_FOR="${device_env}"
  export PYTHONPATH="${test_dir}:${test_dir}/${td}${PYTHONPATH:+:${PYTHONPATH}}"
}

# TERM then KILL a process group: distributed spawn workers and inductor
# compile subprocesses may outlive the main process and hold NPU resources.
terminate_process_group() {
  local pg="$1"
  if ! kill -0 -- "-${pg}" 2>/dev/null; then return; fi
  kill -TERM -- "-${pg}" 2>/dev/null || true
  for _ in {1..10}; do
    if ! kill -0 -- "-${pg}" 2>/dev/null; then return; fi
    sleep 0.5
  done
  kill -KILL -- "-${pg}" 2>/dev/null || true
}

# Run a command into a log file (tailed live); setsid gives it its own
# process group so the whole group can be cleaned up on exit/timeout.
run_logged_command() {
  local log_file="$1"; shift
  local command_pid process_group="" tail_pid
  : > "${log_file}"
  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" > "${log_file}" 2>&1 &
    command_pid=$!
    process_group="${command_pid}"
  else
    "$@" > "${log_file}" 2>&1 &
    command_pid=$!
  fi
  tail --pid="${command_pid}" -n +1 -f "${log_file}" &
  tail_pid=$!
  wait "${command_pid}"
  local status=$?
  if [ -n "${process_group}" ]; then terminate_process_group "${process_group}"; fi
  wait "${tail_pid}" || true
  return "${status}"
}

run_test_target() {
  local category="$1" file="$2"
  test_index=$((test_index + 1))
  local flat log_file
  flat="$(flatten_name "${category}" "${file}")"
  log_file="${log_dir}/${test_index}-${flat}.log"
  echo "=== [${category}] Running target: ${file} ==="

  local -a skipped_nodeids=()
  if [ -n "${skips_tsv}" ]; then
    mapfile -t skipped_nodeids < <(awk -F'\t' -v f="${file#test/}" '$1 == f {print $2}' "${skips_tsv}")
  fi

  # conftest.py declares --hw-classification with nargs="+", so it must come
  # after the test file path and before the option-only --deselect tail.
  local -a pytest_args=(-m pytest "${file#test/}")
  if [ -n "${hw_classification}" ]; then
    pytest_args+=(--hw-classification "${hw_classification}")
  fi
  if [ "${#skipped_nodeids[@]}" -gt 0 ]; then
    echo "  Skipping ${#skipped_nodeids[@]} known-failing case(s) via pytest --deselect"
    local nid
    for nid in "${skipped_nodeids[@]}"; do
      pytest_args+=(--deselect "${nid}")
    done
  fi

  local -a run_cmd=("${python_bin}" -u -m coverage run --source="${source_pkg}" --branch)
  run_cmd+=("${pytest_args[@]}")

  # cd into the test dir first (same cwd convention as upstream run_test.py);
  # -u keeps logs streaming
  local script='cd "$1"; shift; exec "$@"'
  local status=0
  set +e
  setup_coverage "${flat}"
  setup_env "${file}"
  run_logged_command "${log_file}" bash -c "${script}" _ "${test_dir}" \
    timeout --kill-after=30 "${timeout_seconds}" "${run_cmd[@]}"
  status=$?
  set -e

  # pytest exit 5 = nothing selected: expected when hw_classification drops every
  # case of a file, and the PR trigger pipeline collects nothing there either.
  local no_cases=0
  if [ "${status}" -eq 5 ]; then
    echo "  No case selected (hw_classification='${hw_classification}' / all nodeids deselected), not a failure"
    no_cases=1
    status=0
  fi

  # FAILED marker lets downstream drop incomplete coverage of failed tests
  if [ "${status}" -ne 0 ]; then
    echo "1" > "$(dirname "${COVERAGE_FILE}")/FAILED"
  fi

  if [ "${no_cases}" -eq 1 ]; then
    test_results+=( "${category}|${file}|NO_CASES|${log_file}" )
  elif [ "${status}" -eq 0 ]; then
    test_results+=( "${category}|${file}|PASSED|${log_file}" )
  else
    test_results+=( "${category}|${file}|FAILED|${log_file}" )
    failed_logs+=( "${category}|${file}|${log_file}" )
    if [ "${overall_status}" -eq 0 ]; then
      overall_status="${status}"
    fi
  fi
}

# Merge per-test coverage into .coverage + coverage.xml. FAILED-marked data
# still participates; downstream can filter by the FAILED marker if needed.
combine_coverage() {
  local data_files=()
  mapfile -t data_files < <(find "${covdata_root}" -mindepth 3 -maxdepth 3 -name coverage -type f | sort)
  if [ "${#data_files[@]}" -eq 0 ]; then
    echo "=== Combine: no coverage data files found, skip ==="
    return
  fi
  echo "=== Combining ${#data_files[@]} coverage data file(s) ==="
  COVERAGE_FILE="${covdata_root}/.coverage" "${python_bin}" -m coverage combine "${data_files[@]}"
  COVERAGE_FILE="${covdata_root}/.coverage" "${python_bin}" -m coverage xml \
    --include="*/${source_pkg}/*" -o "${covdata_root}/coverage.xml"
  echo "=== Combined coverage: ${covdata_root}/.coverage, report: ${covdata_root}/coverage.xml ==="
  COVERAGE_FILE="${covdata_root}/.coverage" "${python_bin}" -m coverage report --include="*/${source_pkg}/*" | tail -n 5 || true
}

print_summary() {
  local result category file status log_file
  echo "=== TEST SUMMARY ==="
  for result in "${test_results[@]}"; do
    IFS='|' read -r category file status log_file <<< "${result}"
    echo "  [${category}] ${status}: ${file}  (log: ${log_file})"
  done
}

print_failed_summary() {
  if [ "${#failed_logs[@]}" -eq 0 ]; then
    echo "=== Failed cases: none ==="
    return
  fi

  echo "=== FAILED CASES (${#failed_logs[@]}) ==="
  local failed category file log_file
  local json="[" i=0
  for failed in "${failed_logs[@]}"; do
    IFS='|' read -r category file log_file <<< "${failed}"
    echo "  - [${category}] ${file}  (log: ${log_file})"
    [ "${i}" -gt 0 ] && json+=","
    json+="{\"category\":\"${category}\",\"name\":\"${file}\",\"log\":\"${log_file}\"}"
    i=$((i + 1))
  done
  json+="]"
  echo "${json}" > "${log_dir}/failed_cases.json"
  echo "  failed_cases.json: ${log_dir}/failed_cases.json"

  echo ""
  for failed in "${failed_logs[@]}"; do
    IFS='|' read -r category file log_file <<< "${failed}"
    echo "----- tail of [${category}] ${file} -----"
    tail -n 30 "${log_file}" 2>/dev/null || true
    echo ""
  done
}

# Extract failing cases from a shard's junit XMLs into a small JSON file.
write_failed_cases() {
  local junit_dir="$1" out_file="$2"
  python3 - "$junit_dir" "$out_file" <<'PYEOF'
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

junit_dir = Path(sys.argv[1])
out_file = Path(sys.argv[2])
failed = []
total = 0
if junit_dir.is_dir():
    for xml_file in sorted(junit_dir.glob("*.xml")):
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError:
            continue
        for case in root.iter("testcase"):
            total += 1
            status = next(
                (tag for tag in ("failure", "error") if case.find(tag) is not None),
                "",
            )
            if not status:
                continue
            classname = case.get("classname", "")
            name = case.get("name", "")
            failed.append({
                "nodeid": f"{classname}::{name}" if classname else name,
                "status": status,
            })
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

# Shard mode: one collected case shard through run_npu_test_shard.py — same case
# isolation, worker count and NPU card binding as the PR trigger pipeline — with
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
parallel = True
data_file = ${cov_dir}/.coverage
EOF

  local site_dir
  site_dir="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
  printf 'import coverage; coverage.process_startup()\n' > "${site_dir}/zz_coverage_startup.pth"
  export COVERAGE_PROCESS_START="${rc_file}"

  echo "=== Cases: ${cases_json}  workers: ${max_workers}  timeout: ${timeout_seconds}s  device-env: ${device_env} ==="
  echo "=== Coverage rc: ${rc_file}  startup hook: ${site_dir}/zz_coverage_startup.pth ==="

  local status=0
  set +e
  python3 -u "${repo_root}/.github/scripts/run_npu_test_shard.py" \
    --cases-json "${cases_json}" \
    --test-dir "${test_dir}" \
    --report-dir "${report_dir}" \
    --max-workers "${max_workers}" \
    --timeout "${timeout_seconds}" \
    --device-env "${device_env}" \
    --verbose 2>&1 | tee "${shard_dir}/run.log"
  status=${PIPESTATUS[0]}
  set -e

  write_failed_cases "${report_dir}/junit_xmls" "${shard_dir}/failed_cases.json"

  echo "=== Shard done: exit=${status}, coverage data file(s)=$(find "${cov_dir}" -name '.coverage.*' -type f | wc -l) ==="
  overall_status="${status}"
}

# Merge mode: combine every shard's coverage data + failure list into the fixed
# @latest dir (staging first, promote last, so a crashed merge keeps the
# previous @latest).
merge_coverage() {
  local staging="${out_root}/${source_pkg}@staging"
  local merged="${out_root}/${source_pkg}@latest"
  local -a dirs=() data_files=()
  IFS=',' read -r -a dirs <<< "$1"

  rm -rf "${staging}"
  mkdir -p "${staging}/covdata" "${staging}/logs"

  local src data_file shard_name
  for src in "${dirs[@]}"; do
    if [ ! -d "${src}" ]; then
      echo "WARNING: merge input not found: ${src}" >&2
      continue
    fi
    shard_name="$(basename "${src}")"
    echo "=== Merge input: ${src} ==="
    mkdir -p "${staging}/covdata/${shard_name}"
    while IFS= read -r data_file; do
      cp "${data_file}" "${staging}/covdata/${shard_name}/"
      data_files+=("${staging}/covdata/${shard_name}/$(basename "${data_file}")")
    done < <(find "${src}" \( -name '.coverage.*' -o -name 'coverage' \) -type f | sort)
    if [ -f "${src}/run.log" ]; then
      cp "${src}/run.log" "${staging}/logs/${shard_name}.log"
    fi
  done

  if [ "${#data_files[@]}" -eq 0 ]; then
    echo "WARNING: no coverage data files found in: ${dirs[*]}" >&2
  else
    echo "=== Combining ${#data_files[@]} coverage data file(s) ==="
    COVERAGE_FILE="${staging}/.coverage" "${python_bin}" -m coverage combine "${data_files[@]}"
    COVERAGE_FILE="${staging}/.coverage" "${python_bin}" -m coverage xml \
      --include="*/${source_pkg}/*" -o "${staging}/coverage.xml"
    echo "=== Combined coverage: ${staging}/.coverage, report: ${staging}/coverage.xml ==="
    COVERAGE_FILE="${staging}/.coverage" "${python_bin}" -m coverage report \
      --include="*/${source_pkg}/*" | tail -n 5 || true
  fi

  python3 - "${staging}/logs/failed_cases.json" "${dirs[@]}" <<'PYEOF'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
shards = [Path(p) for p in sys.argv[2:]]
all_failed = []
total = 0
sources = []
for shard in shards:
    shard_file = shard / "failed_cases.json"
    if not shard_file.is_file():
        continue
    data = json.loads(shard_file.read_text(encoding="utf-8"))
    total += data.get("total_cases", 0)
    all_failed.extend(data.get("failed_cases", []))
    sources.append(shard.name)
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
PYEOF

  rm -rf "${merged}"
  mv "${staging}" "${merged}"
  echo "=== Promoted: ${merged} ==="
  overall_status=0
}

if [ -n "${merge_dirs}" ]; then
  merge_coverage "${merge_dirs}"
  exit "${overall_status}"
fi

if [ -n "${cases_json}" ]; then
  run_case_shard
  exit "${overall_status}"
fi

dump_source_snapshot

# Serial execution: distributed tests require it, and device/compile
# subprocesses make parallel runs unsafe.
for target in "${targets[@]}"; do
  IFS='|' read -r category file <<< "${target}"
  run_test_target "${category}" "${file}"
done

print_summary
print_failed_summary
combine_coverage

# Promote staging -> @latest. Reaching this line means the full flow
# (all tests + combine) completed end-to-end; test failures do NOT block
# promotion (per-test FAILED markers inside covdata let downstream filter
# them out). A crashed/killed run never gets here, so the previous
# @latest data survives untouched. --no-promote leaves torch@staging for the
# caller (used when the job hands its data to a later merge job).
if [ "${promote}" -eq 1 ]; then
  rm -rf "${covdata_root}"
  mv "${staging_dir}" "${covdata_root}"
else
  echo "=== --no-promote: left ${staging_dir} in place ==="
fi

exit "${overall_status}"
