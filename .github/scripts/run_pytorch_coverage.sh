#!/usr/bin/env bash
set -euo pipefail

# run_pytorch_coverage.sh - PyTorch source coverage collection.
# Called only by .github/workflows/pytorch_coverage_nightly.yml. Runs the
# pr_test_whitelist.yml test files on the pytorch source tree and collects
# coverage of the installed torch python sources.
#
# Usage:
#   ./run_pytorch_coverage.sh --test-dir <pytorch>/test \
#                            [--category c1,c2] [--timeout <seconds>]
# Env:
#   OUT_ROOT    output root for all artifacts (default <repo root>/outputs)
#   PYTHON_BIN  python interpreter (default python)
#
# Outputs land under OUT_ROOT:
#   convstub/torch/            source snapshot (matches the run env)
#   torch@YYYYMMDD/            per-test covdata dirs (+ FAILED markers), logs,
#                              failed_cases.json, combined .coverage/coverage.xml

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

config_file="${repo_root}/.github/config/pr_test_whitelist.yml"
category_filter=""
test_dir="${GITHUB_WORKSPACE:-${repo_root}}/pytorch/test"
source_pkg="torch"
device_env="privateuse1"
timeout_seconds=600
python_bin="${PYTHON_BIN:-python}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --category) category_filter="$2"; shift 2 ;;
    --test-dir) test_dir="$2";        shift 2 ;;
    --timeout)  timeout_seconds="$2"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *)  echo "Unexpected positional argument: $1 (targets come from ${config_file})" >&2; exit 2 ;;
  esac
done

if [ ! -f "${config_file}" ]; then
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
timestamp="$(date +%Y%m%d)"
covdata_root="${out_root}/${source_pkg}@${timestamp}"
log_dir="${covdata_root}/logs"
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

if [ "${#targets[@]}" -eq 0 ]; then
  echo "ERROR: no test targets resolved from ${config_file} (category filter: '${category_filter}')" >&2
  exit 1
fi

echo "=== Targets: ${#targets[@]} file(s) from ${config_file} ==="
echo "=== test-dir: ${test_dir}  source: ${source_pkg}  device-env: ${device_env} ==="

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
  local td; td="$(dirname "$1")"
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

  # cd into the test dir first (same cwd convention as upstream run_test.py);
  # -u keeps logs streaming
  local script='cd "$1"; shift; exec "$@"'
  local status=0
  set +e
  setup_coverage "${flat}"
  setup_env "${file}"
  run_logged_command "${log_file}" bash -c "${script}" _ "${test_dir}" \
    timeout --kill-after=30 "${timeout_seconds}" \
    "${python_bin}" -m coverage run --source="${source_pkg}" --branch -u "${file}"
  status=$?
  set -e

  # FAILED marker lets downstream drop incomplete coverage of failed tests
  if [ "${status}" -ne 0 ]; then
    echo "1" > "$(dirname "${COVERAGE_FILE}")/FAILED"
  fi

  if [ "${status}" -eq 0 ]; then
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

exit "${overall_status}"
