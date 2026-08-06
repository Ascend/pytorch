#!/bin/bash
# test-npu.sh — NPU test discovery, classification, sharding, and execution
#
# Replaces the former _test-collect-new.yml + inline scripts in _test-exec-new.yml
# with a single entry point that follows the upstream pytorch/.ci/pytorch/test.sh
# pattern: dispatch by TEST_CONFIG environment variable.
#
# Required environment variables:
#   TEST_CONFIG    Test category: core | tensor | distributed | graph | others
#
# Optional (with defaults):
#   SHARD_NUMBER        Shard index (1-based, default: 1)
#   NUM_TEST_SHARDS     Total shards  (default: 1)
#   TEST_DIR            Path to pytorch/test/  (default: pytorch/test)
#   REPORTS_DIR         Output directory for test reports (default: test-reports)
#   CLASSIFICATION_CONFIG  Path to nightly_v2_test_whitelist.yml
#   PYTHONPATH_BASE     Path to v2 scripts directory (default: REPO_ROOT/.github/scripts/v2)
#
# Usage:
#   TEST_CONFIG=core SHARD_NUMBER=1 bash test-npu.sh

set -ex -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ==============================================================================
# Defaults (matching upstream test.sh pattern: ${VAR:=default})
# ==============================================================================

SHARD_NUMBER="${SHARD_NUMBER:-1}"
NUM_TEST_SHARDS="${NUM_TEST_SHARDS:-1}"
TEST_DIR="${TEST_DIR:-pytorch/test}"
REPORTS_DIR="${REPORTS_DIR:-test-reports}"
CLASSIFICATION_CONFIG="${CLASSIFICATION_CONFIG:-${REPO_ROOT}/.github/config/nightly_v2_test_whitelist.yml}"
PYTHONPATH_BASE="${PYTHONPATH_BASE:-${REPO_ROOT}/.github/scripts/v2}"

# ==============================================================================
# NPU Environment Setup
# ==============================================================================

setup_npu_env() {
    # Source CANN / NNAL runtime libraries
    source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true
    source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

    # Route device-type tests to privateuse1 (NPU)
    export PYTORCH_TESTING_DEVICE_ONLY_FOR="privateuse1"
    export PYTORCH_TESTING_DEVICE_FOR_CUSTOM="privateuse1"

    # Load NPU poisoning detection plugin inside pytest subprocesses
    export PYTEST_ADDOPTS="-p no:xdist -p npu_poisoning_plugin"

    # Ensure v2 scripts are importable by pytest and inline Python
    export PYTHONPATH="${PYTHONPATH_BASE}:${PYTHONPATH:-}"
}

# ==============================================================================
# Core Execution: discover → classify → shard → execute → parse
# ==============================================================================

test_npu_execute() {
    local category="$1"
    setup_npu_env

    # Resolve pytorch source root once (used by both Step 1 and run_test.py).
    # TEST_DIR is e.g. ${GITHUB_WORKSPACE}/pytorch/test, so parent = pytorch/.
    local pytorch_root
    pytorch_root="$(cd "${TEST_DIR}/.." && pwd)" || {
        echo "ERROR: TEST_DIR not found: ${TEST_DIR}" >&2
        return 1
    }

    echo "============================================"
    echo "NPU Test Execution: ${category}"
    echo "Shard: ${SHARD_NUMBER}/${NUM_TEST_SHARDS}"
    echo "============================================"

    # ---- Verify NPU device availability ----
    echo "=== NPU Device Information ==="
    npu-smi info
    echo "=== End of NPU Device Information ==="

    python -c "
import torch; print(f'torch: {torch.__version__}')
import torch_npu; print(f'torch_npu: {torch_npu.__version__}')
print(f'NPU available: {torch.npu.is_available()}')
print(f'NPU count: {torch.npu.device_count()}')
"

    # ---- Discover, classify, and shard test files ----
    # All done in a single Python invocation using shard_test_files.py functions.
    # No intermediate JSON files — the file list goes directly to --include.

    # Step 1: Get the complete set of valid --include choices from run_test.py.
    # run_test.py uses argparse choices=TESTS, where TESTS is built by
    # discover_tests() which excludes some directories (autograd/, fx/, jit/
    # are executed by their parent test file).  Any file not in TESTS would
    # be rejected with "invalid choice".
    echo "=== Fetching valid --include choices from run_test.py ==="
    VALID_TESTS_FILE="/tmp/valid_tests_${SHARD_NUMBER}.txt"
    python3 -c "
import sys
sys.path.insert(0, '${pytorch_root}')
from tools.testing.discover_tests import TESTS
for t in TESTS:
    print(t)
" > "${VALID_TESTS_FILE}"
    local valid_count=$(wc -l < "${VALID_TESTS_FILE}")
    echo "run_test.py accepts ${valid_count} test modules via --include"

    # Step 2: Scan, classify, shard files → intersect with valid TESTS.
    echo "=== Discovering test files for ${category} shard ${SHARD_NUMBER}/${NUM_TEST_SHARDS} ==="

    # Expected files list (also used as source of truth for --include).
    # run_test.py --include expects module-style paths: 'nn/test_convolution'
    # NOT file paths: 'test/nn/test_convolution.py'.
    EXPECTED_FILES="/tmp/test_npu_expected_${category}_${SHARD_NUMBER}.txt"

    python3 -c "
import sys
sys.path.insert(0, '${PYTHONPATH_BASE}')

from shard_test_files import (
    scan_all_test_files,
    classify_files,
    split_round_robin,
    load_categories_config,
)
from pathlib import Path

config = load_categories_config('${CLASSIFICATION_CONFIG}')
all_files = scan_all_test_files(Path('${TEST_DIR}'))
classified = classify_files(
    all_files,
    config.get('categories', {}),
    config.get('exclude', []),
)
cat_files = classified.get('${category}', [])
num_shards = int(${NUM_TEST_SHARDS})
shard_idx = int(${SHARD_NUMBER}) - 1

shards = split_round_robin(cat_files, num_shards)
my_files = shards[shard_idx] if shard_idx < len(shards) else []

# Load the valid TESTS set fetched in step 1
valid_tests = set()
with open('${VALID_TESTS_FILE}') as vf:
    for line in vf:
        line = line.strip()
        if line:
            valid_tests.add(line)

# Convert 'test/nn/test_convolution.py' -> 'nn/test_convolution'
# Intersect with run_test.py's valid choices.
included = 0
skipped = 0
with open('${EXPECTED_FILES}', 'w') as fh:
    for f in sorted(my_files):
        converted = f.removeprefix('test/').removesuffix('.py')
        if converted not in valid_tests:
            skipped += 1
            print(f'SKIPPED (not in TESTS): {converted}', file=sys.stderr)
            continue
        fh.write(converted + '\n')
        print(converted)
        included += 1
print(f'Included: {included}, skipped: {skipped} (not in run_test.py TESTS)', file=sys.stderr)
" 2>&1 | tee /tmp/test_npu_discover_${category}_${SHARD_NUMBER}.log

    files=$(tr '\n' ' ' < "${EXPECTED_FILES}")
    local file_count=$(wc -w < "${EXPECTED_FILES}")

    echo "Files in this shard: ${file_count}"
    if [ "${file_count}" -eq 0 ]; then
        echo "WARNING: No test files found for category '${category}'"
        return 0
    fi
    echo "Runner: linux-aarch64-a3-8 (8-card NPU)"
    echo "Execution: upstream run_test.py"
    echo "Device routing: PYTORCH_TESTING_DEVICE_ONLY_FOR=privateuse1"
    echo "HW classification: ACCELERATOR"
    local npu_count="${NPU_COUNT:-8}"
    local devices_per_proc="${DEVICES_PER_PROC:-1}"
    local num_procs=$(( npu_count / devices_per_proc ))
    echo "NPU cards: ${npu_count}, devices/proc: ${devices_per_proc}, concurrency: ${num_procs}"

    mkdir -p "${REPORTS_DIR}"

    # Reset NPU device counter (used by npu_poisoning_plugin pytest_configure)
    echo 0 > /tmp/npu_device_counter.lock

    # ---- Execute via upstream run_test.py ----
    # NUM_PROCS = npu_count / devices_per_proc.
    # Each pytest process acquires devices_per_proc cards via npu_poisoning_plugin.
    echo "=== Running tests ==="
    set +e
    NUM_PROCS="${num_procs}" \
    NPU_DEVICES_PER_PROC="${devices_per_proc}" \
    NPU_COUNT="${npu_count}" \
    python pytorch/test/run_test.py \
        --include ${files} \
        --hw-classification ACCELERATOR \
        --continue-through-error \
        --verbose \
        2>&1 | tee "/tmp/test_npu_${category}_${SHARD_NUMBER}.log"
    local test_status=${PIPESTATUS[0]}
    set -e

    echo "=== run_test.py exit status: ${test_status} ==="

    # ---- Parse stderr + JUnit XMLs → JSONL (for report workflow) ----
    echo "=== Generating shard JSONL ==="
    python "${PYTHONPATH_BASE}/generate_shard_jsonl.py" \
        --category "${category}" \
        --shard "${SHARD_NUMBER}" \
        --expected-files "${EXPECTED_FILES}" \
        --execution-log "/tmp/test_npu_${category}_${SHARD_NUMBER}.log" \
        --reports-dir "${REPORTS_DIR}" \
        --runner "${RUNNER:-linux-aarch64-a3-8}"

    return ${test_status}
}

# ==============================================================================
# Dispatch (mirrors upstream test.sh: case $TEST_CONFIG in ...)
# ==============================================================================

if [ -z "${TEST_CONFIG}" ]; then
    echo "Usage: TEST_CONFIG=<category> $0"
    echo "Categories are defined in nightly_v2_test_whitelist.yml"
    exit 1
fi
test_npu_execute "${TEST_CONFIG}"
