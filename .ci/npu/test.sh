#!/bin/bash
# ==============================================================================
# NPU Test Execution Script
#
# This script runs PyTorch upstream tests on NPU machines, aligned with
# PyTorch's native test.sh + run_test.py execution flow.
#
# Key differences from PyTorch native testing:
# 1. Environment: NPU machines + CANN + torch_npu
# 2. Test selection: case_paths_ci.yml whitelist/blacklist
# 3. Disabled tests: disabled_testcases.json (via pytest plugin)
# 4. Crashed files: CRASHED.yml (excluded to prevent CI crashes)
#
# Usage:
#   SHARD_NUMBER=1 NUM_TEST_SHARDS=10 ./test.sh
#
# Environment variables:
#   SHARD_NUMBER       - Shard number (default: 1)
#   NUM_TEST_SHARDS    - Total number of shards (default: 1)
#   PYTORCH_TEST_SRC   - Path to PyTorch source with tests
#   ASCEND_PYTORCH_DIR - Path to ascend_pytorch repository
#   NUM_PARALLEL_PROCS - Number of parallel workers (default: 2)
#   PYTHON_VERSION     - Python version to use (default: 3.11)
# ==============================================================================

set -ex -o pipefail

# ==============================================================================
# Section 1: Environment Initialization
# ==============================================================================

# NPU environment initialization (CANN stack)
echo "[NPU Test] Initializing CANN environment..."
source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

# ==============================================================================
# Section 2: Core Environment Variables
# ==============================================================================

# Enable PyTorch device backend autoload for NPU
export TORCH_DEVICE_BACKEND_AUTOLOAD=1

# Mark that we're running tests on NPU
export PYTORCH_TEST_NPU=1

# Disable target determination (TD) since we don't have TD artifacts
# TD requires pre-generated td_results.json from PyTorch CI
export NO_TD=1

# Mark as CI environment for proper behavior
export CI=true

# Continue through errors to run all tests in shard
export CONTINUE_THROUGH_ERROR=1

# Disable git dirty check (we apply patches which modifies files)
export PYTORCH_TEST_SKIP_ASSERT_GIT_NOT_DIRTY=1

# ==============================================================================
# Section 3: Path Configuration
# ==============================================================================

# Required paths (must be set by caller)
PYTORCH_TEST_SRC="${PYTORCH_TEST_SRC:-$(pwd)}"
ASCEND_PYTORCH_DIR="${ASCEND_PYTORCH_DIR:-$(pwd)/../ascend_pytorch}"

# Validate paths
if [ ! -d "${PYTORCH_TEST_SRC}/test" ]; then
    echo "[NPU Test] Error: PYTORCH_TEST_SRC/test not found: ${PYTORCH_TEST_SRC}/test"
    exit 1
fi

if [ ! -d "${ASCEND_PYTORCH_DIR}" ]; then
    echo "[NPU Test] Error: ASCEND_PYTORCH_DIR not found: ${ASCEND_PYTORCH_DIR}"
    exit 1
fi

# ==============================================================================
# Section 4: Shard Configuration
# ==============================================================================

SHARD_NUMBER="${SHARD_NUMBER:=1}"
NUM_TEST_SHARDS="${NUM_TEST_SHARDS:=1}"
NUM_PARALLEL_PROCS="${NUM_PARALLEL_PROCS:=2}"

echo "[NPU Test] Configuration:"
echo "  PYTORCH_TEST_SRC: ${PYTORCH_TEST_SRC}"
echo "  ASCEND_PYTORCH_DIR: ${ASCEND_PYTORCH_DIR}"
echo "  SHARD_NUMBER: ${SHARD_NUMBER}"
echo "  NUM_TEST_SHARDS: ${NUM_TEST_SHARDS}"
echo "  NUM_PARALLEL_PROCS: ${NUM_PARALLEL_PROCS}"

# ==============================================================================
# Section 5: Disabled Testcases Plugin Setup
# ==============================================================================

DISABLED_TESTCASES_JSON="${PYTORCH_TEST_SRC}/test_upsteam/disabled_testcases.json"

if [ -f "${DISABLED_TESTCASES_JSON}" ]; then
    export NPU_DISABLED_TESTCASES_JSON="${DISABLED_TESTCASES_JSON}"
    echo "[NPU Test] Disabled testcases: ${DISABLED_TESTCASES_JSON}"
else
    echo "[NPU Test] Warning: disabled_testcases.json not found"
fi

# Add pytest plugin to PYTHONPATH
export PYTHONPATH="${ASCEND_PYTORCH_DIR}/.github/scripts:${PYTHONPATH}"

# Enable pytest plugin via PYTEST_ADDOPTS
export PYTEST_ADDOPTS="-p pytest_disabled_testcases_plugin"

# ==============================================================================
# Section 6: Python Configuration
# ==============================================================================

PYTHON_VERSION="${PYTHON_VERSION:=3.11}"
PYTHON="python${PYTHON_VERSION}"

echo "[NPU Test] Using Python: ${PYTHON}"
${PYTHON} --version

# ==============================================================================
# Section 7: Resolve Whitelist/Blacklist/CRASHED
# ==============================================================================

echo "[NPU Test] Resolving whitelist/blacklist..."

CASE_PATHS_CONFIG="${PYTORCH_TEST_SRC}/test_upsteam/case_paths_ci.yml"
CRASHED_CONFIG="${PYTORCH_TEST_SRC}/test_upsteam/CRASHED.yml"
TEST_DIR="${PYTORCH_TEST_SRC}/test"

INCLUDE_OUTPUT="/tmp/include_tests.txt"
EXCLUDE_OUTPUT="/tmp/exclude_tests.txt"
EXTRA_OUTPUT="/tmp/extra_pytest_tests.txt"

${PYTHON} "${ASCEND_PYTORCH_DIR}/.github/scripts/resolve_whitelist.py" \
    --case-paths-config "${CASE_PATHS_CONFIG}" \
    --crashed-files-config "${CRASHED_CONFIG}" \
    --test-dir "${TEST_DIR}" \
    --pytorch-root "${PYTORCH_TEST_SRC}" \
    --output-include "${INCLUDE_OUTPUT}" \
    --output-exclude "${EXCLUDE_OUTPUT}" \
    --output-extra "${EXTRA_OUTPUT}" \
    --verbose

# Build include/exclude arguments for run_test.py
INCLUDE_ARGS=$(cat "${INCLUDE_OUTPUT}" | tr '\n' ' ' | sed 's/ *$//')
EXCLUDE_ARGS=$(cat "${EXCLUDE_OUTPUT}" | tr '\n' ' ' | sed 's/ *//')

echo "[NPU Test] Include tests: $(wc -l < ${INCLUDE_OUTPUT}) files"
echo "[NPU Test] Exclude tests: $(wc -l < ${EXCLUDE_OUTPUT}) files"
echo "[NPU Test] Extra pytest: $(wc -l < ${EXTRA_OUTPUT}) files"

# ==============================================================================
# Section 8: Optional - Download Test Time Statistics
# ==============================================================================

# Download test-times.json to optimize shard distribution
# This is optional - if download fails, sharding falls back to round-robin
echo "[NPU Test] Downloading test time statistics (optional)..."

mkdir -p "${PYTORCH_TEST_SRC}/.additional_ci_files"
curl -sL --connect-timeout 10 --max-time 30 \
    "https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/test-times.json" \
    -o "${PYTORCH_TEST_SRC}/.additional_ci_files/test-times.json" 2>/dev/null || true

if [ -f "${PYTORCH_TEST_SRC}/.additional_ci_files/test-times.json" ]; then
    echo "[NPU Test] Test times downloaded successfully"
else
    echo "[NPU Test] Test times not available, using round-robin sharding"
fi

# ==============================================================================
# Section 9: Change to PyTorch Source Directory
# ==============================================================================

cd "${PYTORCH_TEST_SRC}"

# ==============================================================================
# Section 10: Phase 1 - Run Tests via run_test.py
# ==============================================================================

REPORT_DIR="${PYTORCH_TEST_SRC}/test/test-reports"
mkdir -p "${REPORT_DIR}"

echo "[NPU Test] Phase 1: Running tests via run_test.py..."
echo "[NPU Test] Include args: ${INCLUDE_ARGS}"
echo "[NPU Test] Exclude args: ${EXCLUDE_ARGS}"

# Set NUM_PROCS for parallel execution
export NUM_PARALLEL_PROCS="${NUM_PARALLEL_PROCS}"

# Run regular tests (excluding distributed tests)
# Distributed tests will be handled separately in Phase 2
set +e
${PYTHON} test/run_test.py \
    --include ${INCLUDE_ARGS} \
    --exclude ${EXCLUDE_ARGS} \
    --exclude-distributed-tests \
    --exclude-jit-executor \
    --shard "${SHARD_NUMBER}" "${NUM_TEST_SHARDS}" \
    --continue-through-error \
    --verbose \
    2>&1 | tee /tmp/run_test_phase1.log
PHASE1_STATUS=${PIPESTATUS[0]}
set -e

echo "[NPU Test] Phase 1 completed with status: ${PHASE1_STATUS}"

# ==============================================================================
# Section 11: Phase 2 - Distributed Tests
# ==============================================================================

# Run distributed tests only on shard 1 to avoid duplicate execution
# This aligns with PyTorch's test_distributed() behavior in test.sh
if [ "${SHARD_NUMBER}" -eq 1 ] && [ -n "${INCLUDE_ARGS}" ]; then
    echo "[NPU Test] Phase 2: Running distributed tests..."

    set +e
    ${PYTHON} test/run_test.py \
        --distributed-tests \
        --include ${INCLUDE_ARGS} \
        --exclude ${EXCLUDE_ARGS} \
        --shard "${SHARD_NUMBER}" "${NUM_TEST_SHARDS}" \
        --continue-through-error \
        --verbose \
        2>&1 | tee /tmp/run_test_phase2.log
    PHASE2_STATUS=${PIPESTATUS[0]}
    set -e

    echo "[NPU Test] Phase 2 completed with status: ${PHASE2_STATUS}"
else
    echo "[NPU Test] Phase 2 skipped (shard ${SHARD_NUMBER} != 1)"
    PHASE2_STATUS=0
fi

# ==============================================================================
# Section 12: Phase 3 - Extra pytest Tests (not in TESTS list)
# ==============================================================================

if [ -s "${EXTRA_OUTPUT}" ]; then
    echo "[NPU Test] Phase 3: Running extra pytest tests..."

    EXTRA_FILES=$(cat "${EXTRA_OUTPUT}" | tr '\n' ' ' | sed 's/ *$//')
    echo "[NPU Test] Extra files: ${EXTRA_FILES}"

    # Create separate report directory for extra tests
    EXTRA_REPORT_DIR="${REPORT_DIR}/extra"
    mkdir -p "${EXTRA_REPORT_DIR}"

    set +e
    ${PYTHON} -m pytest \
        -p pytest_disabled_testcases_plugin \
        ${EXTRA_FILES} \
        --timeout 600 \
        -v --tb=short \
        --junitxml="${EXTRA_REPORT_DIR}/extra_pytest.xml" \
        2>&1 | tee /tmp/run_test_phase3.log
    PHASE3_STATUS=${PIPESTATUS[0]}
    set -e

    echo "[NPU Test] Phase 3 completed with status: ${PHASE3_STATUS}"
else
    echo "[NPU Test] Phase 3 skipped (no extra tests)"
    PHASE3_STATUS=0
fi

# ==============================================================================
# Section 13: Final Status
# ==============================================================================

FINAL_STATUS=0
if [ "${PHASE1_STATUS}" -ne 0 ]; then
    FINAL_STATUS=${PHASE1_STATUS}
    echo "[NPU Test] Phase 1 failed with status ${PHASE1_STATUS}"
fi
if [ "${PHASE2_STATUS}" -ne 0 ]; then
    FINAL_STATUS=${PHASE2_STATUS}
    echo "[NPU Test] Phase 2 failed with status ${PHASE2_STATUS}"
fi
if [ "${PHASE3_STATUS}" -ne 0 ]; then
    FINAL_STATUS=${PHASE3_STATUS}
    echo "[NPU Test] Phase 3 failed with status ${PHASE3_STATUS}"
fi

echo "[NPU Test] Final status: ${FINAL_STATUS}"
echo "[NPU Test] Test reports directory: ${REPORT_DIR}"

# List report files
if [ -d "${REPORT_DIR}" ]; then
    echo "[NPU Test] Report files:"
    find "${REPORT_DIR}" -name "*.xml" -type f | head -20
fi

exit ${FINAL_STATUS}