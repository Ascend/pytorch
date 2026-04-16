#!/bin/bash
# ==============================================================================
# NPU Test Execution Script
#
# Aligned with PyTorch .ci/pytorch/test.sh + test/run_test.py flow.
# Differences from PyTorch CI:
#   1. Environment: NPU machine + CANN + torch_npu (vs CUDA/ROCm)
#   2. Test scope: case_paths_ci.yml whitelist (vs full TESTS list)
#   3. Disabled tests: disabled_testcases.json via pytest plugin
#   4. Crashed files: CRASHED.yml excluded to prevent segfaults
#
# Environment variables (set by caller / workflow):
#   SHARD_NUMBER        - current shard (1-indexed, default 1)
#   NUM_TEST_SHARDS     - total shards (default 1)
#   NUM_PARALLEL_PROCS  - pytest-xdist workers (default 2)
#   PYTORCH_TEST_SRC    - path to PyTorch source with patched tests
#   ASCEND_PYTORCH_DIR  - path to ascend_pytorch repository
#   PYTHON_VERSION      - python version (default 3.11)
#   RUN_MODE            - "regular" (default), "distributed", or "all"
# ==============================================================================

set -ex -o pipefail

# ==============================================================================
# 1. NPU Environment
# ==============================================================================

source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

# Core NPU env vars (analogous to CUDA_VISIBLE_DEVICES etc in test.sh)
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export PYTORCH_TEST_NPU=1
export NO_TD=1               # no target-determination artifacts
export CI=true
export CONTINUE_THROUGH_ERROR=1

# ==============================================================================
# 2. Path validation
# ==============================================================================

PYTORCH_TEST_SRC="${PYTORCH_TEST_SRC:?'PYTORCH_TEST_SRC must be set'}"
ASCEND_PYTORCH_DIR="${ASCEND_PYTORCH_DIR:?'ASCEND_PYTORCH_DIR must be set'}"

if [ ! -d "${PYTORCH_TEST_SRC}/test" ]; then
    echo "Error: ${PYTORCH_TEST_SRC}/test not found"; exit 1
fi
if [ ! -d "${ASCEND_PYTORCH_DIR}" ]; then
    echo "Error: ${ASCEND_PYTORCH_DIR} not found"; exit 1
fi

# ==============================================================================
# 3. Parameters
# ==============================================================================

SHARD_NUMBER="${SHARD_NUMBER:-1}"
NUM_TEST_SHARDS="${NUM_TEST_SHARDS:-1}"
NUM_PARALLEL_PROCS="${NUM_PARALLEL_PROCS:-2}"
RUN_MODE="${RUN_MODE:-regular}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTHON="python${PYTHON_VERSION}"

echo "=== NPU Test Config ==="
echo "  SHARD        : ${SHARD_NUMBER}/${NUM_TEST_SHARDS}"
echo "  MODE         : ${RUN_MODE}"
echo "  PARALLEL     : ${NUM_PARALLEL_PROCS}"
echo "  PYTORCH_SRC  : ${PYTORCH_TEST_SRC}"
echo "  ASCEND_DIR   : ${ASCEND_PYTORCH_DIR}"
echo "========================"

# ==============================================================================
# 4. Disabled-testcases pytest plugin
# ==============================================================================

DISABLED_JSON="${PYTORCH_TEST_SRC}/test_upsteam/disabled_testcases.json"
if [ -f "${DISABLED_JSON}" ]; then
    export NPU_DISABLED_TESTCASES_JSON="${DISABLED_JSON}"
fi
export PYTHONPATH="${ASCEND_PYTORCH_DIR}/.github/scripts:${PYTHONPATH}"
export PYTEST_ADDOPTS="-p pytest_disabled_testcases_plugin"

# ==============================================================================
# 5. Resolve whitelist -> include / exclude / extra
# ==============================================================================

INCLUDE_FILE="/tmp/include_tests.txt"
EXCLUDE_FILE="/tmp/exclude_tests.txt"
EXTRA_FILE="/tmp/extra_pytest_tests.txt"

${PYTHON} "${ASCEND_PYTORCH_DIR}/.github/scripts/resolve_whitelist.py" \
    --case-paths-config "${PYTORCH_TEST_SRC}/test_upsteam/case_paths_ci.yml" \
    --crashed-files-config "${PYTORCH_TEST_SRC}/test_upsteam/CRASHED.yml" \
    --test-dir "${PYTORCH_TEST_SRC}/test" \
    --pytorch-root "${PYTORCH_TEST_SRC}" \
    --output-include "${INCLUDE_FILE}" \
    --output-exclude "${EXCLUDE_FILE}" \
    --output-extra "${EXTRA_FILE}" \
    --verbose

INCLUDE_COUNT=$(wc -l < "${INCLUDE_FILE}" | tr -d ' ')
EXCLUDE_COUNT=$(wc -l < "${EXCLUDE_FILE}" | tr -d ' ')
EXTRA_COUNT=$(wc -l < "${EXTRA_FILE}" | tr -d ' ')

if [ "${INCLUDE_COUNT}" -eq 0 ]; then
    echo "Warning: include list is empty, nothing to run."
    exit 0
fi

INCLUDE_ARGS=$(tr '\n' ' ' < "${INCLUDE_FILE}" | sed 's/ *$//')
EXCLUDE_ARGS=$(tr '\n' ' ' < "${EXCLUDE_FILE}" | sed 's/ *$//')

# ==============================================================================
# 6. Optional: download test-times.json for smart sharding
# ==============================================================================

mkdir -p "${PYTORCH_TEST_SRC}/.additional_ci_files"
curl -sL --connect-timeout 10 --max-time 30 \
    "https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/test-times.json" \
    -o "${PYTORCH_TEST_SRC}/.additional_ci_files/test-times.json" 2>/dev/null || true

# ==============================================================================
# 7. cd into PyTorch source (run_test.py expects cwd = repo root)
# ==============================================================================

cd "${PYTORCH_TEST_SRC}"
mkdir -p test/test-reports

# ==============================================================================
# Helper: build common run_test.py flags
# ==============================================================================

build_common_flags() {
    echo "--shard ${SHARD_NUMBER} ${NUM_TEST_SHARDS} --continue-through-error --verbose"
}

# ==============================================================================
# 8. Phase 1: Regular tests via run_test.py
#    (mirrors test_python_shard() in .ci/pytorch/test.sh)
# ==============================================================================

PHASE1_STATUS=0
if [ "${RUN_MODE}" = "regular" ] || [ "${RUN_MODE}" = "all" ]; then
    echo ""
    echo "=== Phase 1: Regular tests (run_test.py, shard ${SHARD_NUMBER}/${NUM_TEST_SHARDS}) ==="
    echo ""

    set +e
    ${PYTHON} test/run_test.py \
        --include ${INCLUDE_ARGS} \
        --exclude ${EXCLUDE_ARGS} \
        --exclude-distributed-tests \
        --exclude-jit-executor \
        $(build_common_flags) \
        2>&1 | tee /tmp/run_test_phase1.log
    PHASE1_STATUS=${PIPESTATUS[0]}
    set -e

    echo "Phase 1 exit: ${PHASE1_STATUS}"
fi

# ==============================================================================
# 9. Phase 2: Distributed tests via run_test.py
#    (mirrors test_distributed() in .ci/pytorch/test.sh)
# ==============================================================================

PHASE2_STATUS=0
if [ "${RUN_MODE}" = "distributed" ] || [ "${RUN_MODE}" = "all" ]; then
    echo ""
    echo "=== Phase 2: Distributed tests ==="
    echo ""

    set +e
    ${PYTHON} test/run_test.py \
        --distributed-tests \
        --include ${INCLUDE_ARGS} \
        --exclude ${EXCLUDE_ARGS} \
        $(build_common_flags) \
        2>&1 | tee /tmp/run_test_phase2.log
    PHASE2_STATUS=${PIPESTATUS[0]}
    set -e

    echo "Phase 2 exit: ${PHASE2_STATUS}"
fi

# ==============================================================================
# 10. Phase 3: Extra pytest tests (files NOT in TESTS list)
#     These are blocklisted by discover_tests.py (fx/, quantization/, etc.)
#     but whitelisted by case_paths_ci.yml.
# ==============================================================================

PHASE3_STATUS=0
if [ -s "${EXTRA_FILE}" ] && { [ "${RUN_MODE}" = "regular" ] || [ "${RUN_MODE}" = "all" ]; }; then
    echo ""
    echo "=== Phase 3: Extra pytest tests (${EXTRA_COUNT} files) ==="
    echo ""

    EXTRA_FILES=$(tr '\n' ' ' < "${EXTRA_FILE}" | sed 's/ *$//')
    mkdir -p test/test-reports/extra

    set +e
    ${PYTHON} -m pytest \
        -p pytest_disabled_testcases_plugin \
        ${EXTRA_FILES} \
        --timeout 600 \
        -v --tb=short \
        --junitxml=test/test-reports/extra/extra_pytest.xml \
        2>&1 | tee /tmp/run_test_phase3.log
    PHASE3_STATUS=${PIPESTATUS[0]}
    set -e

    echo "Phase 3 exit: ${PHASE3_STATUS}"
fi

# ==============================================================================
# 11. Summary
# ==============================================================================

echo ""
echo "=== NPU Test Summary ==="
echo "  Phase 1 (regular):      ${PHASE1_STATUS}"
echo "  Phase 2 (distributed):  ${PHASE2_STATUS}"
echo "  Phase 3 (extra pytest): ${PHASE3_STATUS}"

FINAL=0
[ "${PHASE1_STATUS}" -ne 0 ] && FINAL=${PHASE1_STATUS}
[ "${PHASE2_STATUS}" -ne 0 ] && FINAL=${PHASE2_STATUS}
[ "${PHASE3_STATUS}" -ne 0 ] && FINAL=${PHASE3_STATUS}

echo "  Final:                  ${FINAL}"

if [ -d test/test-reports ]; then
    echo ""
    echo "Report files:"
    find test/test-reports -name "*.xml" -type f | head -20
fi

exit ${FINAL}
