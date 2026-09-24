#!/usr/bin/env bash
# ==============================================================================
# detect_changed_tests.sh — 检测 MR 中变更的测试文件 + 是否需要重新编译
# ==============================================================================
# 功能:
#   1. 通过 git diff 找出 MR 中所有变更文件
#   2. 直接识别 test_*.py 变更 (TestFileStrategy)
#   3. 通过 classify_changed_tests.py 应用 CoreTest/DirectoryMapping/Op 策略
#   4. 合并两套结果 + 判断是否需要重新编译
#
# 输出 (写入 detect_result.env):
#   test_files       - 应跑的测试文件 (逗号分隔)
#   has_test_changes - "true" 或 "false"
#   need_rebuild     - "true" 或 "false"
#
# 策略来源: ascend-pytorch/ci/access_control/strategy/
#   TestFile / CoreTest / DirectoryMapping / Op
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Detecting changed files ==="

# ------------------------------------------------------------------
# Step 1: 获取 diff 范围
# ------------------------------------------------------------------
if [ -n "${CI_MERGE_REQUEST_DIFF_BASE_SHA:-}" ]; then
    BASE="${CI_MERGE_REQUEST_DIFF_BASE_SHA}"
    HEAD="${CI_COMMIT_SHA}"
    echo "MR detected — base: ${BASE:0:8}, head: ${HEAD:0:8}"
elif [ -n "${CI_COMMIT_BEFORE_SHA:-}" ] && [ "${CI_COMMIT_BEFORE_SHA}" != "0000000000000000000000000000000000000000" ]; then
    BASE="${CI_COMMIT_BEFORE_SHA}"
    HEAD="${CI_COMMIT_SHA}"
    echo "Push detected — before: ${BASE:0:8}, after: ${HEAD:0:8}"
else
    BASE="HEAD~1"
    HEAD="HEAD"
    echo "Fallback: comparing ${BASE}..${HEAD}"
fi

# ------------------------------------------------------------------
# Step 2: 全量变更列表
# ------------------------------------------------------------------
ALL_CHANGED=$(git diff --name-only "${BASE}" "${HEAD}" 2>/dev/null || true)
echo ""
echo "=== Changed files (${#ALL_CHANGED} lines) ==="
echo "${ALL_CHANGED}" | sed 's/^/  /'

# ------------------------------------------------------------------
# Step 3: 直接 test_*.py 识别 (TestFileStrategy)
# ------------------------------------------------------------------
DIRECT_TEST_FILES=""
while IFS= read -r f; do
    [ -z "$f" ] && continue
    case "$f" in
        test/**/test_*.py)
            FILENAME=$(echo "$f" | sed 's|^test/||; s|\.py$||')
            DIRECT_TEST_FILES="${DIRECT_TEST_FILES}${FILENAME},"
            echo "  → [test_file] $f"
            ;;
    esac
done <<< "${ALL_CHANGED}"
DIRECT_TEST_FILES="${DIRECT_TEST_FILES%,}"

# ------------------------------------------------------------------
# Step 4: 策略分类 (CoreTest / DirectoryMapping / Op)
#         将所有变更文件传给 Python 脚本 → 获取匹配的测试文件 + need_rebuild
# ------------------------------------------------------------------
echo ""
echo "=== Strategy-based classification ==="
CLASSIFY_OUTPUT=$(echo "${ALL_CHANGED}" | python3 "${SCRIPT_DIR}/classify_changed_tests.py" 2>&1)
echo "${CLASSIFY_OUTPUT}" | grep "^  \[" || true

# 从 classify 输出中提取变量
_classify_get() {
    echo "${CLASSIFY_OUTPUT}" | grep "^$1=" | head -1 | cut -d= -f2-
}

STRATEGY_TEST_FILES=$(_classify_get "test_files")
STRATEGY_HAS_CHANGES=$(_classify_get "has_test_changes")
STRATEGY_NEED_REBUILD=$(_classify_get "need_rebuild")

# ------------------------------------------------------------------
# Step 5: 合并结果
# ------------------------------------------------------------------
# 合并直接匹配 + 策略匹配 (去重)
MERGED=$(echo "${DIRECT_TEST_FILES},${STRATEGY_TEST_FILES}" | tr ',' '\n' | sort -u | grep -v '^$' | tr '\n' ',' | sed 's/,$//')

if [ -n "${MERGED}" ]; then
    has_test_changes="true"
    test_files="${MERGED}"
else
    has_test_changes="false"
    test_files=""
fi

# need_rebuild: 直接匹配 OR 策略判断
NEED_REBUILD="false"
# 检查是否有 C++/build 变更 (只改 torch_npu/ 下的 .py 不需要重编)
if echo "${ALL_CHANGED}" | grep -qE '(\.(cpp|h|cu)$|CMakeLists\.txt|ci/build\.sh|setup\.(py|cfg)|pyproject\.toml)'; then
    NEED_REBUILD="true"
fi
[ "${STRATEGY_NEED_REBUILD}" = "true" ] && NEED_REBUILD="true"

# ------------------------------------------------------------------
# Step 6: 输出结果
# ------------------------------------------------------------------
echo ""
echo "=== Detection Result ==="
echo "Direct test changes: ${DIRECT_TEST_FILES:-none}"
echo "Strategy test files: ${STRATEGY_TEST_FILES:-none}"
echo "Merged test files:   ${test_files:-none}"
echo "has_test_changes=${has_test_changes}"
echo "test_files=${test_files}"
echo "need_rebuild=${NEED_REBUILD}"

cat > detect_result.env <<EOF
has_test_changes=${has_test_changes}
test_files=${test_files}
need_rebuild=${NEED_REBUILD}
EOF

echo "Wrote detect_result.env"

if [ "${has_test_changes}" = "false" ]; then
    echo "No test files changed — downstream test job will be skipped."
fi
if [ "${NEED_REBUILD}" = "false" ] && [ "${has_test_changes}" = "true" ]; then
    echo "Only test files changed — build will be skipped (reuse daily wheel)."
fi
