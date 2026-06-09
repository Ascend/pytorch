#!/bin/bash
# ==============================================================================
# detect_changed_patches.sh
#
# Detect changed patch files in test_upstream/ and derive corresponding test files.
#
# Environment inputs (set by GitHub Actions workflow):
#   EVENT_NAME         - "pull_request" or "workflow_dispatch"
#   BASE_SHA           - PR base commit SHA (pull_request only)
#   HEAD_SHA           - PR head commit SHA (pull_request only)
#   BASE_REF           - PR target branch ref (pull_request only)
#   INPUT_PATCH_FILES  - comma-separated patch paths (workflow_dispatch only)
#
# Outputs (written to $GITHUB_OUTPUT):
#   test_patches       - comma-separated test_upstream/test/ patch paths
#   torch_patches      - comma-separated test_upstream/torch/ patch paths
#   test_files         - comma-separated derived test file names
#   has_test_changes   - "true" or "false"
#   has_torch_changes  - "true" or "false"
#   changed_summary    - one of: test+torch, test-only, torch-only, none
# ==============================================================================
set -euo pipefail

# ------------------------------------------------------------------
# Step 1: Collect changed files from the trigger source
# ------------------------------------------------------------------
if [ "${EVENT_NAME}" = "pull_request" ]; then
    echo "=== PR Event: detecting changes ==="
    echo "Base SHA: ${BASE_SHA:-unknown}"
    echo "Head SHA: ${HEAD_SHA:-unknown}"

    # HEAD is the PR merge commit (checked out by actions/checkout).
    # HEAD^1 = base branch, HEAD^2 = PR head branch.
    # Use three-dot (...) to show only PR-side changes relative to merge-base,
    # excluding upstream changes that happened after the fork point.
    if git cat-file -e HEAD^2 2>/dev/null; then
        echo "Using merge commit parents: HEAD^1...HEAD^2 (PR-side changes only)"
        CHANGED_FILES=$(git diff --name-only HEAD^1...HEAD^2 -- 'test_upstream/' 2>/dev/null || true)
    else
        echo "Merge parents not available, falling back to base/head diff"
        git fetch --no-tags origin "${BASE_REF}" 2>/dev/null || true
        CHANGED_FILES=$(git diff --name-only \
            "${BASE_SHA}" "${HEAD_SHA}" \
            -- 'test_upstream/' 2>/dev/null || true)
    fi
else
    echo "=== Manual Dispatch: using input ==="
    CHANGED_FILES="${INPUT_PATCH_FILES:-}"
fi

echo ""
echo "Raw changed files:"
echo "${CHANGED_FILES}" | sed 's/^/  /'

# ------------------------------------------------------------------
# Step 2: Normalize (handle comma-separated input from dispatch)
# ------------------------------------------------------------------
CHANGED_FILES=$(echo "${CHANGED_FILES}" | tr ',' '\n' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')

# ------------------------------------------------------------------
# Step 3: Classify patches and derive test files
# ------------------------------------------------------------------
TEST_PATCHES=""
TORCH_PATCHES=""
TEST_FILES=""

while IFS= read -r f; do
    [ -z "$f" ] && continue

    case "$f" in
        test_upstream/test/*.patch|test_upstream/test/*.diff)
            # Derive test file by stripping prefix + suffix:
            #   test_upstream/test/test_autograd.py.patch → test_autograd.py
            #   test_upstream/test/ao/test_foo.py.patch  → ao/test_foo.py
            #   test_upstream/test/inductor/test_minifer.diff → inductor/test_minifer.py
            TEST_FILE=$(echo "$f" | sed 's|^test_upstream/test/||; s|\.patch$||; s|\.diff$||')
            # Ensure .py extension for cases where patch suffix was on bare name (e.g. test_foo.diff)
            [[ "$TEST_FILE" != *.py ]] && TEST_FILE="${TEST_FILE}.py"
            TEST_PATCHES="${TEST_PATCHES}${f},"
            TEST_FILES="${TEST_FILES}${TEST_FILE},"
            echo "  → test patch: $f → test file: ${TEST_FILE}"
            ;;
        test_upstream/torch/*.patch|test_upstream/torch/*.diff)
            TORCH_PATCHES="${TORCH_PATCHES}${f},"
            echo "  → torch patch: $f (no direct test mapping)"
            ;;
        *)
            echo "  → skipped: $f (not a patch file)"
            ;;
    esac
done <<< "${CHANGED_FILES}"

# Remove trailing commas
TEST_PATCHES="${TEST_PATCHES%,}"
TORCH_PATCHES="${TORCH_PATCHES%,}"
TEST_FILES="${TEST_FILES%,}"

# Determine change type flags
HAS_TEST="false"
HAS_TORCH="false"
[ -n "${TEST_PATCHES}" ] && HAS_TEST="true"
[ -n "${TORCH_PATCHES}" ] && HAS_TORCH="true"

# Determine summary string
if [ "${HAS_TEST}" = "true" ] && [ "${HAS_TORCH}" = "true" ]; then
    CHANGED_SUMMARY="test+torch"
elif [ "${HAS_TEST}" = "true" ]; then
    CHANGED_SUMMARY="test-only"
elif [ "${HAS_TORCH}" = "true" ]; then
    CHANGED_SUMMARY="torch-only"
else
    CHANGED_SUMMARY="none"
fi

# ------------------------------------------------------------------
# Step 4: Report and write outputs
# ------------------------------------------------------------------
echo ""
echo "=== Detection Result ==="
echo "test_patches=${TEST_PATCHES}"
echo "torch_patches=${TORCH_PATCHES}"
echo "test_files=${TEST_FILES}"
echo "has_test_changes=${HAS_TEST}"
echo "has_torch_changes=${HAS_TORCH}"
echo "changed_summary=${CHANGED_SUMMARY}"

{
    echo "test_patches=${TEST_PATCHES}"
    echo "torch_patches=${TORCH_PATCHES}"
    echo "test_files=${TEST_FILES}"
    echo "has_test_changes=${HAS_TEST}"
    echo "has_torch_changes=${HAS_TORCH}"
    echo "changed_summary=${CHANGED_SUMMARY}"
} >> "${GITHUB_OUTPUT}"

if [ "${HAS_TEST}" = "false" ] && [ "${HAS_TORCH}" = "false" ]; then
    echo ""
    echo "WARNING: No patch files detected in changed files."
    echo "If this is a PR, ensure it modifies .patch or .diff files under test_upstream/."
fi
