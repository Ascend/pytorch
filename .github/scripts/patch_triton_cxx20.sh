#!/usr/bin/env bash
# EXPERIMENTAL (patch_triton_cxx20): triton-ascend's _build_npu_ext (and
# the PCH precompile path on older installs) hardcodes -std=c++17, while
# the nightly torch build's ATen headers error out below C++20 (ATen.h:
# "#error C++20 or later compatible compiler is required to use ATen").
# Every npu_utils.so build therefore fails deterministically, which on
# the graph shards showed up as 549 visible compile failures, 34
# per-case 1200s timeouts (the g++ subprocess occasionally stalls
# instead of failing fast), and ~55h of swallowed compile tax per run -
# triton_backend()'s functools.cache only memoizes success, so every
# codegen retries the doomed build.
#
# This script rewrites the installed backend files in place and
# prebuilds npu_utils.so ONCE, outside the per-case timeout budget and
# ahead of the workers' concurrent first codegens. The triton disk cache
# key is md5(npu_utils.cpp) and is independent of -std, so every later
# per-case pytest.main() call hits the cache. Fails fast on any anchor
# drift: a silently skipped patch would poison the experiment's verdict.
#
# Kept as a script FILE instead of an inline workflow run block because
# the k8s step wrapper failed to hand a ~90-line inline script to the
# container (run 39398824822: every shard died in ~1s with no output),
# while it runs ci-scripts/.github/scripts/* files fine.

set -euo pipefail

source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
export LD_LIBRARY_PATH="$(python3 -c 'import sys; print(sys.prefix)')/lib:${LD_LIBRARY_PATH:-}"

# Guard the triton import: capture its stderr into the variable and
# re-emit via stdout - the step wrapper may drop stderr on early
# failure, which would otherwise turn a diagnosis into a silent exit.
TRITON_OUT="$(python3 -c 'import triton, os; print(os.path.dirname(triton.__file__))' 2>&1)" || {
  echo "::error::import triton failed in this test image:"
  echo "${TRITON_OUT}"
  echo "::error::patch_triton_cxx20 only applies to images with triton-ascend installed; disable it for this pipeline"
  exit 1
}
TRITON_DIR="${TRITON_OUT}"
BACKEND_DIR="${TRITON_DIR}/backends/ascend"
echo "triton package dir: ${TRITON_DIR}"
python3 -c 'import triton; print("triton version:", getattr(triton, "__version__", "unknown"))' || true

if [ ! -d "${BACKEND_DIR}" ]; then
  echo "::error::triton-ascend backend not found at ${BACKEND_DIR}; image layout changed - refusing to run the experiment unpatched"
  exit 1
fi

# Compilers triton's _get_cxx() may pick ($CC > clang++ > g++):
# C++20 needs gcc>=10 / clang>=10 - record what this image has.
echo "=== compiler inventory ==="
echo "CC=${CC:-<unset>}"
command -v clang++ >/dev/null 2>&1 && clang++ --version | head -1 || echo "clang++: not found"
command -v g++ >/dev/null 2>&1 && g++ --version | head -1 || echo "g++: not found"

# Directory-wide anchor rewrite: the installed triton-ascend has
# diverged from upstream main (main has a single -std=c++17 in utils.py;
# older installs also precompile PCH elsewhere), so patch EVERY
# hardcoded anchor instead of asserting one file.
mapfile -t TARGETS < <(grep -rlF --include='*.py' -- '"-std=c++17"' "${BACKEND_DIR}" || true)
ALREADY=$( { grep -rlF --include='*.py' -- '"-std=c++20"' "${BACKEND_DIR}" || true; } | wc -l)

if [ "${#TARGETS[@]}" -eq 0 ]; then
  if [ "${ALREADY}" -gt 0 ]; then
    echo "already patched (-std=c++20 present in ${ALREADY} file(s)) - idempotent no-op"
  else
    echo "::error::no -std=c++17/-std=c++20 anchor under ${BACKEND_DIR}; triton-ascend layout drifted - manual review required"
    exit 1
  fi
else
  TOTAL=0
  for f in "${TARGETS[@]}"; do
    N=$(grep -c -F -- '"-std=c++17"' "${f}" || true)
    sed -i 's/"-std=c++17"/"-std=c++20"/g' "${f}"
    python3 -m py_compile "${f}"
    echo "patched ${f}: ${N} occurrence(s)"
    grep -n -F -- '"-std=c++20"' "${f}"
    TOTAL=$((TOTAL + N))
  done
  echo "total replacements: ${TOTAL} in ${#TARGETS[@]} file(s)"
  if grep -rqF --include='*.py' -- '"-std=c++17"' "${BACKEND_DIR}"; then
    echo "::error::c++17 residue remains after patch"
    exit 1
  fi
fi

# Prebuild npu_utils.so NOW: without this, every worker process retries
# the (c++20) build at its first codegen, concurrently, before any of
# them can see the others' disk cache entry. A failure here IS the
# experiment's verdict - surfaced in minute 1 instead of after a full
# nightly run.
echo "=== prebuilding npu_utils.so with -std=c++20 (first and only build in this container) ==="
time python3 -c "
import torch
import torch_npu
from triton.runtime.driver import driver
target = driver.active.get_current_target()
print('triton driver initialized:', target)
print('npu_utils.so prebuilt OK')
"
