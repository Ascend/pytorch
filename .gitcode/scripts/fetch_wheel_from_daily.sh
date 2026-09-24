#!/usr/bin/env bash
# ==============================================================================
# fetch_wheel_from_daily.sh — 从最近一次成功的日流水线下载 wheel
# ==============================================================================
# 功能:
#   当 PR 只改了测试文件 (need_rebuild=false) 时, 直接下载日流水线最新
#   build job 产出的 dist/*.whl, 跳过 40 分钟编译。
#
# 原理:
#   GitLab API: /api/v4/projects/:id/pipelines?ref=<branch>&status=success
#   → 找到最新的成功 pipeline
#   → /api/v4/projects/:id/pipelines/:pipeline_id/jobs?name=build
#   → 下载 artifacts
#
# 用法:
#   bash fetch_wheel_from_daily.sh \
#     --branch <daily-pipeline-branch> \
#     --job-name build \
#     --output-dir dist
#
# 环境变量:
#   GITLAB_TOKEN — API 访问 token (需在 GitLab Settings → CI/CD → Variables 配置)
#   CI_API_V4_URL — GitLab API 地址 (GitLab Runner 自动提供)
#   CI_PROJECT_ID — 项目 ID (GitLab Runner 自动提供)
# ==============================================================================

set -euo pipefail

BRANCH="${CI_COMMIT_BRANCH:-master}"
JOB_NAME="build"
OUTPUT_DIR="dist"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --branch <ref>    日流水线运行的分支 (默认: CI_COMMIT_BRANCH 或 master)
  --job-name <name> 要下载的 job 名称 (默认: build)
  --output-dir <dir> 输出目录 (默认: dist)
  -h, --help        Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --branch)     BRANCH="$2"; shift 2 ;;
        --job-name)   JOB_NAME="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *) echo "[ERROR] Unknown option: $1"; usage ;;
    esac
done

echo "=== Fetching latest successful wheel from daily pipeline ==="
echo "  Branch: ${BRANCH}"
echo "  Job: ${JOB_NAME}"
echo "  Output: ${OUTPUT_DIR}"

# ------------------------------------------------------------------
# Step 1: 找到最新成功的 pipeline
# ------------------------------------------------------------------
echo "Finding latest successful pipeline on branch '${BRANCH}'..."

PIPELINES=$(curl -sS --header "PRIVATE-TOKEN: ${GITLAB_TOKEN:-}" \
    "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/pipelines?ref=${BRANCH}&status=success&per_page=5" \
    2>/dev/null || echo "[]")

PIPELINE_ID=$(echo "${PIPELINES}" | python3 -c "
import json,sys
try:
    pipelines = json.load(sys.stdin)
    if pipelines:
        print(pipelines[0]['id'])
except:
    print('')
" 2>/dev/null || echo "")

if [ -z "${PIPELINE_ID}" ]; then
    echo "[ERROR] No successful pipeline found on branch '${BRANCH}'"
    exit 1
fi
echo "Latest successful pipeline ID: ${PIPELINE_ID}"

# ------------------------------------------------------------------
# Step 2: 找到该 pipeline 中的 build job
# ------------------------------------------------------------------
echo "Finding job '${JOB_NAME}' in pipeline ${PIPELINE_ID}..."

JOBS=$(curl -sS --header "PRIVATE-TOKEN: ${GITLAB_TOKEN:-}" \
    "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/pipelines/${PIPELINE_ID}/jobs?scope[]=success" \
    2>/dev/null || echo "[]")

JOB_ID=$(echo "${JOBS}" | python3 -c "
import json,sys
try:
    jobs = json.load(sys.stdin)
    for j in jobs:
        if j.get('name') == '${JOB_NAME}':
            print(j['id'])
            break
except:
    print('')
" 2>/dev/null || echo "")

if [ -z "${JOB_ID}" ]; then
    echo "[ERROR] Job '${JOB_NAME}' not found or not successful in pipeline ${PIPELINE_ID}"
    exit 1
fi
echo "Build job ID: ${JOB_ID}"

# ------------------------------------------------------------------
# Step 3: 下载 artifacts
# ------------------------------------------------------------------
echo "Downloading artifacts from job ${JOB_ID}..."

mkdir -p /tmp/wheel-artifacts
curl -sS -L --header "PRIVATE-TOKEN: ${GITLAB_TOKEN:-}" \
    "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/jobs/${JOB_ID}/artifacts" \
    -o /tmp/wheel-artifacts/artifacts.zip 2>/dev/null

if [ ! -f /tmp/wheel-artifacts/artifacts.zip ] || [ ! -s /tmp/wheel-artifacts/artifacts.zip ]; then
    echo "[ERROR] Failed to download artifacts from job ${JOB_ID}"
    exit 1
fi

# ------------------------------------------------------------------
# Step 4: 解压, 只取 dist/*.whl
# ------------------------------------------------------------------
echo "Extracting wheel from artifacts..."
mkdir -p /tmp/wheel-artifacts/extracted
unzip -q -o /tmp/wheel-artifacts/artifacts.zip -d /tmp/wheel-artifacts/extracted

mkdir -p "${OUTPUT_DIR}"
WHEEL_COUNT=$(find /tmp/wheel-artifacts/extracted -name "*.whl" -exec cp {} "${OUTPUT_DIR}/" \; -exec echo {} \; | wc -l)

if [ "${WHEEL_COUNT}" -eq 0 ]; then
    echo "[ERROR] No .whl file found in artifacts from job ${JOB_ID}"
    echo "Artifact contents:"
    ls -la /tmp/wheel-artifacts/extracted/
    exit 1
fi

echo "=== Done: ${WHEEL_COUNT} wheel(s) copied to ${OUTPUT_DIR}/ ==="
ls -lh "${OUTPUT_DIR}/"

# 清理
rm -rf /tmp/wheel-artifacts

exit 0
