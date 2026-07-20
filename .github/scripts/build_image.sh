#!/bin/bash
#
# build_image.sh - 构建 PyTorch NPU Docker 镜像
#
# 功能：按 CANN 版本构建镜像，镜像预装多 Python 版本，通过环境变量切换
#
# 使用方式：
#   ./build_image.sh --cann-version 9.0
#   ./build_image.sh --cann-version 9.0.0-beta.2 --push
#   ./build_image.sh --list-versions     # 查看支持的 CANN 版本
#

set -euo pipefail

# ============================================================
# CANN 版本映射表
# 每个版本对应三个包的下载 URL
# ============================================================

declare -A CANN_VERSIONS=(
    # 版本号 -> toolkit|a3_ops|nnal 的 URL
    # 注意：OBS 上当前只有 9.0.0-beta.2 版本的包
    ["9.0"]="https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/cann-package/20260330/Ascend-cann-toolkit_9.0.0-beta.2_linux-aarch64.run|https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/cann-package/20260330/Ascend-cann-A3-ops_9.0.0-beta.2_linux-aarch64.run|https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/cann-package/20260330/Ascend-cann-nnal_9.0.0-beta.2_linux-aarch64.run"

    ["9.0.0-beta.2"]="https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/cann-package/20260330/Ascend-cann-toolkit_9.0.0-beta.2_linux-aarch64.run|https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/cann-package/20260330/Ascend-cann-A3-ops_9.0.0-beta.2_linux-aarch64.run|https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/cann-package/20260330/Ascend-cann-nnal_9.0.0-beta.2_linux-aarch64.run"

    ["8.0"]="https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/cann-package/20250101/Ascend-cann-toolkit_8.0.RC3_linux-aarch64.run|https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/cann-package/20250101/Ascend-cann-A3-ops_8.0.RC3_linux-aarch64.run|https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/cann-package/20250101/Ascend-cann-nnal_8.0.RC3_linux-aarch64.run"
)

# Stable 版本标记（用于 latest 标签）
CANN_STABLE="9.0"

# 预装的 Python 版本列表
PYTHON_VERSIONS=("3.10" "3.11" "3.12" "3.13")

# manylinux 对应的 Python 目录名映射
declare -A PYTHON_DIR_MAP=(
    ["3.10"]="cp310-cp310"
    ["3.11"]="cp311-cp311"
    ["3.12"]="cp312-cp312"
    ["3.13"]="cp313-cp313"
)

# ============================================================
# 默认配置
# ============================================================

DEFAULT_REGISTRY="quay.io"
DEFAULT_QUAY_ORG="kerer"
DEFAULT_IMAGE_NAME="pytorch"

# 参数变量
CANN_VERSION_INPUT=""
REGISTRY=""
QUAY_ORG=""
IMAGE_NAME=""
PUSH_IMAGE=false
FORCE_BUILD=false
VERBOSE=false
LIST_VERSIONS=false

# ============================================================
# 日志函数
# ============================================================

log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

log_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo "[VERBOSE] $1"
    fi
}

# ============================================================
# 显示帮助信息
# ============================================================

show_help() {
    cat << EOF
用法: $0 [OPTIONS]

构建支持不同 CANN 版本的 PyTorch NPU Docker 镜像。

镜像特性：
  - 预装多个 Python 版本 (3.10/3.11/3.12/3.13)
  - 通过环境变量切换 Python 版本
  - 按 CANN 版本构建镜像

CANN 参数:
  --cann-version VERSION      CANN 版本号（支持简化版或完整版）
                              简化版: 9.0, 8.0
                              完整版: 9.0.0-beta.2
  --list-versions             显示支持的 CANN 版本列表

镜像参数:
  --registry REGISTRY         Docker registry 地址 (默认: quay.io)
  --quay-org ORG              Quay.io 组织名 (默认: kerer)
  --image-name NAME           镜像名称 (默认: pytorch)

构建选项:
  --push                      构建后推送镜像到 registry
  --force                     强制构建，即使镜像已存在
  --verbose                   显示详细日志

Python 版本切换（运行时）:
  镜像预装多个 Python 版本，使用时通过环境变量切换：
  export PATH=/opt/python/cp311-cp311/bin:\$PATH  # 使用 Python 3.11
  export PATH=/opt/python/cp312-cp312/bin:\$PATH  # 使用 Python 3.12

示例:
  $0 --cann-version 9.0
  $0 --cann-version 9.0.0-beta.2 --push
  $0 --list-versions

支持的 CANN 版本：
$(show_supported_versions)

EOF
}

show_supported_versions() {
    echo "简化版本      完整版本"
    echo "-----------   ----------------"
    for version in "${!CANN_VERSIONS[@]}"; do
        if [[ ! "$version" =~ -beta ]] && [[ ! "$version" =~ -rc ]]; then
            echo "$version       (完整版见映射表)"
        fi
    done
    echo ""
    echo "完整版本示例："
    echo "  9.0.0-beta.2"
    echo "  8.0.RC3"
}

# ============================================================
# 解析命令行参数
# ============================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --cann-version)
                CANN_VERSION_INPUT="$2"
                shift 2
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --quay-org)
                QUAY_ORG="$2"
                shift 2
                ;;
            --image-name)
                IMAGE_NAME="$2"
                shift 2
                ;;
            --push)
                PUSH_IMAGE=true
                shift
                ;;
            --force)
                FORCE_BUILD=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --list-versions)
                LIST_VERSIONS=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # 设置默认值
    REGISTRY="${REGISTRY:-$DEFAULT_REGISTRY}"
    QUAY_ORG="${QUAY_ORG:-$DEFAULT_QUAY_ORG}"
    IMAGE_NAME="${IMAGE_NAME:-$DEFAULT_IMAGE_NAME}"

    # 显示版本列表
    if [[ "$LIST_VERSIONS" == "true" ]]; then
        echo "支持的 CANN 版本："
        echo ""
        for version in "${!CANN_VERSIONS[@]}"; do
            echo "  - $version"
        done
        echo ""
        echo "Stable 版本（用于 latest 标签）: $CANN_STABLE"
        exit 0
    fi

    # 验证参数
    if [[ -z "$CANN_VERSION_INPUT" ]]; then
        log_error "必须指定 --cann-version 或使用 --list-versions"
        show_help
        exit 1
    fi
}

# ============================================================
# 解析 CANN 版本
# ============================================================

parse_cann_version() {
    local input="$CANN_VERSION_INPUT"

    log_verbose "解析 CANN 版本: $input"

    # 检查版本是否在映射表中
    if [[ ! -v CANN_VERSIONS[$input] ]]; then
        log_error "不支持的 CANN 版本: $input"
        log_info "支持的版本: ${!CANN_VERSIONS[*]}"
        log_info "使用 --list-versions 查看完整列表"
        exit 1
    fi

    # 解析 URL
    local urls="${CANN_VERSIONS[$input]}"
    CANN_TOOLKIT_URL=$(echo "$urls" | cut -d'|' -f1)
    CANN_A3OPS_URL=$(echo "$urls" | cut -d'|' -f2)
    CANN_NNAL_URL=$(echo "$urls" | cut -d'|' -f3)

    # 提取版本号（去掉 beta/rc 后缀）
    CANN_VERSION_FULL="$input"
    CANN_VERSION_MAJOR=$(echo "$input" | sed 's/-beta.*//' | sed 's/-rc.*//')

    # 判断是否为 stable 版本
    if [[ "$CANN_VERSION_MAJOR" == "$CANN_STABLE" ]]; then
        IS_STABLE="true"
    else
        IS_STABLE="false"
    fi

    log_verbose "Toolkit URL: $CANN_TOOLKIT_URL"
    log_verbose "A3-ops URL: $CANN_A3OPS_URL"
    log_verbose "NNAL URL: $CANN_NNAL_URL"
    log_verbose "Full version: $CANN_VERSION_FULL"
    log_verbose "Major version: $CANN_VERSION_MAJOR"
    log_verbose "Is stable: $IS_STABLE"
}

# ============================================================
# 生成镜像标签
# ============================================================

generate_tags() {
    local timestamp=$(date +%Y%m%d)
    local tags=()

    # 提取大版本号（去掉 patch 号，但保留 beta/rc）
    # 例如：9.0.0-beta.2 → 9.0，9.0 → 9.0，8.0.RC3 → 8.0
    local cann_major

    # 如果版本号已经是简化格式（没有第二个点），则保持原样
    if [[ "$CANN_VERSION_FULL" =~ ^[0-9]+\.[0-9]+$ ]]; then
        cann_major="$CANN_VERSION_FULL"
    else
        # 提取前两位数字（去掉 patch 号和 beta/rc 后缀）
        cann_major=$(echo "$CANN_VERSION_FULL" | grep -oP '^[0-9]+\.[0-9]+')
    fi

    # 1. 完整版本标签（带时间戳）- 用于追溯
    tags+=("cann${CANN_VERSION_FULL}-${timestamp}")

    # 2. 标准版本标签（无时间戳）- 用于日常使用
    # 如果输入已经是简化版本，则跳过完整版本标签，避免重复
    if [[ "$CANN_VERSION_FULL" != "$cann_major" ]]; then
        tags+=("cann${CANN_VERSION_FULL}")
    fi

    # 3. 大版本简化标签 - 用于快速识别
    tags+=("cann${cann_major}")

    # 4. latest 标签（仅 stable 版本）
    if [[ "$IS_STABLE" == "true" ]]; then
        tags+=("latest")
        tags+=("cann-latest")
        tags+=("cann${cann_major}-latest")
    fi

    # 输出所有标签
    for tag in "${tags[@]}"; do
        echo "$tag"
    done
}

# ============================================================
# 构建镜像
# ============================================================

build_image() {
    log_info "=========================================="
    log_info "构建镜像: CANN $CANN_VERSION_FULL"
    log_info "=========================================="

    log_info "预装 Python 版本: ${PYTHON_VERSIONS[*]}"

    # 生成镜像标签
    local tags=$(generate_tags)
    local tag_args=""
    while IFS= read -r tag; do
        tag_args+=" --tag ${REGISTRY}/${QUAY_ORG}/${IMAGE_NAME}:${tag}"
    done <<< "$tags"

    log_info "镜像标签:"
    while IFS= read -r tag; do
        log_info "  - ${REGISTRY}/${QUAY_ORG}/${IMAGE_NAME}:${tag}"
    done <<< "$tags"

    # 检查镜像是否已存在（除非强制构建）
    if [[ "$FORCE_BUILD" == "false" && "$PUSH_IMAGE" == "true" ]]; then
        local first_tag=$(echo "$tags" | head -n1)
        if docker pull "${REGISTRY}/${QUAY_ORG}/${IMAGE_NAME}:${first_tag}" &>/dev/null; then
            log_info "镜像已存在，跳过构建: ${REGISTRY}/${QUAY_ORG}/${IMAGE_NAME}:${first_tag}"
            return 0
        fi
    fi

    # 确认需要构建，执行登录（如果需要推送）
    # 如果环境变量 SKIP_DOCKER_LOGIN=true，则跳过（用于 CI，已通过 login-action 登录）
    if [[ "$PUSH_IMAGE" == "true" ]]; then
        if [[ "${SKIP_DOCKER_LOGIN:-false}" != "true" ]]; then
            login_registry
        else
            log_verbose "跳过登录（SKIP_DOCKER_LOGIN=true）"
        fi
    fi

    # Dockerfile 路径
    # 使用 git 获取项目根目录（更可靠）
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local project_root

    # 尝试使用 git 获取项目根目录
    if git rev-parse --show-toplevel &>/dev/null; then
        project_root="$(git rev-parse --show-toplevel)"
    else
        # 如果不在 git 仓库中，从脚本目录向上推导
        project_root="$(cd "${script_dir}/.." && pwd)"
    fi

    local dockerfile_dir="${project_root}/.github/docker"
    local dockerfile="${dockerfile_dir}/pytorch-npu-builder.Dockerfile"

    log_verbose "Script dir: ${script_dir}"
    log_verbose "Project root: ${project_root}"
    log_verbose "Dockerfile dir: ${dockerfile_dir}"

    if [[ ! -f "$dockerfile" ]]; then
        log_error "Dockerfile 不存在: $dockerfile"
        exit 1
    fi

    log_verbose "Dockerfile: $dockerfile"

    # 构建参数（单行格式，避免换行符问题）
    local build_args="--build-arg CANN_TOOLKIT_URL=${CANN_TOOLKIT_URL} --build-arg CANN_A3OPS_URL=${CANN_A3OPS_URL} --build-arg CANN_NNAL_URL=${CANN_NNAL_URL} --build-arg CANN_VERSION=${CANN_VERSION_FULL}"

    # 构建命令（单行格式）
    local build_cmd="docker buildx build ${build_args} ${tag_args} --file ${dockerfile} --platform linux/arm64 ${dockerfile_dir}"

    if [[ "$PUSH_IMAGE" == "true" ]]; then
        build_cmd+=" --push"
    else
        build_cmd+=" --load"
    fi

    log_verbose "构建命令: $build_cmd"

    # 执行构建
    log_info "开始构建..."
    if ! eval "$build_cmd"; then
        log_error "构建失败"
        return 1
    fi

    log_info "构建成功"

    # 输出构建信息
    echo ""
    log_info "构建信息:"
    log_info "  CANN 版本: $CANN_VERSION_FULL"
    log_info "  CANN 大版本: $CANN_VERSION_MAJOR"
    log_info "  Stable: $IS_STABLE"
    log_info "  预装 Python: ${PYTHON_VERSIONS[*]}"
    log_info "  镜像地址:"
    while IFS= read -r tag; do
        log_info "    ${REGISTRY}/${QUAY_ORG}/${IMAGE_NAME}:${tag}"
    done <<< "$tags"

    echo ""
    log_info "使用方法："
    log_info "  docker run -it ${REGISTRY}/${QUAY_ORG}/${IMAGE_NAME}:cann${CANN_VERSION_MAJOR} bash"
    log_info "  # 切换 Python 版本："
    log_info "  export PATH=/opt/python/cp311-cp311/bin:\$PATH  # Python 3.11"
    log_info "  export PATH=/opt/python/cp312-cp312/bin:\$PATH  # Python 3.12"
    echo ""

    return 0
}

# ============================================================
# 检查依赖
# ============================================================

check_dependencies() {
    log_verbose "检查依赖..."

    # 检查 docker
    if ! command -v docker &>/dev/null; then
        log_error "未安装 docker"
        exit 1
    fi

    # 检查 docker buildx
    if ! docker buildx version &>/dev/null; then
        log_error "docker buildx 不可用"
        exit 1
    fi

    log_verbose "依赖检查通过"
}

# ============================================================
# 登录 registry
# ============================================================

login_registry() {
    if [[ "$PUSH_IMAGE" == "true" ]]; then
        log_info "登录 Registry: $REGISTRY"

        case "$REGISTRY" in
            quay.io)
                if [[ -z "${QUAY_USERNAME:-}" || -z "${QUAY_PASSWORD:-}" ]]; then
                    log_error "需要设置环境变量 QUAY_USERNAME 和 QUAY_PASSWORD"
                    exit 1
                fi
                docker login quay.io -u "$QUAY_USERNAME" --password-stdin <<< "$QUAY_PASSWORD"
                ;;
            ghcr.io)
                if [[ -z "${GITHUB_TOKEN:-}" ]]; then
                    log_error "需要设置环境变量 GITHUB_TOKEN"
                    exit 1
                fi
                echo "$GITHUB_TOKEN" | docker login ghcr.io -u "${GITHUB_ACTOR:-}" --password-stdin
                ;;
            *)
                log_error "不支持的 registry: $REGISTRY"
                exit 1
                ;;
        esac

        log_info "登录成功"
    fi
}

# ============================================================
# 主函数
# ============================================================

main() {
    parse_args "$@"
    check_dependencies
    parse_cann_version
    build_image
}

# 执行主函数
main "$@"