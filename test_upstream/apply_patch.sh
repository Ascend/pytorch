#!/bin/bash

# ====================== 【重要】配置项 ======================
# 源码根目录相对于patch目录的路径 对于我们的项目来说是根目录相对于test_upstream的路径，也就是..
RELATIVE_TO_ROOT=..
# ==========================================================

# 自动获取当前脚本所在目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# 自动计算源码根目录
ROOT_DIR=$(cd "$SCRIPT_DIR/$RELATIVE_TO_ROOT" &>/dev/null && pwd)
PATCH_DIR="$SCRIPT_DIR"

# 检查根目录是否合法
if [ ! -d "$ROOT_DIR" ]; then
    echo "错误：无法定位源码根目录！"
    echo "请检查 RELATIVE_TO_ROOT 配置"
    exit 1
fi

echo "================================================"
echo "          自动批量应用 patch"
echo "================================================"
echo "源码根目录：$ROOT_DIR"
echo "Patch 目录：$PATCH_DIR"
echo "================================================"

# 进入源码根目录
cd "$ROOT_DIR" || exit 1

# 递归查找所有 patch 文件并排序
PATCH_FILES=$(find "$PATCH_DIR" -type f \( -name "*.patch" -o -name "*.diff" \) | sort)

if [ -z "$PATCH_FILES" ]; then
    echo "未找到任何 .patch / .diff 文件"
    exit 0
fi

count=0
success=0
fail=0

# 逐个应用
for patch in $PATCH_FILES; do
    count=$((count+1))
    echo -e "\n[$count] 应用：$patch"

    patch -p1 --no-backup-if-mismatch -f < "$patch"

    if [ $? -eq 0 ]; then
        echo "成功"
        success=$((success+1))
    else
        echo "失败！停止执行"
        fail=$((fail+1))
        exit 1
    fi
done

echo -e "\n================================================"
echo "                      全部完成"
echo "================================================"
echo "总计：$count 个"
echo "成功：$success 个"
echo "失败：$fail 个"
echo "================================================"