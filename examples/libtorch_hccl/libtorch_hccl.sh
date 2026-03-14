#!/bin/bash

set -e

# 默认使用2个进程
NUM_PROCESSES=${NUM_PROCESSES:-2}

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检查二进制文件是否存在
if [ ! -f "${SCRIPT_DIR}/build/example_allreduce_hccl" ]; then
    echo "错误: 未找到example_allreduce_hccl二进制文件"
    echo "请先构建项目:"
    echo "  cd ${SCRIPT_DIR}"
    echo "  mkdir build && cd build"
    echo "  cmake .."
    echo "  make"
    exit 1
fi

echo "运行HCCL allreduce示例，进程数: ${NUM_PROCESSES}"
echo "二进制文件: ${SCRIPT_DIR}/build/example_allreduce_hccl"

# 清理之前的临时文件
rm -f /tmp/c10d_hccl_example

# 启动多个进程
pids=()
for ((rank=0; rank<NUM_PROCESSES; rank++)); do
    echo "启动进程 ${rank}"
    RANK=${rank} SIZE=${NUM_PROCESSES} \
        "${SCRIPT_DIR}/build/example_allreduce_hccl" &
    pids+=($!)
done

# 等待所有进程完成
for pid in "${pids[@]}"; do
    wait ${pid}
done

echo "所有进程已完成!"
# 清理临时文件
rm -f /tmp/c10d_hccl_example
