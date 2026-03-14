# HCCL AllReduce 示例

这是一个独立的HCCL集合通信测试用例，用于验证NPU设备上的ProcessGroupHCCL功能。

## 文件说明

- `allreduce_hccl.cpp` - HCCL allreduce示例代码
- `CMakeLists.txt` - 独立的CMake构建文件
- `libtorch_hccl.sh` - 运行脚本

## 构建

### 1. 环境准备

CMakeLists.txt会自动从Python环境检测libtorch的安装路径。

**自动检测**（推荐）:
- 确保Python环境中已安装torch
- CMake会自动从Python获取路径

**手动指定**（可选）:
```bash
# 如果自动检测失败，可以手动设置环境变量
export TORCH_INSTALL_DIR=/path/to/torch  # 通常在 site-packages/torch
export TORCH_NPU_INSTALL_DIR=/path/to/torch_npu
```

### 2. 构建示例

```bash
cd examples/libtorch_hccl
mkdir build && cd build
cmake ..
make
```

## 运行

### 方法:

```bash
cd examples/libtorch_hccl
chmod +x libtorch_hccl.sh
./libtorch_hccl.sh
```

自定义进程数:
```bash
NUM_PROCESSES=4 ./libtorch_hccl.sh
```

## 预期输出

```
启动HCCL allreduce示例: rank=0, size=2
NPU设备 0 初始化完成
在NPU设备 0 上创建了 10 个张量
已提交 10 个allreduce操作
所有操作已完成!
张量 0 第一个元素: 2.0
张量 1 第一个元素: 2.0
张量 2 第一个元素: 2.0
HCCL allreduce示例运行成功!
```

值 `2.0` 是2个进程对全1张量进行allreduce求和的结果 (1+1=2)。

## 代码说明

示例演示了以下关键步骤：

1. **NPU设备初始化**: 使用 `torch_npu::init_npu("npu:0")` 初始化设备
2. **创建ProcessGroupHCCL**: 用于NPU集合通信
3. **创建张量**: 在NPU设备上创建张量
4. **执行allreduce**: 对张量进行全局归约操作
5. **验证结果**: 检查归约结果是否正确
6. **资源清理**: 调用 `torch_npu::finalize_npu()` 释放资源

## 注意事项

- 确保NPU设备可用且CANN已正确安装
- 使用NPU设备结束后必须调用 `torch_npu::finalize_npu()` 释放资源
- 所有进程需要同时启动才能正常工作
- libtorch通常安装在Python的site-packages/torch目录下
- libtorch_npu通常使用build_libtorch_npu.py脚本构建安装
