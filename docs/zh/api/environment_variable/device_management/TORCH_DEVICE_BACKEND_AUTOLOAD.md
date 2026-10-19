# TORCH\_DEVICE\_BACKEND\_AUTOLOAD

## 功能描述

通过此环境变量可控制`import torch`时是否自动加载树外（out-of-tree）设备后端。该变量由PyTorch定义和解析（PyTorch 2.5.0起引入），TorchNPU以pip方式安装时，会在entry point分组`torch.backends`下注册加载入口，接入该机制。

- 未配置或配置为“1”时：`import torch`自动导入torch_npu并注册NPU后端，无需显式执行`import torch_npu`。
- 配置为“0”（或其他任意值）时：关闭自动加载，使用NPU前需显式执行`import torch_npu`。

该变量对应PyTorch的[TORCH_DEVICE_BACKEND_AUTOLOAD](https://docs.pytorch.org/docs/stable/miscellaneous_environment_variables.html)，变量名称、默认值和取值语义一致。

> [!NOTE]
>
> - torch_npu导入过程中会临时将该变量置为“0”以避免与`import torch`产生循环依赖，导入完成后自动恢复原值，子进程将继承恢复后的取值。
> - CANN设备初始化为懒加载：`import torch`仅导入torch_npu并注册NPU后端，首次调用NPU相关接口时才执行初始化，设备相关环境变量（如`ASCEND_RT_VISIBLE_DEVICES`）需在首次使用NPU前设置。

## 配置示例

关闭自动加载，改为显式导入torch_npu（适用于需显式控制导入和CANN初始化时机的场景）：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

## 使用约束

- 此环境变量需在进程启动前（首次`import torch`前）设置，进程运行过程中修改不会生效。
- 该环境变量仅在值严格等于“1”时开启自动加载，未配置时默认开启；配置为其他任意值（如“true”、“on”）均视为关闭。
- 该机制依赖PyTorch 2.5.0及以上版本的设备后端自动加载能力，且torch_npu需以pip方式安装（安装时会在`torch.backends`分组下自动注册加载入口）；直接引用源码目录（如设置PYTHONPATH）时无安装元数据，自动加载不生效。
- 同时安装多个加速器后端（如GPU版PyTorch与torch_npu）时，自动加载会在`import torch`时报错。此时可配置为“0”避免自动加载torch_npu，使用其他加速器；如需使用NPU，应安装CPU版本的PyTorch。

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id4 -->
