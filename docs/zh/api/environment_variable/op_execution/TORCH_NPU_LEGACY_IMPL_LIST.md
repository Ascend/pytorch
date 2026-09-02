# TORCH_NPU_LEGACY_IMPL_LIST

## 功能描述

该环境变量通过配置项指定需要使用旧版本实现（以下简称legacy实现）的功能，适用于产品升级或迁移到新一代硬件后需要保留原有执行行为的场景。未配置该环境变量时，相关功能使用当前产品的默认实现。

该环境变量的优先级高于`TORCH_NPU_USE_COMPATIBLE_IMPL`和`torch_npu.npu.use_compatible_impl()`全局开关。配置项加入列表后，其对应功能使用legacy实现；未加入列表时，对应功能不受该环境变量影响。

当前支持的配置项如下：

| 配置项 | 说明 |
| ------ | ---- |
| `randomness` | 用于控制已支持切换的原生PyTorch随机数API在NPU上的执行路径。 |

`randomness`当前适用于以下原生PyTorch API：

| API类别 | API |
| ------- | --- |
| 随机数生成接口 | `torch.rand`、`torch.rand_like`、`torch.randn`、`torch.randn_like`、`torch.randint`、`torch.randint_like`、`torch.normal` |
| Tensor原地接口 | `Tensor.uniform_`、`Tensor.normal_`、`Tensor.random_(from, to)`、`Tensor.random_(to)` |
| 参数初始化接口 | `torch.nn.init.uniform_`、`torch.nn.init.normal_`、`torch.nn.init.trunc_normal_`、`torch.nn.init.xavier_uniform_`、`torch.nn.init.xavier_normal_`、`torch.nn.init.kaiming_uniform_`、`torch.nn.init.kaiming_normal_`、`torch.nn.init.orthogonal_`、`torch.nn.init.sparse_` |

在<term>Ascend 950DT</term>上，上述随机数API默认使用与PyTorch原生社区完全对齐的实现。配置`randomness`后，相关API切换为与<term>Atlas A2 训练系列产品</term>、<term>Atlas A3 训练系列产品</term>一致的随机数实现。

## 配置示例

配置随机数API使用legacy实现：

```bash
export TORCH_NPU_LEGACY_IMPL_LIST=randomness
```

## 使用约束

- 必须在启动Python进程前配置该环境变量。
- 配置多个配置项时，使用英文逗号分隔。
- 配置项区分大小写。配置了不支持的配置项时会抛出异常，并在错误信息中提示当前支持的配置项。
- `randomness`需要配套使用支持随机数实现切换的CANN版本。CANN不支持该能力时，相关API保留当前产品的默认实现。

## 支持的型号

- <term>Ascend 950DT</term>
