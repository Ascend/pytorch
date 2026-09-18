# TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_FWD_MASK_WORKSPACE_BYTES

## 功能描述

设置FlexAttention前向预生成掩码的有界工作区预算，单位为字节，默认值为268435456（256 MiB）。

| 值 | 说明 |
| --- | --- |
| 正整数 | 在容量估算不超过该预算且其他条件满足时，使用有界工作区分配。 |
| 0 | 关闭有界工作区分配，按实际部分块数量计算精确容量。 |

该值是选择有界分配策略的预算，不是整个Attention调用的显存上限。容量超过预算或不满足条件时使用精确容量路径，不截断掩码，也不保证总工作区小于该值。

## 配置示例

```shell
export TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_FWD_MASK_WORKSPACE_BYTES=268435456
python flex_attention_example.py
```

其中`flex_attention_example.py`见[快速入门](./overview.md)。请传入表示字节数的十进制整数，不要使用`256M`等带单位字符串。

## 使用约束

- 仅影响前向预生成掩码工作区；不限制反向工作区、`BlockMask`构造内存或整个进程的NPU内存。
- 需要静态形状、非AOT编译、非多流调度等条件，并受int32元数据和容量范围约束；用于扫描的掩码计数行数不超过4096。
- 预算增大可能提高有界路径的覆盖范围，也可能增加预留内存。预算为0时仍会分配实际需要的工作区。
- 设置环境变量后使用新进程。

## 支持的型号

- <term>Ascend 950PR&950DT系列产品</term>
