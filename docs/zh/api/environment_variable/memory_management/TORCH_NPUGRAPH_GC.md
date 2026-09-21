# TORCH\_NPUGRAPH\_GC

## 功能描述

通过此环境变量可控制图捕获模式（NPUGraph Capture）过程中是否主动触发Python GC（Garbage Collection）。

- 配置为"0"时，NPUGraph Capture不会主动触发Python GC。
- 配置为"1"时，NPUGraph Capture会主动触发Python GC。

默认值："0"。

## 配置示例

```bash
export TORCH_NPUGRAPH_GC=1
```

## 使用约束

- TORCH\_NPUGRAPH\_GC环境变量读取依赖PyTorch模块，可配置为"0"或"1"。对于其他值，在PyTorch不同版本间存在差异，并且未来可能发生变动，不建议配置。
    >       PyTorch 2.7.1之前的版本，设置非"0"或"1"的值时，会取默认值"0"。
    >       PyTorch 2.7.1及之后的版本，设置非"0"或"1"的值时，会取默认值"1"。

- TORCH\_NPUGRAPH\_GC设置为"1"时，会导致NPUGraph Capture性能下降。

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas 训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 训练系列产品</term>
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas 推理系列产品</term>
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT</term>
<!-- end id5 -->
