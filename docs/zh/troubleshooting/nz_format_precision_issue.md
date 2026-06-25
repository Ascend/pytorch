# 使用NZ格式后精度异常

## 问题现象描述

在网络调优时，用户开启[`torch.npu.config.allow_internal_format = True`](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/zh/custom_APIs/torch_npu-npu/%EF%BC%88beta%EF%BC%89torch_npu-npu-config-allow_internal_format.md)后，模型的计算结果出现精度异常。

## 原因分析

昇腾NPU内部支持两种张量数据排布格式：ND（N-Dimensional）是标准的多维排布格式，与PyTorch原生格式一致，通用性好；NZ（Z-order）是一种分形排布格式，将数据按16×16的小块重新组织，能提升部分算子的访存效率，常用于网络调优场景。

用户代码中存在NZ格式的tensor输入。在未启用`torch.npu.config.allow_internal_format`开关（默认值`False`）时，torch_npu会隐式地将NZ格式转换为ND格式后再进行计算，用户使用上无感知。

开启`torch.npu.config.allow_internal_format`开关后，torch_npu不再进行隐式格式转换，NZ格式的tensor直接参与算子计算。由于部分算子在NZ格式下的计算路径与ND格式存在数值差异，导致最终结果精度异常。

## 解决措施

根据实际场景选择以下方式之一：

1. **推荐**：将NZ格式的输入数据转换为ND格式后再传入模型，例如：

   ```python
   input_tensor = input_tensor.float().npu_format_cast(2)  # 2 即 ND 格式
   ```

2. 继续关闭`torch.npu.config.allow_internal_format`开关（保持默认值`False`）。
