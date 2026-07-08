# inf/nan值处理差异说明

在TorchNPU中，部分API对`inf`/`nan`特殊值的处理行为与PyTorch原生实现存在差异，可能导致计算结果不一致。本文档主要列出两类差异场景：异常输入（输入包含`inf`/`nan`）时的行为差异，以及数值溢出时输出`inf`/`nan`的差异。

**异常输入处理有差异的API清单**

当输入Tensor中包含`inf`/`nan`值时，TorchNPU与PyTorch原生的输出结果存在差异，具体API列表请参见[表1](#输入包含inf-nan时处理与原生不一致的API列表)。

**表1**  输入包含inf/nan时处理与原生不一致的API列表<a id="输入包含inf-nan时处理与原生不一致的API列表"></a>

| API名称 |
| --- |
| Tensor.sign |
| torch.sign |
| torch.linalg.svd |
| torch.linalg.qr |

**数值溢出时输出inf/nan存在差异的场景**

当输入本身不包含`inf`/`nan`，但计算过程中发生数值溢出时，可能出现以下差异场景：

1. 原生输出为`nan`，NPU输出为±`inf`；
2. 原生输出为±`inf`，NPU输出为`nan`；
3. 原生和NPU均输出`inf`，但符号相反。

以上差异由底层硬件计算机制不同导致，属于硬件行为差异，不属于精度问题。
