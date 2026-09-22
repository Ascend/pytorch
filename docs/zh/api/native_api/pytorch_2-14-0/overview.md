# 概述

介绍PyTorch2.14.0版本原生API在昇腾NPU上的支持情况与限制说明，PyTorch2.14.0版本原生API具体使用方法请参考[PyTorch社区文档](https://pytorch.org/docs/2.14/)。

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.14/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。

## 版本继承与兼容性说明

本清单继承PyTorch 2.13.0版本的产品支持情况和限制说明，并依据2.14版本变更修正接口与链接。未收录的新接口不表示已完成昇腾NPU验证。

- 2.14已移除`torch.qr`、`torch.Tensor.qr`、`torch.cholesky`和`torch.Tensor.cholesky`的实现，调用会报错。本清单移除原有的`torch.qr`条目；请使用`torch.linalg.qr`或`torch.linalg.cholesky`。`qr`的`some`参数需改为`mode`；Cholesky上三角结果可通过`torch.linalg.cholesky(A).mH`获取。
- `torch.profiler.profile`和`torch.autograd.profiler.profile`已移除`use_cuda`参数，分别使用`activities`和`use_device`选择设备；`with_modules`参数已废弃，动态图模块跟踪请使用`with_stack=True`。

版本变更依据：[PyTorch 2.14发布说明](https://github.com/pytorch/pytorch/releases/tag/v2.14.0)。
