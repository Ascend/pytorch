# 概述

介绍PyTorch2.9.0版本原生API在昇腾NPU上的支持情况与限制说明，具体使用方法请参考[PyTorch社区文档](https://pytorch.org/docs/2.9/)。原生API的支持情况可划分为如下四类：

- **支持**：当“是否支持”标记为“是”，且“限制与说明”标记为“-”，表示该API在昇腾NPU上的支持度与PyTorch原生API完全一致。
- **差异化支持**：当“是否支持”标记为“是”，但“限制与说明”不为“-”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请注意查阅具体的限制与说明，确保适配昇腾NPU平台。
- **暂不支持**：当“是否支持”标记为“否”，“限制与说明”标记为“-”，表示当前在昇腾NPU上暂不支持该API。
- **未验证**：部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.9/)中存在，但未收录于本支持清单。这类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。

> [!NOTE] 
>
> **产品支持范围说明：**
原生API支持度仅在<term>Ascend 950DT</term>、<term>Atlas A3 训练系列产品</term>和<term>Atlas A2 训练系列产品</term>上经过验证，默认仅在以上三种产品上支持。
> 
> - 若API“是否支持”为“是”，则代表在<term>Ascend 950DT</term>、<term>Atlas A3 训练系列产品</term>和<term>Atlas A2 训练系列产品</term>上均支持。<br>
> 
> - 若API“是否支持”为“是、暂不支持<term>Ascend 950DT</term>”，则代表在<term>Atlas A3 训练系列产品</term>、<term>Atlas A2 训练系列产品</term>上支持，<term>Ascend 950DT</term>不支持。
