# 概述

介绍PyTorch2.8.0版本原生API接口在昇腾NPU上的支持情况与限制说明，PyTorch2.8.0版本原生API接口具体使用方法请参考[PyTorch社区文档](https://pytorch.org/docs/2.8/)。原生API接口在昇腾NPU上的支持情况与限制说明可分为如下四类：

-   API“是否支持“为“是“，“限制与说明“为“-“，说明此API和原生API支持度保持一致。
-   API“是否支持“为“是“，“限制与说明“不为“-“，说明此API和原生API支持度不一致，请注意昇腾NPU上的支持度。
-   API“是否支持“为“否“，“限制与说明“为“-“，说明在昇腾NPU上暂不支持此API。
-   部分API在[PyTorch社区文档](https://pytorch.org/docs/2.8/)中存在，但此文档中未承载，为昇腾NPU未验证API，在昇腾NPU上请谨慎使用，后续验证后会持续更新文档。

