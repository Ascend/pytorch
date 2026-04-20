## CATLASS_EPILOGUE_FUSION （同社区CUTLASS_EPILOGUE_FUSION）

## 功能描述

是否开启catlass cv融合，与社区保持一致，社区环境变量为CUTLASS_EPILOGUE_FUSION。

"0"为关闭，"1"为开启

默认配置为CATLASS_EPILOGUE_FUSION="0"

## 配置示例

开启catlass cv融合功能：

```shell
export CATLASS_EPILOGUE_FUSION=1
```

关闭catlass cv融合功能：

```shell
export CATLASS_EPILOGUE_FUSION=0
```

## 使用约束

无

## 支持的型号

-   <term>Atlas A5 系列产品</term>