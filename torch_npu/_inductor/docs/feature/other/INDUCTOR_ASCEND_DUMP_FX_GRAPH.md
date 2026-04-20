# INDUCTOR_ASCEND_DUMP_FX_GRAPH

## 功能描述

dump 功能用于单算子用例的调试和问题排查。当INDUCTOR_ASCEND_CHECK_ACCURACY或AOTI_ASCEND_DEBUG_KERNEL启用时，会自动启用此功能，默认值为空。

| 值 | 说明 |
|---|---|
| 未设置或空 | 关闭fx graph dump（默认值） |
| 1、true、yes等 | 开启fx graph dump |

## 配置示例

```bash
export INDUCTOR_ASCEND_DUMP_FX_GRAPH=1
```

## 使用约束

- 开启dump功能会生成额外的文件，占用磁盘空间，建议仅在调试和问题排查时使用。
- 当INDUCTOR_ASCEND_CHECK_ACCURACY或AOTI_ASCEND_DEBUG_KERNEL启用时，会自动启用此功能。

## 支持的型号

-   <term>Atlas A5 系列产品</term>
