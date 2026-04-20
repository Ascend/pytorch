# INDUCTOR_ASCEND_CHECK_ACCURACY

## 功能描述

开启triton后端精度对比工具，dump单算子用例。当启用时，会自动启用INDUCTOR_ASCEND_DUMP_FX_GRAPH功能，默认值为空。

| 值 | 说明 |
|---|---|
| 未设置或空 | 关闭精度对比工具（默认值） |
| 1、true、yes等 | 开启精度对比工具 |

## 配置示例

```bash
export INDUCTOR_ASCEND_CHECK_ACCURACY=1
```

## 使用约束

- 开启精度对比工具会影响编译效率，建议仅在调试和精度验证时使用。
- 当此变量启用时，会自动启用INDUCTOR_ASCEND_DUMP_FX_GRAPH功能。

## 支持的型号

-   <term>Atlas A5 系列产品</term>
