# ENABLE_INPLACE_BUFFERS

## 功能描述

社区Inductor默认生成的triton kernel中输入/输出参数会复用地址指针(例in_out_ptr0)。但这种方式阻碍了编译器后端(Ascend NPU-IR)的multi-buffer特性，即在(load-compute-store)之间实施pipeline，实现流水掩盖。

该环境变量设置为`export ENABLE_INPLACE_BUFFERS=0`时，Inductor-Ascend在生成Triton Kernel中的参数不会复用地址空间(例in_ptr0, out_ptr0)。

使用者可以通过设置该开关，对比前后性能，选择最优配置。

| 值 | 说明 |
|---|---|
| 未设置 或 设置为1、true、yes | 输入/输出参数复用地址空间（默认值） |
| 0、false、no等 |  输入/输出参数不复用地址空间 |

## 配置示例

```bash
export ENABLE_INPLACE_BUFFERS=0
```

## 使用约束

- 无

## 支持的型号

-   <term>Atlas A5 系列产品</term>
