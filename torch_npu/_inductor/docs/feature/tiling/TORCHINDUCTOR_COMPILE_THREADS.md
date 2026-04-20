# TORCHINDUCTOR_COMPILE_THREADS(同社区)

## 功能
并发编译的进程数量，默认值为32，与社区逻辑保持一致，控制并发编译kernel的进程数量，大于1时，使用多进程编译。

## 配置示例
```shell
export TORCHINDUCTOR_COMPILE_THREADS=32
```

## 使用约束
无

## 支持型号
-   <term>Atlas A2 系列产品</term>
-   <term>Atlas A3 系列产品</term>
-   <term>Atlas A5 系列产品</term>