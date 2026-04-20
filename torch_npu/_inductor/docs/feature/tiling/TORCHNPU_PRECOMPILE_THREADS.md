# TORCHNPU_PRECOMPILE_THREADS
## 功能
控制多线程编译线程数量，默认为最大核数的一半（max_precompiled_thread_num = os.cpu_count() // 2），大于1时，使用并发编译。

## 配置示例
```shell
export TORCHNPU_PRECOMPILE_THREADS=32
```

## 使用约束
无

## 支持型号
-   <term>Atlas A2 系列产品</term>
-   <term>Atlas A3 系列产品</term>
-   <term>Atlas A5 系列产品</term>