# KINETO\_USE\_DAEMON

## 功能描述

PyTorch训练场景用于设置是否通过msMonitor nputrace方式开启dynamic\_profile采集功能。

## 配置示例

```
export KINETO_USE_DAEMON=1
```

详细使用方式请参见《CANN 性能调优工具用户指南》中的“[dynamic\_profile动态采集](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/Profiling/atlasprofiling_16_0033.html#ZH-CN_TOPIC_0000002534478481__section17272160135118)”章节。

## 使用约束

-   脚本不手动添加代码的情况，此环境变量适用于PyTorch训练场景。
-   脚本内添加dynamic\_profile模块后，此环境变量可以在非训练场景使用。例如：

    ```python
    # 加载dynamic_profile模块
    from torch_npu.profiler import dynamic_profile as dp
    # 设置Profiling配置文件的路径
    dp.init("profiler_config_path")
    …
    for step in steps:
    	train_one_step()
    	# 划分step
    	dp.step()
    ```

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>

