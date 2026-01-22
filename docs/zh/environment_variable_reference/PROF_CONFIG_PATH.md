# PROF\_CONFIG\_PATH

## 功能描述

PyTorch训练场景Ascend PyTorch Profiler接口的dynamic\_profile采集功能profiler\_config.json配置文件路径环境变量。

## 配置示例

```
export PROF_CONFIG_PATH="/path/to/profiler_config_path"
```

-   配置该环境变量后启动训练，dynamic\_profile会在profiler\_config\_path下自动创建模板文件profiler\_config.json，用户可基于模板文件自定义修改配置项。
-   PROF\_CONFIG\_PATH指定的路径可自定义（要求有读写权限），路径格式仅支持由字母、数字和下划线组成的字符串，不支持软链接，例如"/home/xxx/profiler\_config\_path"。
-   dynamic\_profile采集功能及profiler\_config.json文件的详细介绍请参见《CANN 性能调优工具用户指南》中的“[dynamic\_profile动态采集](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/Profiling/atlasprofiling_16_0033.html#ZH-CN_TOPIC_0000002534478481__section17272160135118)”章节。

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

