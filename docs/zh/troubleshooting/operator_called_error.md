# 算子调用报错

## 问题现象描述

关键词"**Cannot find bin of op ...**"

```ColdFusion
Traceback (most recent call last):
  File "/home/HwHiAiUser/workspace/qwen2.5-Math-deepseek-R1.py", line 38, in <module>
    generated_ids = model.generate(
  File "/home/HwHiAiUser/.local/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/home/HwHiAiUser/.local/lib/python3.10/site-packages/transformers/generation/utils.py", line 1575, in generate
    result = self._sample(
  File "/home/HwHiAiUser/.local/lib/python3.10/site-packages/transformers/generation/utils.py", line 2690, in _sample
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
RuntimeError: call aclnnArange failed, detail:EZ9999: Inner Error!
EZ9999: [PID: 775] 2025-02-16-12:18:25.258.321 Cannot find bin of op Range, integral key 0/1/|int64/ND/int64/ND/int64/ND/int64/ND/.
        TraceBack (most recent call last):
       Cannot find binary for op Range.
       Kernel GetWorkspace failed. opType: 9
       ArangeAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.
[ERROR] 2025-02-16-12:18:25 (PID:775, Device:0, RankID:-1) ERR01100 OPS call acl api failed
```

## 原因分析

算子调用报错，并打印错误码“ERR01100”。可能原因：
- 未安装配套Kernels或ops包
- 未找到匹配的算子二进制文件

## 解决措施

1. 检查安装配套的Kernels或ops包。
2. 查看算子输入数据类型是否支持，如果不支持，请使用支持的数据类型。具体可参考《CANN 算子库接口参考》中“[Ascend IR算子规格说明](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/aolapi/operatorlist_00094.html)”章节进行查看。

