# 添加二进制黑名单示例

参见如下示例设置"NPU\_FUZZY\_COMPILE\_BLACKLIST"选项，添加二进制黑名单。

单个算子使用示例：

```
import torch
import torch_npu

option = {}
option['NPU_FUZZY_COMPILE_BLACKLIST'] = "DynamicGRUV2"       #根据实际场景进行替换
torch.npu.set_option(option)
```

多个算子使用示例：

```
import torch
import torch_npu

option = {}
option['NPU_FUZZY_COMPILE_BLACKLIST'] = "DynamicGRUV2,DynamicRNN"          #根据实际场景进行替换
torch.npu.set_option(option)
```

