# Example of Adding a Binary Blocklist

Set the "NPU\_FUZZY\_COMPILE\_BLACKLIST" option as shown in the following example to add a binary blocklist.

Example of using a single operator:

```python
import torch
import torch_npu

option = {}
option['NPU_FUZZY_COMPILE_BLACKLIST'] = "DynamicGRUV2"       #Replace it based on the actual scenario
torch.npu.set_option(option)
```

Example of using multiple operators:

```python
import torch
import torch_npu

option = {}
option['NPU_FUZZY_COMPILE_BLACKLIST'] = "DynamicGRUV2,DynamicRNN"          #Replace it based on the actual scenario
torch.npu.set_option(option)
```
