# Example of Adding a Binary Blocklist

Refer to the following examples to set the "NPU\\_FUZZY\\_COMPILE\\_BLACKLIST" option and add a binary blocklist.

Example for a single operator:

```python
import torch
import torch_npu

option = {}
option['NPU_FUZZY_COMPILE_BLACKLIST'] = "DynamicGRUV2"       # Replace based on the actual scenario
torch.npu.set_option(option)
```

Example for multiple operators:

```python
import torch
import torch_npu

option = {}
option['NPU_FUZZY_COMPILE_BLACKLIST'] = "DynamicGRUV2,DynamicRNN"          # Replace based on the actual scenario
torch.npu.set_option(option)
```
