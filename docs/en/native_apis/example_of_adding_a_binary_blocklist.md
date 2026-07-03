# Example of Adding a Binary Blocklist

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T02:24:24.698Z pushedAt=2026-06-15T03:25:49.242Z -->

Refer to the following examples to set the `NPU\_FUZZY\_COMPILE\_BLACKLIST` option and add a binary blocklist.

Example for a single operator:

```python
import torch
import torch_npu

option = {}
option['NPU_FUZZY_COMPILE_BLACKLIST'] = "DynamicGRUV2"       #Replace it according to the actual scenario
torch.npu.set_option(option)
```

Example for multiple operators:

```python
import torch
import torch_npu

option = {}
option['NPU_FUZZY_COMPILE_BLACKLIST'] = "DynamicGRUV2,DynamicRNN"          #Replace it according to the actual scenario
torch.npu.set_option(option)
```
