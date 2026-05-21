import torch.nn.functional as F
import torch.distributed.tensor.experimental._context_parallel._attention as _native

from ._npu_attention import (
    npu_disable_cp_dtensor_dispatcher,
    npu_enable_cp_dtensor_dispatcher,
)


# Replace dispatcher registration functions in the native module
_native._enable_cp_dtensor_dispatcher = npu_enable_cp_dtensor_dispatcher
_native._disable_cp_dtensor_dispatcher = npu_disable_cp_dtensor_dispatcher
