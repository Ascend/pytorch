from torch_npu._compat.version import CURRENT_VERSION

# COMPAT(>= 2.11): register_op_strategy / register_prop_rule moved from
#   _ops.registration to _ops.utils in PyTorch 2.11.
# CAN REMOVE else branch when MIN_SUPPORTED >= (2, 11)
if CURRENT_VERSION >= (2, 11):
    from torch.distributed.tensor._ops.utils import register_op_strategy, register_prop_rule
else:
    from torch.distributed.tensor._ops.registration import register_op_strategy, register_prop_rule
