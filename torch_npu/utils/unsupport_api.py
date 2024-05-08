import torch

"""
key: attr_name(str)
value: parent_module(object)
"""

unsupported_Tensor_api = {
    "is_shared": torch.Tensor,
    "share_memory_": torch.Tensor
}

unsupported_nn_api = {
    "share_memory": torch.nn.Module,
    "add_module": torch.jit.ScriptModule,
    "bfloat16": torch.jit.ScriptModule,
    "register_buffer": torch.jit.ScriptModule,
    "register_parameter": torch.jit.ScriptModule,
    "register_module": torch.jit.ScriptModule
}

unsupported_nested_api = {
    "nested_tensor": torch.nested,
    "as_nested_tensor": torch.nested
}
