unsupport_Tensor_api = [
  "torch.Tensor.is_shared",
  "torch.Tensor.share_memory_"
]

unsupport_nn_api = [
  "torch.nn.DataParallel.__init__",
  "torch.nn.parallel.data_parallel",
  "torch.nn.Module.share_memory",
  "torch.jit.ScriptModule.add_module",
  "torch.jit.ScriptModule.bfloat16",
  "torch.jit.ScriptModule.register_buffer",
  "torch.jit.ScriptModule.register_parameter",
  "torch.jit.ScriptModule.register_module"
]

unsupport_nested_api = [
  "torch.nested.nested_tensor",
  "torch.nested.as_nested_tensor"
]
