def _add_deterministic_patch():
    """
    Patch torch.utils.deterministic.fill_uninitialized_memory setter so that
    setting it also synchronizes the NPU-side option
    TORCH_NPU_FILL_UNINITIALIZED_MEMORY via torch_npu._C._npu_setOption.
    """
    import torch_npu
    import torch.utils.deterministic as det_mod

    # The module instance's class has been replaced with _Deterministic
    # (see torch/utils/deterministic.py: sys.modules[__name__].__class__ = _Deterministic).
    _Deterministic = det_mod.__class__

    _orig_prop = _Deterministic.fill_uninitialized_memory
    _orig_fget = _orig_prop.fget
    _orig_fset = _orig_prop.fset

    def _new_fset(self, mode):
        # 1. call original setter: torch._C._set_deterministic_fill_uninitialized_memory(mode)
        _orig_fset(self, mode)
        # 2. sync NPU-side option
        option = {"TORCH_NPU_FILL_UNINITIALIZED_MEMORY": "1" if mode else "0"}
        torch_npu._C._npu_setOption(option)

    _Deterministic.fill_uninitialized_memory = property(_orig_fget, _new_fset)
