import torch
import torch_npu


def _rename_npu_profiler_activity() -> None:
    """Alias upstream ProfilerActivity.PrivateUse1 as ProfilerActivity.NPU.

    torch_npu's own enum already has NPU; this mirrors it onto torch's upstream
    enum. torch 2.12 lacks upstream's _rename_profiler_activity, so we do it
    here. Idempotent; prefers upstream's helper when present.
    """
    from torch._C._profiler import ProfilerActivity

    alias = "NPU"
    if hasattr(ProfilerActivity, alias):
        return

    import torch.utils.backend_registration as _br
    if hasattr(_br, "_rename_profiler_activity"):
        _br._rename_profiler_activity("npu")
        return

    setattr(ProfilerActivity, alias, ProfilerActivity.PrivateUse1)

    pu1 = ProfilerActivity.PrivateUse1
    original_repr = ProfilerActivity.__repr__
    original_str = ProfilerActivity.__str__

    def custom_repr(self):
        if self == pu1:
            return f"<ProfilerActivity.{alias}: {pu1.value}>"
        return original_repr(self)

    def custom_str(self):
        if self == pu1:
            return f"ProfilerActivity.{alias}"
        return original_str(self)

    ProfilerActivity.__repr__ = custom_repr
    ProfilerActivity.__str__ = custom_str
    ProfilerActivity.name = property(
        lambda self: alias if self == pu1 else original_str(self).split(".")[-1]
    )


def register_privateuse1_backend():
    torch.utils.rename_privateuse1_backend("npu")
    _rename_npu_profiler_activity()
    torch._register_device_module("npu", torch_npu.npu)
    unsupported_dtype = [
        torch.quint8,
        torch.quint4x2,
        torch.quint2x4,
        torch.qint32,
        torch.qint8,
    ]
    torch_npu.unsupported_dtype = unsupported_dtype
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True,
        for_module=True,
        for_storage=True,
        unsupported_dtype=unsupported_dtype,
    )
    torch.nn.parameter.UninitializedTensorMixin._allowed_methods.append(
        torch.Tensor.npu
    )
