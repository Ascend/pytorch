from torch_npu._init.patches.patch_manager import PatchManager


@PatchManager.register_patch("npu")
def apply_npu_format_patch():
    from torch_npu.npu._format import _apply_npu_format_patch

    _apply_npu_format_patch()


@PatchManager.register_patch("npu")
def apply_flex_attention_patch():
    from torch_npu.utils.patch_flexattention import (
        _patch_flex_attention_device,
        _register_npu_flex_attention_autocast,
    )

    _patch_flex_attention_device()
    _register_npu_flex_attention_autocast()


@PatchManager.register_patch("npu")
def apply_npu_internal_format_repr_patch():
    # Internal-format tensors (e.g. FRACTAL_NZ) hit the internal-format guard in
    # _tensor_str cat/stack. Force .cpu() to trigger d2h + format cast first.
    import torch
    import torch_npu

    _orig_repr = torch.Tensor.__repr__

    _symbolic_tensor_cls = None

    def _is_symbolic_tensor(t):
        # FakeTensor claims device npu (fake_device) but has no NPUStorageImpl;
        # the get_npu_format probe is pointless for it and only burns a
        # raise/catch cycle per repr now that the C++ kernel raises a
        # catchable error (!46582).
        nonlocal _symbolic_tensor_cls
        if _symbolic_tensor_cls is None:
            try:
                from torch._subclasses.fake_tensor import FakeTensor
                _symbolic_tensor_cls = FakeTensor
            except ImportError:
                _symbolic_tensor_cls = False
        return bool(_symbolic_tensor_cls) and isinstance(t, _symbolic_tensor_cls)

    def _npu_internal_format_repr(self, *, tensor_contents=None):
        if self.device.type == "npu" and not _is_symbolic_tensor(self):
            try:
                is_internal_format = (
                    torch_npu.get_npu_format(self) != int(torch_npu.Format.ND)
                )
            except Exception:
                is_internal_format = False
            if is_internal_format:
                with torch.no_grad():
                    return _orig_repr(self.cpu(), tensor_contents=tensor_contents)
        return _orig_repr(self, tensor_contents=tensor_contents)

    torch.Tensor.__repr__ = _npu_internal_format_repr
