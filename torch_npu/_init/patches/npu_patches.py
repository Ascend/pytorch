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

    def _npu_internal_format_repr(self, *, tensor_contents=None):
        if self.device.type == "npu":
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
