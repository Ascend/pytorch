from torch_npu._init.patches.patch_manager import PatchManager


@PatchManager.register_patch("profiler")
def apply_mstx_patch():
    from torch_npu.profiler._add_mstx_patch import _apply_mstx_patch

    _apply_mstx_patch()


@PatchManager.register_patch("profiler")
def apply_perf_dump_patch():
    from torch_npu.utils._step import add_perf_dump_patch

    add_perf_dump_patch()
