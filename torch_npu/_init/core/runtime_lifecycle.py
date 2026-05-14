import atexit

import torch_npu


def _npu_shutdown():
    """
    NPU exit, need to synchronize devices
    """
    from torch_npu.asd.asd import matmul_check
    from torch_npu.utils._error_code import _except_handler

    success = torch_npu._C._npu_shutdown_synchronize()
    torch_npu.distributed.distributed_c10d._destructor_process_group()
    torch_npu._C._npu_shutdown(success)
    _except_handler.handle_exception()

    matmul_check._cleanup()

    if torch_npu.npu.aclnn._use_static_aclnn_kernel:
        from torch_npu._inductor.npu_static_kernel import uninstall_static_kernel

        uninstall_static_kernel()


def _initialize_runtime_lifecycle():
    """
    Complete C extension initialization and register process-exit cleanup.

    This entry is expected to be called once by torch_npu top-level import.
    """
    if not hasattr(torch_npu, "_C"):
        raise RuntimeError("torch_npu._C is not available before extension init")

    if not hasattr(torch_npu._C, "_initExtension"):
        raise RuntimeError("torch_npu._C._initExtension is not available")

    # final extension barrier, this must be placed at the end
    torch_npu._C._initExtension()
    atexit.register(_npu_shutdown)
