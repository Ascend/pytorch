# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""Physically-isolated triton_experimental Inductor backend for NPU.

Importing this package is deliberately cheap: it only re-exports
``get_current_raw_stream`` (generated kernels import it by this path) and defines
``_activate``. All backend registration / monkeypatching runs lazily in
``_activate`` — invoked by ``torch_npu._inductor._load_triton_experimental_backend``
when the backend is selected — so that importing submodules (e.g. from a generated
kernel at runtime) never re-triggers registration.
"""
import logging
import os

from torch._inductor import config as inductor_config

from . import config as ncfg
# Re-exported here because generated kernels import it as
# ``from torch_npu._inductor.triton_experimental import get_current_raw_stream``.
from .device import get_current_raw_stream  # noqa: F401


def _route_worker_logs_for_debug():
    """Under debug, raise the inductor log level and route worker stdout to the terminal.

    The backend logs through ``logging.getLogger("torch._inductor")``; that logger
    stays at the root default (WARNING) unless raised, so DEBUG/INFO messages are
    dropped. Under ``config.debug`` we set it to DEBUG here (equivalent to
    ``TORCH_LOGS="+inductor"`` for this logger, but without requiring a shell var).

    Inductor also compiles each kernel in a subprocess worker whose stdout/stderr
    goes to a log file or /dev/null, so NPU debug prints only reach the terminal for
    the one kernel compiled in-process. Point worker logging at /dev/stdout instead;
    multi-process compilation is preserved and each "[AUTOTUNE] <kernel>:" line is
    self-identifying (lines from 32 workers may interleave).
    """
    if not ncfg.debug:
        return
    logging.getLogger("torch._inductor").setLevel(logging.DEBUG)
    if os.getenv("TORCHINDUCTOR_WORKER_SUPPRESS_LOGGING") is None:
        inductor_config.worker_suppress_logging = False
    if not os.getenv("TORCHINDUCTOR_WORKER_LOGPATH"):
        inductor_config.torchinductor_worker_logpath = ncfg.worker_log_path


def _activate():
    """Activate the triton_experimental Inductor backend.

    Invoked by ``torch_npu._inductor._load_triton_experimental_backend`` when this
    backend is selected (env ``TORCHINDUCTOR_NPU_BACKEND=triton_experimental`` or
    ``torch.compile(options={"npu_backend": "triton_experimental"})``). The work is
    deliberately NOT run at package import: generated kernels import submodules of
    this package (e.g. ``npu_triton_heuristics``) at runtime, and importing the
    package must not re-trigger backend registration / monkeypatching.

    Statement order mirrors the original standalone package's import-time sequence.
    """
    from . import device
    from .npu_triton_heuristics import enable_mspti_autotune
    # overrides' module body has an import-time side effect (size_asserts off);
    # importing it here fires it at activation. The aten Library + dispatcher impls
    # and the rest of the decomposition overrides now live in
    # torch_npu._inductor.decomposition and are applied by the loader
    # (_load_triton_experimental_backend) before this _activate() runs.
    from .overrides import apply_npu_overrides

    device.register_backend_for_npu()

    inductor_config.layout_optimization = False
    inductor_config.coordinate_descent_tuning = False
    # Split reductions emit multi-kernel splits whose intermediate buffers have hit
    # OOB / shape-mismatch on the NPU Triton backend. Off by default.
    inductor_config.split_reductions = False

    _route_worker_logs_for_debug()

    # Giant fused kernels make the bishengir/MLIR compiler explode: T5 backward
    # horizontally fuses many layers' softmax-backward (shared operands, one output
    # buffer) into ONE pointwise kernel with 70+ input pointers / ~290 loads, ~12 min
    # device compile. The upstream fusion caps measure node count or a single node's
    # reads, not the fused kernel's load count, so none fire here. NPUTritonScheduling.
    # can_fuse gates on combined load count via config.max_fused_reads (default 24):
    # the monster splits into several <=15-load kernels, ~45s, bit-exact.
    apply_npu_overrides()

    device.register_device_op_overrides_for_npu()

    from .lowering import _register_npu_inductor_fallbacks
    _register_npu_inductor_fallbacks()

    device.register_interface_for_npu()

    enable_mspti_autotune()
