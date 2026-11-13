from .channel_shuffle import ChannelShuffle
from .prefetcher import Prefetcher
from .crossentropy import LabelSmoothingCrossEntropy
from .roi_align import ROIAlign
from .deform_conv import ModulatedDeformConv
from .activations import Mish, SiLU, Swish
from .ps_roi_pooling import PSROIPool
from .ensemble_dropout import NpuFairseqDropout, NpuCachedDropout
from ._ensemble_dropout import NpuPreGenDropout
from .multihead_attention import MultiheadAttention
from .drop_path import NpuDropPath
from .focus import Focus
from ._batchnorm_with_int32_count import FastBatchNorm1d, \
    FastBatchNorm2d, FastBatchNorm3d, FastSyncBatchNorm
from .linear_a8w8_quant import LinearA8W8Quant
from .linear_quant import LinearQuant
from .linear_weight_quant import LinearWeightQuant
from .npu_modules import DropoutWithByteMask

__all__ = [
    "ChannelShuffle",
    "Prefetcher",
    "LabelSmoothingCrossEntropy",
    "ROIAlign",
    "ModulatedDeformConv",
    "PSROIPool",
    "NpuFairseqDropout",
    "NpuCachedDropout",
    "MultiheadAttention",
    "NpuDropPath",
    "Focus",
    "LinearA8W8Quant",
    "LinearQuant",
    "LinearWeightQuant",
    "DropoutWithByteMask",
]

from torch_npu.contrib._migrated import MIGRATED_TO_TACO, warn_migrated


def __getattr__(name):
    if name in MIGRATED_TO_TACO:
        import importlib
        warn_migrated(name)
        impl = importlib.import_module(f"{__name__}.{MIGRATED_TO_TACO[name]}")
        return getattr(impl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
