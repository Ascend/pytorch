from enum import IntEnum

import torch
import torch_npu


class Format(IntEnum):
    """NPU storage format enumeration class"""
    UNDEFINED = -1
    NCHW = 0
    NHWC = 1
    ND = 2
    NC1HWC0 = 3
    FRACTAL_Z = 4
    NC1HWC0_C04 = 12
    HWCN = 16
    NDHWC = 27
    FRACTAL_NZ = 29
    NCDHW = 30
    NDC1HWC0 = 32
    FRACTAL_Z_3D = 33
    NC = 35
    NCL = 47

    def __str__(self):
        return self.name


def _apply_npu_format_patch():
    orig_get_format = torch_npu.get_npu_format
    
    def patched_get_format(tensor):
        """get the Format type of tensor"""
        format_int = orig_get_format(tensor)
        return Format(format_int)
    
    torch_npu.get_npu_format = patched_get_format
    torch_npu.Format = Format
