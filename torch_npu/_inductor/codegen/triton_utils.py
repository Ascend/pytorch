from enum import Enum

import re

import torch

# wrapper npu 32 bytes align, get and pass unalign info to triton meta
# then autotune choose tiling param and send them to bishengIR
byte_per_numel = {
    torch.float32: 4,  # torch.float32 or torch.float
    torch.float64: 8,  # torch.float64 or torch.double
    torch.float16: 2,  # torch.float16 or torch.half
    torch.bfloat16: 2,  # torch.bfloat16
    torch.int32: 4,  # torch.int32 or torch.int
    torch.int64: 8,  # torch.int64 or torch.long
    torch.int16: 2,  # torch.int16 or torch.short
    torch.int8: 1,  # torch.int8
    torch.uint8: 1,  # torch.uint8
    torch.bool: 1,  # torch.bool
    torch.complex32: 4,  # torch.complex32 (not yet available in PyTorch as of the latest stable release)
    torch.complex64: 8,  # torch.complex64
    torch.complex128: 16  # torch.complex128
}


class NPUKernelType(Enum):
    SIMD = "simd"
    SIMT_ONLY = "simt_only"
    SIMT_TEMPLATE = "simt_template"
    SIMD_SIMT_MIX = "simd_simt_mix"

    def __str__(self):
        return self.value

    def compile_mode(self):
        if self == NPUKernelType.SIMT_TEMPLATE:
            return "unstructured_in_simt"
        return str(self)


def get_byte_per_numel(dtype):
    if dtype is None:
        return 1
    return byte_per_numel[dtype]


def get_aligned_numel(dtype):
    if dtype in byte_per_numel:
        return 32 // byte_per_numel[dtype]
    else:
        return 1


def get_indirect_var(node_name):
    match = re.compile(r"indirect").search(node_name)
    if match is None:
        return None 
    return node_name[match.start():]


def get_indirect_mem_var(node_name):
    indirect_mem_pattern = r'index_select|gather_template|indexput_template|scatter_template'
    match = re.compile(indirect_mem_pattern).search(node_name)
    if match is None:
        return None 
    return node_name[match.start():]