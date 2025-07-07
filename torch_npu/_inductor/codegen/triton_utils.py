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


def get_aligned_numel(dtype):
    if dtype in byte_per_numel:
        return 32 // byte_per_numel[dtype]
    else:
        return 1
