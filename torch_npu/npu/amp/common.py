import torch_npu


def amp_definitely_not_available():
    return not torch_npu.npu.is_available()
