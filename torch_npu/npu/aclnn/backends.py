import warnings


def version():
    """Currently, the ACLNN version is not available and does not support it. 
    By default, it returns None.
    """
    warnings.warn("torch.npu.aclnn.version isn't implemented!")
    return None
