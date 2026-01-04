from typing import List
import os
import torch


IS_INFERENCE = False


def is_inference_check() -> bool:
    """
        判断图是否为“纯推理图”
    """
    is_optimizer_test = os.environ.get("IS_OPTIMIZER_TEST")
    if is_optimizer_test:
        return True
    return not torch.is_grad_enabled()  # 纯推理图