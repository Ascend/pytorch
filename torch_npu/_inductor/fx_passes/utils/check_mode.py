from typing import List
import os
import torch


def is_inference_check() -> bool:
    """
        判断图是否为“纯推理图”
    """
    return not torch.is_grad_enabled()  # 纯推理图