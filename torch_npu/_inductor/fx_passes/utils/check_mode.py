import torch


def is_inference_check() -> bool:
    """
        Tell whether the graph is a pure inference graph
    """
    return not torch.is_grad_enabled()  # pure inference graph
