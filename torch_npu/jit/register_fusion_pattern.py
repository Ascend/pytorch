import torch
from .fusion_pass.fast_gelu import fast_gelu_pass

__all__ = ["optimize"]


def optimize(jit_mod):
    if isinstance(jit_mod, torch.jit.ScriptModule):
        torch.jit.optimize_for_inference(jit_mod)
    
    fast_gelu_pass(jit_mod)
    