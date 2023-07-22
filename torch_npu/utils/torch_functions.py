import torch

# only to patch the operator `torch.linalg.vector_norm` used in torch/nn/utils/clip_grad.py
def _linalg_vector_norm(self, ord=2, dim=None, keepdim=False, *, dtype=None):
    if (dim is None and keepdim is False and dtype is None):
        if ord == torch.inf:
            return self.abs().max()
        else:
            return torch.norm(self, ord)
    else:
        return torch._C._linalg.linalg_vector_norm(self, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)

def add_torch_functions():
    torch.linalg.vector_norm = _linalg_vector_norm
