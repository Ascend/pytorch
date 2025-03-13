from typing import List, Tuple

import torch


lib = torch.library.Library("fsdp", "FRAGMENT")


@torch.library.impl(lib, "chunk_cat", "PrivateUse1")
def chunk_cat(
    tensors: List[torch.Tensor],
    dim: int,
    num_chunks: int,
    out: torch.Tensor,
) -> None:
    tensors = [tensor.contiguous() for tensor in tensors]
    out = out.contiguous()
    torch._chunk_cat(tensors, dim, num_chunks, out=out)


@torch.library.impl(lib, "all_gather_copy_in", "PrivateUse1")
def all_gather_copy_in_npu(
    all_gather_inputs: List[torch.Tensor],
    inp_split_sizes: List[int],
    all_gather_input_numel: int,
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_gather_output = torch.empty(
        (all_gather_input_numel * world_size,), dtype=dtype, device=device
    )
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    with torch.no_grad():
        torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs)
    return all_gather_input, all_gather_output


@torch.library.impl(lib, "split_with_sizes_copy", "PrivateUse1")
def split_with_sizes_copy(
    all_gather_output: torch.Tensor,
    all_gather_input_split_sizes: List[int],
    dim: int,
    out: List[torch.Tensor],
) -> None:
    torch.split_with_sizes_copy(
        all_gather_output, all_gather_input_split_sizes, dim=dim, out=out
    )
