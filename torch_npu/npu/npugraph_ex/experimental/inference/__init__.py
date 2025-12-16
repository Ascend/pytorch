import torch


def use_internal_format_weight(model: torch.nn.Module):
    from torch_npu.dynamo.torchair.experimental import inference
    return inference.use_internal_format_weight(model)