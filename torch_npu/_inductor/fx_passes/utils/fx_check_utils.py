import torch
import torch.fx as fx


def has_storage(node: fx.Node) -> bool:
    """We can evaluate only nodes that represent tensors with defined storage."""
    if "val" not in node.meta or not isinstance(node.meta["val"], torch.Tensor):
        return False

    try:
        node.meta["val"].untyped_storage()
    except NotImplementedError:
        return False

    return True