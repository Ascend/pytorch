import functools

import torch


_original_fork_rng = torch.random.fork_rng


@functools.wraps(_original_fork_rng)
def _npu_fork_rng(
    devices=None,
    enabled=True,
    _caller="fork_rng",
    _devices_kw="devices",
    device_type=None,
):
    if device_type is None:
        device_type = "npu"

    return _original_fork_rng(
        devices=devices,
        enabled=enabled,
        _caller=_caller,
        _devices_kw=_devices_kw,
        device_type=device_type,
    )


def _add_fork_rng_patch():
    torch_version = tuple(
        int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2]
    )
    if torch_version >= (2, 13):
        return

    torch.random.fork_rng = _npu_fork_rng
