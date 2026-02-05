from typing import Iterable, Union
import torch

import torch_npu
from . import _lazy_init, _lazy_call, device_count, current_device

__all__ = ['get_rng_state', 'set_rng_state',
           'get_rng_state_all', 'set_rng_state_all',
           'manual_seed', 'manual_seed_all',
           'seed', 'seed_all', 'initial_seed']


def get_rng_state(device: Union[int, str, torch.device] = 'npu') -> torch.Tensor:
    r"""Returns the random number generator state of the specified NPU as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'npu'`` (i.e., ``torch.device('npu')``, the current NPU device).

    .. warning::
        This function eagerly initializes NPU.
    """
    _lazy_init()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device('npu', device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = torch_npu.npu.default_generators[idx]
    return default_generator.get_state()


def get_rng_state_all():
    r"""Returns a list of ByteTensor representing the random number states of all devices."""

    results = []
    for i in range(device_count()):
        results.append(get_rng_state(i))
    return results


def set_rng_state(new_state: torch.Tensor, device: Union[int, str, torch.device] = 'npu') -> None:
    r"""Sets the random number generator state of the specified NPU.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'npu'`` (i.e., ``torch.device('npu')``, the current NPU device).
    """
    with torch._C._DisableFuncTorch():
        new_state_copy = new_state.clone(memory_format=torch.contiguous_format)
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device('npu', device)

    def cb():
        idx = device.index
        if idx is None:
            idx = current_device()
        default_generator = torch_npu.npu.default_generators[idx]
        default_generator.set_state(new_state_copy)

    _lazy_call(cb)


def set_rng_state_all(new_states):
    r"""Sets the random number generator state of all devices.

    Args:
        new_states (Iterable of torch.ByteTensor): The desired state for each device
    """
    for i, state in enumerate(new_states):
        set_rng_state(state, i)


def manual_seed(seed):
    r"""Sets the seed for generating random numbers for the current NPU.
    It's safe to call this function if NPU is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-NPU model, this function is insufficient
        to get determinism.  To seed all NPUs, use :func:`manual_seed_all`.
    """
    seed = int(seed)

    def cb():
        idx = current_device()
        default_generator = torch_npu.npu.default_generators[idx]
        default_generator.manual_seed(seed)

    _lazy_call(cb)


def manual_seed_all(seed):
    r"""Sets the seed for generating random numbers on all NPUs.
    It's safe to call this function if NPU is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)

    def cb():
        for i in range(device_count()):
            default_generator = torch_npu.npu.default_generators[i]
            default_generator.manual_seed(seed)

    _lazy_call(cb)


def seed():
    r"""Sets the seed for generating random numbers to a random number for the current NPU.
    It's safe to call this function if NPU is not available; in that
    case, it is silently ignored.

    .. warning::
        If you are working with a multi-NPU model, this function will only initialize
        the seed on one NPU.  To initialize all NPUs, use :func:`seed_all`.
    """

    def cb():
        idx = current_device()
        default_generator = torch_npu.npu.default_generators[idx]
        default_generator.seed()

    _lazy_call(cb)


def seed_all():
    r"""Sets the seed for generating random numbers to a random number on all NPUs.
    It's safe to call this function if NPU is not available; in that
    case, it is silently ignored.
    """

    def cb():
        random_seed = 0
        seeded = False
        for i in range(device_count()):
            default_generator = torch_npu.npu.default_generators[i]
            if not seeded:
                default_generator.seed()
                random_seed = default_generator.initial_seed()
                seeded = True
            else:
                default_generator.manual_seed(random_seed)

    _lazy_call(cb)


def initial_seed():
    r"""Returns the current random seed of the current NPU.

    .. warning::
        This function eagerly initializes NPU.
    """
    _lazy_init()
    idx = current_device()
    default_generator = torch_npu.npu.default_generators[idx]
    return default_generator.initial_seed()
