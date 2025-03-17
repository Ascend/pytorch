__all__ = ["NPUTensorPipeRpcBackendOptions"]

from typing import Dict, List, Optional, Union

import torch
from torch._C._distributed_rpc import _TensorPipeRpcBackendOptionsBase
from torch._C import _get_privateuse1_backend_name
import torch.distributed.rpc.constants as rpc_constants

from torch_npu.utils._error_code import ErrCode, dist_error

DeviceType = Union[int, str, torch.device]


def _to_device(device: DeviceType) -> torch.device:
    device = torch.device(device)
    if device.type != _get_privateuse1_backend_name():
        raise ValueError(
            "`set_devices` expect a list of "
            f"{_get_privateuse1_backend_name()} devices, but got "
            f"device type {device.type}." + dist_error(ErrCode.VALUE)
        )
    return device


def _to_device_map(
    device_map: Dict[DeviceType, DeviceType]
) -> Dict[torch.device, torch.device]:
    full_device_map: Dict[torch.device, torch.device] = {}
    reverse_map: Dict[torch.device, torch.device] = {}
    for k, v in device_map.items():
        k, v = torch.device(k), torch.device(v)
        if v in reverse_map:
            raise ValueError(
                "`device_map` only supports 1-to-1 mapping, "
                f"trying to map {k} and {reverse_map[v]} to {v}" + dist_error(ErrCode.VALUE)
            )
        full_device_map[k] = v
        reverse_map[v] = k
    return full_device_map


def _to_device_list(devices: List[DeviceType]) -> List[torch.device]:
    return list(map(_to_device, devices))


class NPUTensorPipeRpcBackendOptions(_TensorPipeRpcBackendOptionsBase):
    def __init__(
        self,
        *,
        num_worker_threads: int = rpc_constants.DEFAULT_NUM_WORKER_THREADS,
        rpc_timeout: float = rpc_constants.DEFAULT_RPC_TIMEOUT_SEC,
        init_method: str = rpc_constants.DEFAULT_INIT_METHOD,
        device_maps: Optional[Dict[str, Dict[DeviceType, DeviceType]]] = None,
        devices: Optional[List[DeviceType]] = None,
        _transports: Optional[List] = None,
        _channels: Optional[List] = None,
    ):
        full_device_maps = (
            {}
            if device_maps is None
            else {k: _to_device_map(v) for k, v in device_maps.items()}
        )
        full_device_list = [] if devices is None else _to_device_list(devices)
        super().__init__(
            num_worker_threads,
            _transports,
            _channels,
            rpc_timeout,
            init_method,
            full_device_maps,
            full_device_list,
        )

    def set_device_map(self, to: str, device_map: Dict[DeviceType, DeviceType]):
        full_device_map = _to_device_map(device_map)
        curr_device_maps = super().device_maps

        if to in curr_device_maps:
            for k, v in full_device_map.items():
                if k in curr_device_maps[to] and v != curr_device_maps[to][k]:
                    raise ValueError(
                        "`set_device_map` only supports 1-to-1 mapping, trying"
                        f" to map {k} to {v} and {curr_device_maps[to][k]}" + dist_error(ErrCode.VALUE)
                    )

        super()._set_device_map(to, full_device_map)

    def set_devices(self, devices: List[DeviceType]):
        self.devices = _to_device_list(devices)
