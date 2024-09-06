import struct
from typing import List
from dataclasses import dataclass

from ..prof_common_func._constant import Constant

__all__ = []


@dataclass
class KeyAndParam:
    key: int
    module_params: List[str]
    optimizer_params: List[str]


class ParamTensorBean:
    TLV_TYPE_DICT = {
        Constant.MODULE_PARAM: 2,
        Constant.OPTIMIZER_PARAM: 3,
    }
    CONSTANT_STRUCT = "<Q"

    def __init__(self, data: dict):
        self._origin_data = data
        self._constant_data = struct.unpack(self.CONSTANT_STRUCT, data.get(Constant.CONSTANT_BYTES))
        self._key = self._constant_data[0]
        self._module_params = None
        self._optimizer_params = None
        
        module_params = self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.MODULE_PARAM))
        if module_params is not None:
            self._module_params = [param for param in module_params.split('}')]
        
        optimizer_params = self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.OPTIMIZER_PARAM))
        if optimizer_params is not None:
            self._optimizer_params = [param for param in optimizer_params.split('}')]
    
    @property
    def key(self) -> int:
        return self._key
    
    @property
    def params(self) -> KeyAndParam:
        return KeyAndParam(self._key, self._module_params, self._optimizer_params)