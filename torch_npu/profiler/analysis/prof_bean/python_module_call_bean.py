import struct
from enum import Enum
from ..profiler_config import ProfilerConfig
from ..prof_common_func.constant import Constant

__all__ = []


class PythonModuleCallEnum(Enum):
    IDX = 0
    THREAD_ID = 1
    PROCESS_ID = 2


class PythonModuleCallBean:
    CONSTANT_STRUCT = "<3Q"
    TLV_TYPE_DICT = {
        Constant.MODULE_UID: 2,
        Constant.MODULE_NAME: 3
    }

    def __init__(self, data: dict):
        self._origin_data = data
        self._constant_data = struct.unpack(self.CONSTANT_STRUCT, data.get(Constant.CONSTANT_BYTES))
        self._idx = int(self._constant_data[PythonModuleCallEnum.IDX.value])
        self._tid = int(self._constant_data[PythonModuleCallEnum.THREAD_ID.value])
        self._pid = int(self._constant_data[PythonModuleCallEnum.PROCESS_ID.value])
        self._module_uid = self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.MODULE_UID), "")
        self._module_name = self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.MODULE_NAME), "")

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def tid(self) -> int:
        return self._tid

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def module_uid(self) -> str:
        return self._module_uid
    
    @property
    def module_name(self) -> str:
        return self._module_name
