import struct
from enum import Enum

from ..profiler_config import ProfilerConfig
from ..prof_common_func.constant import Constant


class TorchOpEnum(Enum):
    START_NS = 0
    END_NS = 1
    SEQUENCE_UNMBER = 2
    PROCESS_ID = 3
    START_THREAD_ID = 4
    END_THREAD_ID = 5
    FORWORD_THREAD_ID = 6
    IS_ASYNC = 7


class TorchOpBean:
    TLV_TYPE_DICT = {
        Constant.OP_NAME: 3,
        Constant.INPUT_SHAPES: 5,
        Constant.INPUT_DTYPES: 4,
        Constant.CALL_STACK: 6,
        Constant.MODULE_HIERARCHY: 7,
        Constant.FLOPS: 8
    }
    CONSTANT_STRUCT = "<3q4Q?"

    def __init__(self, data: dict):
        self._origin_data = data
        self._constant_data = struct.unpack(self.CONSTANT_STRUCT, data.get(Constant.CONSTANT_BYTES))
        self._kernel_list = []
        self._pid = None
        self._tid = None
        self._name = None
        self._start_ns = None
        self._end_ns = None
        self._args = None
        self.init()

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def tid(self) -> int:
        return self._tid

    @property
    def name(self) -> str:
        return self._name

    @property
    def ts(self) -> int:
        return self._start_ns

    @property
    def dur(self) -> int:
        return int(self._end_ns) - int(self._start_ns)

    @property
    def end_ns(self):
        return self._end_ns

    @property
    def args(self):
        return self._args

    @property
    def is_torch_op(self):
        return True

    def init(self):
        self._pid = int(self._constant_data[TorchOpEnum.PROCESS_ID.value])
        self._tid = int(self._constant_data[TorchOpEnum.START_THREAD_ID.value])
        self._name = str(self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.OP_NAME), ""))
        self._start_ns = ProfilerConfig().get_local_time(
            ProfilerConfig().get_timestamp_from_syscnt(self._constant_data[TorchOpEnum.START_NS.value]))
        self._end_ns = ProfilerConfig().get_local_time(
            ProfilerConfig().get_timestamp_from_syscnt(self._constant_data[TorchOpEnum.END_NS.value]))
        self._args = self.get_args()

    def get_args(self) -> dict:
        args = {
            Constant.SEQUENCE_UNMBER: int(self._constant_data[TorchOpEnum.SEQUENCE_UNMBER.value]),
            Constant.FORWORD_THREAD_ID: int(
                self._constant_data[TorchOpEnum.FORWORD_THREAD_ID.value])}
        for type_name, type_id in self.TLV_TYPE_DICT.items():
            if type_name == Constant.OP_NAME:
                continue
            if type_id not in self._origin_data.keys():
                continue
            if type_name in [Constant.INPUT_SHAPES, Constant.INPUT_DTYPES, Constant.CALL_STACK]:
                args[type_name] = self._origin_data.get(type_id).replace(";", ";\r\n")
            else:
                args[type_name] = self._origin_data.get(type_id)
        return args
