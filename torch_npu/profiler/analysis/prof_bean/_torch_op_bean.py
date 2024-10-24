import struct
from enum import Enum

from .._profiler_config import ProfilerConfig
from ..prof_common_func._constant import Constant

__all__ = []


class TorchOpEnum(Enum):
    START_NS = 0
    END_NS = 1
    SEQUENCE_NUMBER = 2
    PROCESS_ID = 3
    START_THREAD_ID = 4
    END_THREAD_ID = 5
    FORWARD_THREAD_ID = 6
    SCOPE = 7
    IS_ASYNC = 8


class TorchOpBean:
    TLV_TYPE_DICT = {
        Constant.OP_NAME: 2,
        Constant.INPUT_SHAPES: 4,
        Constant.INPUT_DTYPES: 3,
        Constant.INPUT_TENSORS: 5,
        Constant.INPUT_TENSORLISTS: 6,
        Constant.INPUT_SCALARS: 7,
        Constant.CALL_STACK: 8,
        Constant.MODULE_HIERARCHY: 9,
        Constant.FLOPS: 10
    }
    CONSTANT_STRUCT = "<3q4QB?"

    def __init__(self, data: dict):
        self._origin_data = data
        self._constant_data = struct.unpack(self.CONSTANT_STRUCT, data.get(Constant.CONSTANT_BYTES))
        self._kernel_list = []
        self._pid = None
        self._tid = None
        self._name = None
        self._start_ns = None
        self._end_ns = None
        self._call_stack = None
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
    def call_stack(self):
        return self._call_stack
    
    @property
    def inputs(self):
        return self._inputs
    
    @property
    def scope(self):
        return self._scope

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
        self._scope = int(self._constant_data[TorchOpEnum.SCOPE.value])
        self._call_stack = self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.CALL_STACK), "").replace(";", ";\r\n")
        self._args = self.get_args()
        self._inputs = {
            Constant.INPUT_TENSORS: self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.INPUT_TENSORS)),
            Constant.INPUT_TENSORLISTS: self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.INPUT_TENSORLISTS)),
            Constant.INPUT_SCALARS: self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.INPUT_SCALARS))}

    def get_args(self) -> dict:
        args = {
            Constant.SEQUENCE_NUMBER: int(self._constant_data[TorchOpEnum.SEQUENCE_NUMBER.value]),
            Constant.FORWARD_THREAD_ID: int(
                self._constant_data[TorchOpEnum.FORWARD_THREAD_ID.value])}
        for type_name, type_id in self.TLV_TYPE_DICT.items():
            if type_name in [Constant.OP_NAME, Constant.INPUT_TENSORS,
                             Constant.INPUT_TENSORLISTS, Constant.INPUT_SCALARS]:
                continue
            if type_id not in self._origin_data.keys():
                continue
            if type_name in [Constant.INPUT_SHAPES, Constant.INPUT_DTYPES, Constant.CALL_STACK]:
                args[type_name] = self._origin_data.get(type_id).replace(";", ";\r\n")
            else:
                args[type_name] = self._origin_data.get(type_id)
        return args
