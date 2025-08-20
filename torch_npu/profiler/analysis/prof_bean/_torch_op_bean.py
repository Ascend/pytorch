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
    CONSTANT_UNPACKER = struct.Struct(CONSTANT_STRUCT)

    REPLACE_FIELDS = {Constant.INPUT_SHAPES, Constant.INPUT_DTYPES, Constant.CALL_STACK}
    SKIP_FIELDS = {Constant.OP_NAME, Constant.INPUT_TENSORS, Constant.INPUT_TENSORLISTS, Constant.INPUT_SCALARS}

    def __init__(self, data: dict):
        self._origin_data = data
        self._constant_data = self.CONSTANT_UNPACKER.unpack(data.get(Constant.CONSTANT_BYTES))
        self._kernel_list = []
        self._pid = None
        self._tid = None
        self._name = None
        self._start_ns = None
        self._end_ns = None
        self._call_stack = None
        self._args = None
        self._inputs = None
        self._scope = None
        self._dur = None

    @property
    def pid(self) -> int:
        if self._pid is None:
            self._pid = self._constant_data[TorchOpEnum.PROCESS_ID.value]
        return self._pid

    @property
    def tid(self) -> int:
        if self._tid is None:
            self._tid = self._constant_data[TorchOpEnum.START_THREAD_ID.value]
        return self._tid

    @property
    def name(self) -> str:
        if self._name is None:
            self._name = str(self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.OP_NAME), ""))
        return self._name

    @property
    def ts(self) -> int:
        if self._start_ns is None:
            self._init_timestamps()
        return self._start_ns

    @property
    def dur(self) -> int:
        if self._dur is None:
            if self._start_ns is None or self._end_ns is None:
                self._init_timestamps()
            self._dur = self._end_ns - self._start_ns
        return self._dur

    @property
    def end_ns(self):
        if self._end_ns is None:
            self._init_timestamps()
        return self._end_ns

    @property
    def call_stack(self):
        if self._call_stack is None:
            self._call_stack = self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.CALL_STACK), "").replace(";", ";\r\n")
        return self._call_stack

    @property
    def inputs(self):
        if self._inputs is None:
            self._inputs = {
                Constant.INPUT_TENSORS: self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.INPUT_TENSORS)),
                Constant.INPUT_TENSORLISTS: self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.INPUT_TENSORLISTS)),
                Constant.INPUT_SCALARS: self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.INPUT_SCALARS))
            }
        return self._inputs

    @property
    def scope(self):
        if self._scope is None:
            self._scope = self._constant_data[TorchOpEnum.SCOPE.value]
        return self._scope

    @property
    def args(self):
        if self._args is None:
            self._args = self.get_args()
        return self._args

    @property
    def is_torch_op(self):
        return True
    
    def _init_timestamps(self):
        profiler_config = ProfilerConfig()
        start_syscnt = self._constant_data[TorchOpEnum.START_NS.value]
        end_syscnt = self._constant_data[TorchOpEnum.END_NS.value]

        self._start_ns = profiler_config.get_local_time(profiler_config.get_timestamp_from_syscnt(start_syscnt))
        self._end_ns = profiler_config.get_local_time(profiler_config.get_timestamp_from_syscnt(end_syscnt))

    def get_args(self) -> dict:
        args = {
            Constant.SEQUENCE_NUMBER: self._constant_data[TorchOpEnum.SEQUENCE_NUMBER.value],
            Constant.FORWARD_THREAD_ID: self._constant_data[TorchOpEnum.FORWARD_THREAD_ID.value]
        }
        origin_keys = self._origin_data.keys()
        for type_name, type_id in self.TLV_TYPE_DICT.items():
            if type_name in self.SKIP_FIELDS or type_id not in origin_keys:
                continue

            value = self._origin_data[type_id]
            if type_name in self.REPLACE_FIELDS:
                args[type_name] = value.replace(";", ";\r\n") if value else ""
            else:
                args[type_name] = value
        return args
