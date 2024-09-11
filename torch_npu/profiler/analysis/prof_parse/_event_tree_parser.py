from queue import PriorityQueue
from typing import List, Dict, Optional, Union, Tuple
from enum import Enum

from ..prof_bean._torch_op_bean import TorchOpBean
from ..prof_bean._memory_use_bean import MemoryUseBean
from ..prof_common_func._constant import Constant, print_error_msg
from ..prof_common_func._file_tag import FileTag
from ._fwk_file_parser import FwkFileParser
from ._python_trace_parser import PythonTraceParser, PyTraceEvent, CallType

__all__ = []


BASE_16 = 16


class _EventType(Enum):
    TorchOp = 0
    Allocation = 1
    PyCall = 2


class _AllocEventType(Enum):
    MALLOC = 0
    FREE = 1
    BLOCK_FREE = 2


class _RecordScope(Enum):
    FUNCTION = 0
    BACKWARD_FUNCTION = 1
    TORCHSCRIPT_FUNCTION = 2
    KERNEL_FUNCTION_DTYPE = 3
    CUSTOM_CLASS = 4
    BUILD_FEATURE = 5
    LIFE_INTERPRETER = 6
    USER_SCOPE = 7
    STATIC_RUNTIME_OP = 8
    STATIC_RUNTIME_MODEL = 9
    NUM_SCOPES = 10


class _DeviceType(Enum):
    CPU = 0
    CUDA = 1
    NPU = 20        # PrivateUse1 in pytorch


class TensorEnum(Enum):
    TENSOR_IMPL = 0
    STORAGE_PTR = 1
    DTYPE = 2
    DTYPE_SIZE = 3
    SIZES = 4
    STRIDES = 5
    DEVICE_TYPE = 6
    DEVICE_INDEX = 7
    NUM_FIELDS = 8


class ModuleParamEnum(Enum):
    NAME = 0
    METADATA = 1
    GRAD = 2
    NUM_FIELDS = 3


class OptimizerParamEnum(Enum):
    METADATA = 0
    GRAD = 1
    STATE = 2
    NUM_FIELDS = 3


class StateParamEnum(Enum):
    NAME = 0
    METADATA = 1
    NUM_FIELDS = 2


# A tensor is assigned an ID, and a storage is assigned an allocation ID.
# There can be a one-to-many relationshape between a tensor and its storage.
# The tensor ID and allocation ID are calculated in calculate_unique_id().
class _IDReference:
    def __init__(self, tensor_id: Optional[int], allocation_id: Optional[int]):
        self.tensor_id = tensor_id
        self.allocation_id = allocation_id


class _RawTensorInfo:
    def __init__(self, impl: Optional[int], ptr: Optional[int], device_type: int,
                 device_index: int, is_free: bool, id_ref: _IDReference):
        self.impl = impl
        self.ptr = ptr
        self.device_type = device_type
        self.device_index = device_index
        self.is_free = is_free
        self.id_ref = id_ref


class _TensorMetadata:
    def __init__(self, impl: int, ptr: Optional[int], dtype: str, dtype_size: int,
                 sizes: List[int], strides: List[int], device_type: int, device_index: int):
        self.impl = impl
        self.ptr = ptr
        self.dtype = dtype
        self.dtype_size = dtype_size
        self.sizes = sizes
        self.strides = strides
        self.device_type = device_type
        self.device_index = device_index
        self.id = _IDReference(None, None)
    
    @property
    def tensor_id(self) -> Optional[int]:
        return self.id.tensor_id
    
    @property
    def allocation_id(self) -> Optional[int]:
        return self.id.allocation_id


def parse_tensor_metadata(tensor_str: str) -> Optional[_TensorMetadata]:
    parts = tensor_str.split(';')
    if len(parts) != TensorEnum.NUM_FIELDS.value:
        return None
    
    impl = int(parts[TensorEnum.TENSOR_IMPL.value], BASE_16)
    ptr = int(parts[TensorEnum.STORAGE_PTR.value], BASE_16) if parts[TensorEnum.STORAGE_PTR.value] else None
    dtype = parts[TensorEnum.DTYPE.value]
    dtype_size = int(parts[TensorEnum.DTYPE_SIZE.value])
    sizes = list(map(int, parts[TensorEnum.SIZES.value].split(','))) if parts[TensorEnum.SIZES.value] else []
    strides = list(map(int, parts[TensorEnum.STRIDES.value].split(','))) if parts[TensorEnum.STRIDES.value] else []
    device_type = int(parts[TensorEnum.DEVICE_TYPE.value])
    device_index = int(parts[TensorEnum.DEVICE_INDEX.value])

    return _TensorMetadata(
        impl=impl,
        ptr=ptr,
        dtype=dtype,
        dtype_size=dtype_size,
        sizes=sizes,
        strides=strides,
        device_type=device_type,
        device_index=device_index,
    )


def parse_input_from_string(types: Optional[str], 
                            tensors: Optional[str], 
                            scalars: Optional[str]
                            ) -> List[Union[_TensorMetadata, str]]:
    input_types = types.strip().split(';\r\n') if types is not None else []
    input_tensors = tensors.strip().split(')') if tensors is not None else []
    input_scalars = scalars.strip().split(';') if scalars is not None else []

    inputs: List[Union[_TensorMetadata, str]] = []

    tensors_index: int = 0
    scalars_index: int = 0
    for t in input_types:
        if t not in ["None", "Scalar", "ScalarList", "TensorList"]:
            inputs.append(parse_tensor_metadata(input_tensors[tensors_index]))
            tensors_index += 1
        elif t == "Scalar":
            inputs.append(input_scalars[scalars_index])
            scalars_index += 1
        else:
            inputs.append(t)
    return inputs


class _ExtraFields_TorchOp:
    def __init__(self, bean: TorchOpBean):
        self.name = bean.name
        self.end_time_ns = bean.end_ns
        self.scope = bean.scope
        self.forward_tid = bean.args.get(Constant.FORWARD_THREAD_ID, -1)
        self.sequence_num = bean.args.get(Constant.SEQUENCE_NUMBER, -1)
        
        types_string = bean.args.get(Constant.INPUT_DTYPES, None)
        tensors_string = bean.inputs.get(Constant.INPUT_TENSORS, None)
        scalars_string = bean.inputs.get(Constant.INPUT_SCALARS, None)
        self.inputs = parse_input_from_string(types_string, tensors_string, scalars_string)


class _ExtraFields_Allocation:
    def __init__(self, bean: MemoryUseBean):
        self.ptr = bean.ptr
        self.alloc_size = bean.alloc_size_for_db
        self.total_active = bean.total_active_for_db
        self.total_allocated = bean.total_allocated_for_db
        self.total_reserved = bean.total_reserved_for_db
        self.device_type = bean.device_type
        self.device_index = bean.device_index
        self.id = _IDReference(None, None)
    
    @property
    def tensor_id(self) -> Optional[int]:
        return self.id.tensor_id
    
    @property
    def allocation_id(self) -> Optional[int]:
        return self.id.allocation_id


class _ModuleParam:
    def __init__(self, name: str, tensor: _TensorMetadata, grad: Optional[_TensorMetadata]):
        self.name = name
        self.tensor = tensor
        self.grad = grad


class _OptimizerParam:
    def __init__(self,
                 tensor: _TensorMetadata,
                 grad: Optional[_TensorMetadata],
                 state: Optional[List[Tuple[str, _TensorMetadata]]]):
        self.tensor = tensor
        self.grad = grad
        self.state = state


def parse_module_param(param_str: str) -> Optional[_ModuleParam]:
    param_list = param_str.strip().split(')')
    if len(param_list) != ModuleParamEnum.NUM_FIELDS.value:
        return None
    
    name = param_list[ModuleParamEnum.NAME.value]
    tensor = parse_tensor_metadata(param_list[ModuleParamEnum.METADATA.value])
    grad = (parse_tensor_metadata(param_list[ModuleParamEnum.GRAD.value])
            if param_list[ModuleParamEnum.GRAD.value] else None)
    return _ModuleParam(name, tensor, grad)


def parse_state_param(state_str: str) -> Optional[List[Tuple[str, _TensorMetadata]]]:
    state_list = state_str.strip().split(']')
    state_pairs: List[Tuple[str, _TensorMetadata]] = []
    for st in state_list:
        if not st:
            continue
        state_pair = st.strip().split('>')
        if len(state_pair) != StateParamEnum.NUM_FIELDS.value:
            return None
        state_pairs.append(tuple([state_pair[StateParamEnum.NAME.value],
                                parse_tensor_metadata(state_pair[StateParamEnum.METADATA.value])]))
    
    return state_pairs


def parse_optimizer_param(param_str: str) -> Optional[_OptimizerParam]:
    param_list = param_str.strip().split(')')
    if len(param_list) != OptimizerParamEnum.NUM_FIELDS.value:
        return None
    
    tensor = parse_tensor_metadata(param_list[OptimizerParamEnum.METADATA.value])
    grad = (parse_tensor_metadata(param_list[OptimizerParamEnum.GRAD.value])
            if param_list[OptimizerParamEnum.GRAD.value] else None)
    state = (parse_state_param(param_list[OptimizerParamEnum.STATE.value])
            if param_list[OptimizerParamEnum.STATE.value] else [])
    return _OptimizerParam(tensor, grad, state)


class _ExtraFields_PyCall:
    def __init__(self, bean: PyTraceEvent):
        self.name = bean.name
        self.end_time_ns = bean.end_time
        self.key = None
        self.module_parameters = None
        self.optimizer_parameters = None

        # params contain key of PyCall and a list of parameters in PyCall
        if bean.params is not None:
            self.key = bean.params.key
            if bean.params.module_params:
                self.module_parameters = [parse_module_param(param) for param in bean.params.module_params]
            # OptimizerCall belong PyCall and has parameters
            elif bean.params.optimizer_params:
                self.optimizer_parameters = [parse_optimizer_param(param) for param in bean.params.optimizer_params]


class _ProfilerEvent:
    def __init__(self, bean: Union[TorchOpBean, MemoryUseBean, PyTraceEvent]):
        self.finished = False
        self.parent = None
        self.children: List['_ProfilerEvent'] = []
        if isinstance(bean, TorchOpBean):
            self.tag = _EventType.TorchOp
            self.tid = bean._tid
            self.start_time_ns = bean._start_ns
            self.extra_fields = _ExtraFields_TorchOp(bean)
        elif isinstance(bean, MemoryUseBean):
            self.tag = _EventType.Allocation
            self.tid = bean.tid
            self.start_time_ns = bean.time_ns
            self.extra_fields = _ExtraFields_Allocation(bean)
        elif isinstance(bean, PyTraceEvent):
            self.tag = _EventType.PyCall
            self.tid = bean.tid
            self.start_time_ns = bean.ts
            self.extra_fields = _ExtraFields_PyCall(bean)
    
    @property
    def name(self) -> str:
        if self.tag == _EventType.TorchOp:
            return str(self.extra_fields.name)
        elif self.tag == _EventType.Allocation:
            return "[Memory]"
        elif self.tag == _EventType.PyCall:
            return self.extra_fields.name
        return ""
    
    @property
    def end_time_ns(self) -> int:
        if self.tag == _EventType.TorchOp:
            return self.extra_fields.end_time_ns
        elif self.tag == _EventType.Allocation:
            return self.start_time_ns
        elif self.tag == _EventType.PyCall:
            return self.extra_fields.end_time_ns
        return -1
    
    def __lt__(self, other: '_ProfilerEvent') -> bool:
        return self.end_time_ns < other.end_time_ns


def mark_finished(event: _ProfilerEvent) -> bool:
    if event.finished:
        print_error_msg("Error when building tree: the event finished.")
        return False
    event.finished = True
    return True


def push_event(event: _ProfilerEvent,
               thread_event: Dict[int, _ProfilerEvent],
               unfinished_events: PriorityQueue) -> bool:
    if event.parent:
        print_error_msg("Error when building tree: the event had parent.")
        return False
    for child in event.children:
        if not child.finished:
            print_error_msg("Error when building tree: the child dose not finished.")
            return False
    if event.finished:
        print_error_msg("Error when building tree: the event finished.")
        return False
    
    parent = thread_event.get(event.tid)
    if parent is None:
        fwd_tid = event.extra_fields.forward_tid if event.tag == _EventType.TorchOp else 0
        if fwd_tid:
            parent = thread_event.get(fwd_tid)
    
    if parent is not None:
        event.parent = parent
        parent.children.append(event)
    
    if event.end_time_ns > event.start_time_ns:
        thread_event[event.tid] = event
        unfinished_events.put((event.end_time_ns, event))
    # In Allocation event, start time equals to end time.
    else:
        if not mark_finished(event):
            return False
    
    return True


def pop_event(event: _ProfilerEvent, thread_event: Dict[int, _ProfilerEvent]) -> bool:
    if event.finished:
        return True
    
    tid = event.tid
    cur_event = thread_event.get(tid)
    if cur_event is None:
        print_error_msg("Error when building tree: current event is none.")
        return False
    
    while cur_event != event:
        if not mark_finished(cur_event):
            return False
        if cur_event.parent is None:
            print_error_msg("Error when building tree: current event's parent is None.")
            return False
        cur_event = cur_event.parent
    
    if not mark_finished(event):
        return False
    thread_event.pop(tid, None)
    if event.parent:
        thread_event[tid] = event.parent
    
    return True


def build_event_tree(sorted_events: List[_ProfilerEvent]) -> None:
    thread_event: Dict[int, _ProfilerEvent] = {}        # Record current event for each thread
    unfinished_events = PriorityQueue()                 # A priority_queue sorted by end_time_ns

    # Stack replay loop.
    for ev in sorted_events:
        while not unfinished_events.empty() and unfinished_events.queue[0][0] < ev.start_time_ns:
            _, top_event = unfinished_events.get()
            if not pop_event(top_event, thread_event):
                return
        if not push_event(ev, thread_event, unfinished_events):
            return
    
    # Cleanup remaining exit events.
    while not unfinished_events.empty():
        _, top_event = unfinished_events.get()
        if not pop_event(top_event, thread_event):
            return


def get_tensor_info(sorted_events: List[_ProfilerEvent]) -> List[_RawTensorInfo]:
    tensors: List[_RawTensorInfo] = []

    seen_pycalls = set()
    for ev in sorted_events:
        if ev.tag == _EventType.TorchOp:
            tensors.extend(
                _RawTensorInfo(i.impl, i.ptr, i.device_type, i.device_index, False, i.id)
                for i in ev.extra_fields.inputs
                if isinstance(i, _TensorMetadata)
            )
        elif ev.tag == _EventType.Allocation:
            tensors.append(_RawTensorInfo(None,
                                            ev.extra_fields.ptr,
                                            ev.extra_fields.device_type,
                                            ev.extra_fields.device_index,
                                            ev.extra_fields.alloc_size < 0,
                                            ev.extra_fields.id))
        elif ev.tag == _EventType.PyCall:
            if ev.extra_fields.key is None or ev.extra_fields.key in seen_pycalls:
                continue
            
            seen_pycalls.add(ev.extra_fields.key)
            if ev.extra_fields.module_parameters is not None:
                for p in ev.extra_fields.module_parameters:
                    if p is None:
                        continue
                    tensors.append(_RawTensorInfo(p.tensor.impl, p.tensor.ptr, p.tensor.device_type,
                                                  p.tensor.device_index, False, p.tensor.id))
                    if p.grad:
                        tensors.append(_RawTensorInfo(p.grad.impl, p.grad.ptr, p.grad.device_type,
                                                      p.grad.device_index, False, p.grad.id))
            elif ev.extra_fields.optimizer_parameters is not None:
                for p in ev.extra_fields.optimizer_parameters:
                    if p is None:
                        continue
                    tensors.append(_RawTensorInfo(p.tensor.impl, p.tensor.ptr, p.tensor.device_type,
                                                  p.tensor.device_index, False, p.tensor.id))
                    if p.grad:
                        tensors.append(_RawTensorInfo(p.grad.impl, p.grad.ptr, p.grad.device_type,
                                                      p.grad.device_index, False, p.grad.id))
                    tensors.extend(
                        _RawTensorInfo(t.impl, t.ptr, t.device_type, t.device_index, False, t.id)
                        for _, t in p.state
                    )
    
    return tensors


# Assign Allocation ID for each Storage, and ID for each tensor.
# A tensor has a unique id, but it can have multiple allocation IDs, 
# because the tensor might use memory multiple times. 
def calculate_unique_id(sorted_events: List[_ProfilerEvent]):
    # Step 1: Flatten events to a uniform representation
    tensors = get_tensor_info(sorted_events)
    
    # Step 2: Assign Allocation IDs for Storage
    counter: int = 0
    storage_map: Dict[Tuple[int, int, int], int] = {}
    for t in tensors:
        key = (t.ptr, t.device_type, t.device_index)
        if key not in storage_map:
            storage_map[key] = counter
            counter += 1
        t.id_ref.allocation_id = storage_map[key]
        if t.is_free:
            storage_map.pop(key, None)
    
    # Step 3: Handle allocation events which we cannot prove are for Tensor storage
    tensor_set = {t.id_ref.allocation_id for t in tensors if t.impl is not None}
    tensors = [t for t in tensors if t.id_ref.allocation_id in tensor_set]
    
    # Step 4: Assign tensor IDs using allocation IDs
    id_map: Dict[int, int] = {}
    counter = 0
    for t in tensors:
        if t.id_ref.allocation_id not in id_map:
            id_map[t.id_ref.allocation_id] = counter
            counter += 1
        
    # Step 5: Write back to Tensor IDs
    for t in tensors:
        if t.id_ref.allocation_id not in id_map:
            print_error_msg("Error when calcuating tensor id: can't find allocation id.")
            return
        t.id_ref.tensor_id = id_map[t.id_ref.allocation_id]


class EventTree:
    def __init__(self, profiler_path: str):
        self.profiler_path = profiler_path
        
        self.events: List[_ProfilerEvent] = []
        self.fetch_op_events(FwkFileParser(self.profiler_path))
        self.fetch_allocation_events(FwkFileParser(self.profiler_path))
        self.fetch_pycall_events(FwkFileParser(self.profiler_path))

        self.sorted_events: List[_ProfilerEvent] = sorted(self.events, key=lambda ev: ev.start_time_ns)
        build_event_tree(self.sorted_events)
        calculate_unique_id(self.sorted_events)
        self.validate_events()

    def fetch_op_events(self, fwk_file_parser: FwkFileParser) -> None:
        op_bean_list: List[TorchOpBean] = fwk_file_parser.get_file_data_by_tag(FileTag.TORCH_OP)
        if not op_bean_list:
            return
        
        op_events = [_ProfilerEvent(op_bean) for op_bean in op_bean_list]

        # Connected Autograd info to the top level annotation
        for i in range(len(op_events) - 1):
            if (op_events[i].extra_fields.scope == _RecordScope.FUNCTION.value
                and op_events[i + 1].extra_fields.scope == _RecordScope.BACKWARD_FUNCTION.value
                and op_events[i].extra_fields.name.startswith("autograd::engine::evaluate_function: ")):
                op_events[i].extra_fields.sequence_num = op_events[i + 1].extra_fields.sequence_num
                op_events[i].extra_fields.forward_tid = op_events[i + 1].extra_fields.forward_tid
        
        self.events.extend(op_events)
    
    def fetch_allocation_events(self, fwk_file_parser: FwkFileParser) -> None:
        mem_bean_list: List[MemoryUseBean] = fwk_file_parser.get_file_data_by_tag(FileTag.MEMORY)
        if not mem_bean_list:
            return
        
        mem_events = [_ProfilerEvent(mem_bean) for mem_bean in mem_bean_list
                        if mem_bean.data_type != _AllocEventType.BLOCK_FREE.value]
        
        self.events.extend(mem_events)
    
    def fetch_pycall_events(self, fwk_file_parser: FwkFileParser) -> None:
        trace_hash_data = fwk_file_parser.get_file_data_by_tag(FileTag.PYTHON_TRACER_HASH)
        func_call_data = fwk_file_parser.get_file_data_by_tag(FileTag.PYTHON_TRACER_FUNC)
        python_param_data = fwk_file_parser.get_file_data_by_tag(FileTag.PARAM_TENSOR_INFO)
        python_trace_parser = PythonTraceParser(trace_hash_data, func_call_data, python_param_data)

        pycall_bean_list = python_trace_parser.get_pycall_data()
        if not pycall_bean_list:
            return
        
        pycall_events = [_ProfilerEvent(pycall_bean) for pycall_bean in pycall_bean_list]

        self.events.extend(pycall_events)
    
    def validate_events(self) -> None:
        for ev in self.sorted_events:
            # Check the time of events is right
            if ev.start_time_ns > ev.end_time_ns:
                print_error_msg(f"Error in {ev.name}: {ev.start_time_ns} > {ev.end_time_ns}.")
                return
            
            # Check the inputs in TorchOp
            if ev.tag == _EventType.TorchOp:
                for i in ev.extra_fields.inputs:
                    if i is None:
                        print_error_msg(f"Error in the inputs of {ev.name}.")
                        return
            elif ev.tag == _EventType.PyCall:
                if (ev.extra_fields.module_parameters is not None
                    and ev.extra_fields.optimizer_parameters is not None):
                    print_error_msg(f"{ev.name} has module parameters and optimizer parameters.")
                    return
                if ev.extra_fields.module_parameters is not None:
                    for p in ev.extra_fields.module_parameters:
                        if p is None:
                            print_error_msg("Error in parsed module parameters.")
                            return
                elif ev.extra_fields.optimizer_parameters is not None:
                    for p in ev.extra_fields.optimizer_parameters:
                        if p is None or p.state is None:
                            print_error_msg("Error in parsed optimizer parameters.")
                            return

    def get_root_nodes(self) -> List[_ProfilerEvent]:
        events: List[_ProfilerEvent] = []
        for ev in self.sorted_events:
            if ev.tag == _EventType.Allocation:
                device_type = ev.extra_fields.device_type
            else:
                device_type = _DeviceType.CPU.value
            if ev.parent is None and device_type == _DeviceType.CPU.value:
                events.append(ev)
        
        return events