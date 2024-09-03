from queue import PriorityQueue
from typing import List, Dict, Optional, Union, Tuple
from enum import Enum

from ..prof_bean._torch_op_bean import TorchOpBean
from ..prof_bean._memory_use_bean import MemoryUseBean
from ..prof_common_func._constant import Constant, print_error_msg
from ..prof_common_func._file_tag import FileTag
from ._fwk_file_parser import FwkFileParser

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
    # A tensor info have 8 parts in string split by ";".
    if len(parts) != 8:
        return None
    
    impl = int(parts[0], BASE_16)                                       # part 0: TensorImpl
    ptr = int(parts[1], BASE_16) if parts[1] else None                  # part 1: memory address
    dtype = parts[2]                                                    # part 2: data type
    dtype_size = int(parts[3])                                          # part 3: size of dtype
    sizes = list(map(int, parts[4].split(','))) if parts[4] else []     # part 4: tensor sizes
    strides = list(map(int, parts[5].split(','))) if parts[5] else []   # part 5: tensor strides
    device_type = int(parts[6])                                         # part 6: device type
    device_index = int(parts[7])                                        # part 7: device index

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
        self.device_type = bean.device_type
        self.device_index = bean.device_index
        self.id = _IDReference(None, None)
    
    @property
    def tensor_id(self) -> Optional[int]:
        return self.id.tensor_id
    
    @property
    def allocation_id(self) -> Optional[int]:
        return self.id.allocation_id


class _ProfilerEvent:
    def __init__(self, bean: Union[TorchOpBean, MemoryUseBean]):
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
    
    @property
    def name(self) -> str:
        if self.tag == _EventType.TorchOp:
            return str(self.extra_fields.name)
        elif self.tag == _EventType.Allocation:
            return "[Memory]"
        return ""
    
    @property
    def end_time_ns(self) -> int:
        if self.tag == _EventType.TorchOp:
            return self.extra_fields.end_time_ns
        elif self.tag == _EventType.Allocation:
            return self.start_time_ns
        return -1


def mark_finished(event: _ProfilerEvent) -> bool:
    if event.finished:
        print_error_msg("The event finished.")
        return False
    event.finished = True
    return True


def push_event(event: _ProfilerEvent,
               thread_event: Dict[int, _ProfilerEvent],
               unfinished_events: PriorityQueue) -> bool:
    if event.parent:
        print_error_msg("The event had parent.")
        return False
    for child in event.children:
        if not child.finished:
            print_error_msg("The child dose not finished.")
            return False
    if event.finished:
        print_error_msg("The event finished.")
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
        print_error_msg("Current event should not be None.")
        return False
    
    while cur_event != event:
        if not mark_finished(cur_event):
            return False
        if cur_event.parent is None:
            print_error_msg("Current event's parent should not be None.")
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
            print_error_msg("Can't find allocation id.")
            return
        t.id_ref.tensor_id = id_map[t.id_ref.allocation_id]


class EventTree:
    def __init__(self, profiler_path: str):
        self.profiler_path = profiler_path
        
        self.events: List[_ProfilerEvent] = []
        self.fetch_op_events(FwkFileParser(self.profiler_path))
        self.fetch_allocation_events(FwkFileParser(self.profiler_path))

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
                        if mem_bean.data_type != _AllocEventType.FREE.value]
        
        self.events.extend(mem_events)
    
    def validate_events(self) -> None:
        for ev in self.sorted_events:
            # Check the time of events is right
            if ev.start_time_ns > ev.end_time_ns:
                print_error_msg(f"The start time is later than end time in {ev.name}.")
                return
            
            # Check the inputs in TorchOp
            if ev.tag == _EventType.TorchOp:
                for i in ev.extra_fields.inputs:
                    if i is None:
                        print_error_msg(f"The inputs info of {ev.name} is wrong.")
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