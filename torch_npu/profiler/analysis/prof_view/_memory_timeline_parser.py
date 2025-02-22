import io
import dataclasses
import itertools as it
from enum import Enum
from collections import defaultdict
from typing import List, Set, Dict, DefaultDict, Optional, Tuple, Iterator, Any
from base64 import b64encode
import importlib.util

import numpy as np
import torch
from torch._C import FunctionSchema
from torch.profiler._utils import traverse_dfs

from ._base_parser import BaseParser
from ..prof_common_func._path_manager import ProfilerPathManager
from ..prof_common_func._file_manager import FileManager
from ..prof_common_func._log import ProfilerLogger
from ..prof_common_func._constant import Constant, print_warn_msg, print_error_msg
from ..prof_parse._event_tree_parser import (
    EventTree,
    _EventType,
    _RecordScope,
    _DeviceType,
    _ExtraFields_TorchOp,
    _ExtraFields_Allocation,
    _ModuleParam,
    _OptimizerParam,
    _TensorMetadata,
    _ProfilerEvent,
)

__all__ = []

_DEVICE_DICT = {
    "cpu": _DeviceType.CPU.value,
    "npu": _DeviceType.NPU.value,
}

DeviceKeyAndVersion = Tuple["DeviceKey", int]
TensorKeyAndVersion = Tuple["TensorKey", int]


class Category(Enum):
    INPUT = 0
    TEMPORARY = 1
    ACTIVATION = 2
    GRADIENT = 3
    AUTOGRAD_DETAIL = 4
    PARAMETER = 5
    OPTIMIZER_STATE = 6

_CATEGORY_TO_COLORS = {
    Category.PARAMETER: "darkgreen",
    Category.OPTIMIZER_STATE: "goldenrod",
    Category.INPUT: "black",
    Category.TEMPORARY: "mediumpurple",
    Category.ACTIVATION: "red",
    Category.GRADIENT: "mediumblue",
    Category.AUTOGRAD_DETAIL: "royalblue",
    None: "grey",
}
_CATEGORY_TO_INDEX = {c: i for i, c in enumerate(_CATEGORY_TO_COLORS)}


class Action(Enum):
    PREEXISTING = 0
    CREATE = 1
    INCREMENT_VERSION = 2
    DESTROY = 3

_ACTION_TO_INDEX = {i: i.value for i in Action}


@dataclasses.dataclass(unsafe_hash=False, frozen=True)
class DeviceKey:
    """
    Bundle device type and device index.
    """
    device_type: int
    device_index: int

    def __eq__(self, other: "DeviceKey") -> bool:
        return (self.device_type, self.device_index) == (other.device_type, other.device_index)
    
    def __lt__(self, other: "DeviceKey") -> bool:
        return (self.device_type, self.device_index) < (other.device_type, other.device_index)


@dataclasses.dataclass
class Storage:
    """
    Bundle storage pointer and allocation id.
    The `allocation_id` is a unique id for each storage, calculated in
    calculate_unique_id(). Actually the allocation ID is sufficient for
    memory profiling, sometimes however memory address pointers can help
    verify the correctness of the results.
    """
    allocation_id: int
    ptr: int

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Storage) and self.allocation_id == other.allocation_id
    
    def __hash__(self) -> int:
        return hash(self.allocation_id)


@dataclasses.dataclass(eq=True, unsafe_hash=True, frozen=True)
class TensorKey(DeviceKey):
    """
    Assign a unique `id` to each tensor. If the tensor's information
    matches the storage's information using the `id`, the tensor can be
    associated with its storage. Some storages without a corresponding
    tensor will not have this `id`.
    """
    id: int
    storage: Storage

    def __lt__(self, other: "TensorKey") -> bool:
        return self._as_sortable < other._as_sortable
    
    @staticmethod
    def _make(tensor_id: Optional[int], allocation_id: Optional[int], storage_ptr: Optional[int],
              device_type: int, device_index: int) -> Optional["TensorKey"]:
        if tensor_id is None or storage_ptr is None or allocation_id is None:
            return None
        return TensorKey(device_type, device_index, tensor_id, Storage(allocation_id, storage_ptr))
    
    @classmethod
    def from_allocation(cls, alloc: _ExtraFields_Allocation) -> Optional["TensorKey"]:
        return cls._make(alloc.tensor_id, alloc.allocation_id, alloc.ptr, alloc.device_type, alloc.device_index)
    
    @classmethod
    def from_tensor(cls, t: Optional[_TensorMetadata]) -> Optional["TensorKey"]:
        if t is not None:
            return cls._make(t.tensor_id, t.allocation_id, t.ptr, t.device_type, t.device_index)
        return None
    
    @property
    def _as_sortable(self) -> Tuple[int, int, DeviceKey]:
        return self.id, self.storage.allocation_id, DeviceKey(self.device_type, self.device_index)


def check_tensor_in_accumulategrad(node: _ProfilerEvent, child: _ProfilerEvent) -> bool:
    """
    AccumulateGrad is used to handle gradient updates. There are two possible cases:
    1) For a newly created gradient Tensor, there is nothing to accumulate,
       so autograd simply detaches the Tensor in (aten::detach) op.
    2) For a preexisting gradient Tensor, using (aten::add_) op to add update.
    """
    # Accumulate is in backward
    in_backward_scope = (node.extra_fields.scope == _RecordScope.BACKWARD_FUNCTION.value)
    # Check the name. In Windows pltaform, the name have "struct" prefix.
    is_accumulate_grad = (node.name in ("torch::autograd::AccumulateGrad", "struct torch::autograd::AccumulateGrad"))
    if not (in_backward_scope and is_accumulate_grad):
        return False

    # Check the first child of AccumulateGrad op.
    is_torch_op = (child.tag == _EventType.TorchOp)
    # The first child must be `aten::detach` or `aten::add_`.
    is_valid_name = (child.name in ("aten::detach", "aten::add_"))
    # Get first input which is a Tensor.
    have_inputs = (len(child.extra_fields.inputs) > 0)
    if not (is_torch_op and is_valid_name and have_inputs):
        return False

    # Check the first input is Tensor.
    if not isinstance(child.extra_fields.inputs[0], _TensorMetadata):
        return False

    return True


def extract_parameters_and_gradients(
    node: _ProfilerEvent
) -> Iterator[Tuple[Optional[TensorKey], Optional[TensorKey]]]:
    children = node.children
    # Extract the Tensor in "AccumulateGrad"
    if (node.tag == _EventType.TorchOp and children and check_tensor_in_accumulategrad(node, children[0])):
        yield None, TensorKey.from_tensor(children[0].extra_fields.inputs[0])
    # Extract Tensors in `torch.nn.Module` and `torch.optim.Optimizer`
    elif node.tag == _EventType.PyCall:
        typed_fields = node.extra_fields
        if typed_fields.module_parameters is not None:
            for param in typed_fields.module_parameters:
                yield TensorKey.from_tensor(param.tensor), TensorKey.from_tensor(param.grad)
        if typed_fields.optimizer_parameters is not None:
            for param in typed_fields.optimizer_parameters:
                yield TensorKey.from_tensor(param.tensor), TensorKey.from_tensor(param.grad)


def extract_parameters(event: _ProfilerEvent) -> Iterator[TensorKey]:
    for param, _ in extract_parameters_and_gradients(event):
        if param is not None:
            yield param


def extract_gradients(event: _ProfilerEvent) -> Iterator[TensorKey]:
    for _, param_grad in extract_parameters_and_gradients(event):
        if param_grad is not None:
            yield param_grad


def get_scopes(event: Optional[_ProfilerEvent]) -> Tuple[_RecordScope, ...]:
    """
    Get all scopes from the current node to the root node.
    """
    scopes = []
    while event:
        if event.tag == _EventType.TorchOp:
            scopes.append(event.extra_fields.scope)
        event = event.parent
    return tuple(scopes)


class StorageSizeDict:
    """
    Build a Dict to get storage size of Tensors with TensorKey.
    Traverse all events to get tensors, calculate their sizes, and
    store them in a dictionary by `TensorKey`. Additionally, for all
    allocation events, store the allocated memory size by `TensorKey`.
    """
    def __init__(self, sorted_events: List[_ProfilerEvent]):
        self._size_dict: Dict[TensorKey, int] = {}
        for ev in sorted_events:
            if ev.tag == _EventType.TorchOp:
                self._process_tensor_inputs(self._flat_tensor_inputs(ev.extra_fields))
            elif ev.tag == _EventType.PyCall:
                if ev.extra_fields.module_parameters is not None:
                    self._process_module_parameters(ev.extra_fields.module_parameters)
                elif ev.extra_fields.optimizer_parameters is not None:
                    self._process_optimizer_parameters(ev.extra_fields.optimizer_parameters)
        
        allocations: Dict[TensorKey, int] = {}
        for ev in sorted_events:
            if ev.tag == _EventType.Allocation:
                key = TensorKey.from_allocation(ev.extra_fields)
                if key is not None:
                    allocations.setdefault(key, abs(ev.extra_fields.alloc_size))
        self._size_dict.update(allocations)

    def _process_tensor_inputs(self, tensors: List[_TensorMetadata]) -> None:
        for tensor in tensors:
            self._update_size_dict(tensor)

    def _process_module_parameters(self, parameters: List[_ModuleParam]) -> None:
        for p in parameters:
            self._update_size_dict(p.tensor)
            self._update_size_dict(p.grad)

    def _process_optimizer_parameters(self, parameters: List[_OptimizerParam]) -> None:
        for p in parameters:
            self._update_size_dict(p.tensor)
            self._update_size_dict(p.grad)
            for _, t in p.state:
                self._update_size_dict(t)
            
    def _update_size_dict(self, t: Optional[_TensorMetadata]) -> None:
        key = TensorKey.from_tensor(t)
        if key is not None and t is not None:
            num_bytes = t.dtype_size
            for size in t.sizes:
                num_bytes *= size
            self._size_dict[key] = max(self._size_dict.get(key, 0), num_bytes)
    
    @staticmethod
    def _flat_tensor_inputs(op: _ExtraFields_TorchOp) -> List[_TensorMetadata]:
        flat_inputs: List[_TensorMetadata] = []
        for item in op.inputs:
            if isinstance(item, _TensorMetadata):
                flat_inputs.append(item)
            elif isinstance(item, list):
                flat_inputs.extend(t for t in item)
        return flat_inputs
    
    def __getitem__(self, key: TensorKey):
        return self._size_dict.get(key, 0)


class SchemaMatcher:
    """
    Lookup torch operation schema based on its name and arguments.
    Schema information includes key attributes of input tensors that are
    useful for profiling, but this information is not currently collected.
    As an alternative, matching the operator name and input arguments types
    to the registered schemas to obtain the schema information.
    """
    @classmethod
    def inputs_are_mutable(cls, t: _ExtraFields_TorchOp) -> Tuple[Optional[bool], ...]:
        """
        Determine which inputs may have mutated based on function schema.
        Usually, only one schema will match. If multiple schemas match,
        the tensor is considered mutable if any of the schemas are mutable.
        If no schema matches, assume all inputs are mutable.
        """
        mutable: Optional[List[bool]] = None
        for schema in cls.match_schemas(t):
            mutable = mutable or [False for _ in schema.arguments]
            for i, arg in enumerate(schema.arguments):
                mutable[i] = mutable[i] or getattr(arg.alias_info, "is_write", False)
        return tuple(mutable or (None for _ in t.inputs))
    
    @classmethod
    def match_schemas(cls, op: _ExtraFields_TorchOp) -> Tuple[FunctionSchema, ...]:
        signature = tuple(TensorKey.from_tensor(input) if isinstance(input, _TensorMetadata)
                          else [TensorKey.from_tensor(t) for t in input] if isinstance(input, list)
                          else input
                          for input in op.inputs)

        schemas_with_same_name = cls.lookup_schemas(op.name)
        schemas_with_same_pattern: List[FunctionSchema] = []
        
        # This op name can't match a register operation schema.
        if schemas_with_same_name is None:
            return []
        
        for schema in schemas_with_same_name:
            # Match the numbers of arguments
            if len(schema.arguments) != len(signature):
                continue

            # Match arguments type
            matched = True
            for observed, schema_arg in zip(signature, schema.arguments):
                matched = matched and cls._types_match(observed, schema_arg.type)
            if matched:
                schemas_with_same_pattern.append(schema)
        
        return tuple(schemas_with_same_pattern)
    
    @classmethod
    def _types_match(cls, observed, schema_type) -> bool:
        if isinstance(schema_type, torch._C.OptionalType):
            schema_type = schema_type.getElementType()
            return observed == "None" or cls._types_match(observed, schema_type)
        if isinstance(schema_type, torch._C.AnyType):
            return True
        if isinstance(schema_type, torch._C.TensorType):
            return isinstance(observed, TensorKey)
        if schema_type.isSubtypeOf(torch._C.ListType.ofTensors()):
            return isinstance(observed, list) and all(
                isinstance(t, TensorKey) for t in observed
            )
        
        return not (isinstance(observed, TensorKey) or isinstance(observed, list))
    
    @staticmethod
    def lookup_schemas(name: str) -> Optional[Tuple[FunctionSchema, ...]]:
        # Operator names are always namespaced and must include "::".
        # If no schema matching the operator name is found, return None.
        try:
            if "::" not in name:
                return None
            return tuple(torch._C._jit_get_schemas_for_operator(name))
        except RuntimeError:
            return None


@dataclasses.dataclass()
class DataFlowEdge:
    """
    In the dataflow graph, edges represent the input and output tensors of operators.
    Each edge has two attributes: `is_allocation`, which indicates if the tensor required
    memory allocation during the operator's execution, and `is_deletion`, which indicates
    if the tensor was freed during execution.
    """
    input_version: Optional[int] = None
    mutated: Optional[bool] = False

    @property
    def is_allocation(self) -> bool:
        return self.input_version is None
    
    @property
    def is_deletion(self) -> bool:
        return self.mutated is None


class DataFlowNode:
    def __init__(self, event: _ProfilerEvent, graph: "DataFlowGraph"):
        self._event = event
        self._graph = graph
        self._edges: Dict[TensorKey, DataFlowEdge] = self._determine_edges()

        for key, edge in self._edges.items():
            if edge.mutated and not edge.is_allocation:
                self._graph.increase_version(key)
    
    def _determine_edges(self) -> Optional[Dict[TensorKey, DataFlowEdge]]:
        subtree = tuple(traverse_dfs([self._event]))

        # Firstly, populating edges from op input tensors.
        mutable_by_key: Dict[Optional[TensorKey], Set[Optional[bool]]] = {}
        for op in (event.extra_fields for event in subtree if event.tag == _EventType.TorchOp):
            for op_input, mutable in zip(op.inputs, SchemaMatcher.inputs_are_mutable(op)):
                if isinstance(op_input, _TensorMetadata):
                    key = TensorKey.from_tensor(op_input)
                    mutable_by_key.setdefault(key, set()).add(mutable)
                
                if isinstance(op_input, list):
                    for op_input_i in op_input:
                        key = TensorKey.from_tensor(op_input_i)
                        mutable_by_key.setdefault(key, set()).add(mutable)
        
        edges: DefaultDict[Optional[TensorKey], DataFlowEdge] = defaultdict(DataFlowEdge)
        for key, mutable_set in mutable_by_key.items():
            if key is not None:
                edges[key].input_version = self._graph.lookup(key) if key else -1

                # The conservation view is that a tensor to be mutable if encounter
                # a schema where it is a mutable argument OR if it is ambiguous.
                # If a tensor is mutable, assume it is mutated by the operator.
                mutated = (True in mutable_set) or (tuple(mutable_set) == (None,))
                edges[key].mutated = mutated
        
        # Then handle deletions. Note that deleting a Tensor implicitly adds it as an input edge.
        for event in subtree:
            if event.tag == _EventType.Allocation and event.extra_fields.alloc_size < 0:
                key = TensorKey.from_allocation(event.extra_fields)
                edge = edges[key]
                edge.mutated = None
                edge.input_version = self._graph.lookup(key) if key else -1
        
        # Finally handle allocations. This must be handled last because the previous two steps add
        # as many input edges as possible, including tensors generated and released by the operator.
        for event in subtree:
            if event.tag == _EventType.Allocation and event.extra_fields.alloc_size > 0:
                edges[TensorKey.from_allocation(event.extra_fields)].input_version = None
        
        return dict(sorted((key, edge) for key, edge in edges.items() if key is not None))
    
    @property
    def inputs(self) -> Dict[TensorKey, Tuple[bool, int]]:
        """
        Return all input Tensors. Any tensor associated with the operator that
        wasn't generated by it is considered an input.
        """
        return {key: (bool(edge.mutated), edge.input_version)
                for key, edge in self._edges.items()
                if not edge.is_allocation}
    
    @property
    def outputs(self) -> Dict[TensorKey, int]:
        """
        Return all output Tensors. These include tensors allocated but not deleted
        (version set to 0) and mutated input tensors (version incremented by 1).
        """
        return {key: 0 if edge.input_version is None else edge.input_version + 1
                for key, edge in self._edges.items()
                if (edge.is_allocation and not edge.is_deletion) or edge.mutated}
    
    @property
    def intermediates(self) -> Tuple[TensorKey, ...]:
        return tuple(k for k, v in self._edges.items() if v.is_allocation and v.is_deletion)
    
    @property
    def start_time(self) -> int:
        return self._event.start_time_ns


class DataFlowGraph:
    """
    A dataflow graph is a directed graph where the nodes represent torch ops or
    allocation events, and the edges represent the inputs and outputs of the operators.
    """
    def __init__(self, root_nodes: List[_ProfilerEvent]):
        self._root_nodes = root_nodes
        self._leaf_events = self._extract_leaf_events(self._root_nodes)
        self._active_version: Dict[TensorKey, Optional[int]] = {}
        self._flow_nodes = [DataFlowNode(event, self) for event in self.leaf_events]
        self._flow_nodes.sort(key=lambda x: x.start_time)
        self.validate()
    
    @property
    def flow_nodes(self) -> Tuple[DataFlowNode, ...]:
        return tuple(self._flow_nodes)
    
    def validate(self) -> None:
        # Check that each (TensorKey, version) pair has a unique creation node.
        outputs: Set[Tuple[TensorKey, int]] = set()
        for node in self.flow_nodes:
            node_outputs = set(node.outputs.items())
            duplicates = outputs & node_outputs
            if duplicates and any(key.device_type == _DeviceType.NPU.value for key, _ in duplicates):
                print_warn_msg(f"Warn in data flow graph: the NPU memory record maybe incomplete.")
                return
            outputs |= node_outputs

    @property
    def leaf_events(self) -> Tuple[_ProfilerEvent, ...]:
        return self._leaf_events
    
    @staticmethod
    def _leaf_op(event: _ProfilerEvent) -> bool:
        return event.tag == _EventType.TorchOp and (
            event.extra_fields.scope == _RecordScope.BACKWARD_FUNCTION.value
            or bool(SchemaMatcher.match_schemas(event.extra_fields)))
    
    def _get_children(self, event: _ProfilerEvent) -> List[_ProfilerEvent]:
        if self._leaf_op(event) or event.tag == _EventType.Allocation:
            return []
        return event.children

    def _extract_leaf_events(self, root_nodes: List[_ProfilerEvent]) -> Tuple[_ProfilerEvent, ...]:
        """
        Get nodes which are vertices in data flow graph by travesing
        the event tree partially.
        Consider the following code:
        ```
        with record_function("## Init ##"):
            x.zero_()
            y.zero_()
        ```
        The event tree will look like:
          <Python context>
            TorchOp: "## Init ##"
              TorchOp: zero_
                TorchOp: fill_
              TorchOp: zero_
                TorchOp: fill_

        It's important to select the right operator as a node in the
        dataflow graph. In this case, choosing "## Init ##" loses
        detail from subsequent calls, while `fill_` makes the graph
        too detailed. The best nodes are top-level torch ops matching
        the torch operator schema. Memory allocations and frees should
        also be included to capture all memory usage.
        """
        leaf_events: List[_ProfilerEvent] = []
        for event in traverse_dfs(root_nodes, children_fn=lambda e: self._get_children(e)):
            if self._leaf_op(event) or event.tag == _EventType.Allocation:
                leaf_events.append(event)
        return tuple(sorted(leaf_events, key=lambda x: x.start_time_ns))
    
    def lookup(self, key: TensorKey) -> int:
        version = self._active_version.setdefault(key, 0)
        return version
    
    def increase_version(self, key: TensorKey):
        prior_version = self._active_version.get(key)
        self._active_version[key] = prior_version + 1


@dataclasses.dataclass
class CategoryElement:
    """
    Set category by tensor id or TensorKey or (TensorKey, version).
    Note the PARAMETER, GRADIENT, OPTIMIZER_STATE are set by tensor id. 
    The TEMPORARY is set by TensorKey. The INPUT, ACTIVATION, AUTOGRAD_DETAIL
    are set by (TensorKey, version).
    """
    by_id: Optional[Category] = None
    by_key: Dict[TensorKey, Category] = dataclasses.field(default_factory=dict)
    by_version: Dict[TensorKeyAndVersion, Category] = dataclasses.field(default_factory=dict)
    by_id_keyset: Set[TensorKey] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class CategoryDict:
    _category_dict: DefaultDict[int, CategoryElement] = dataclasses.field(
        default_factory=lambda: defaultdict(CategoryElement))

    def set_by_id(self, key: TensorKey, category: Category) -> None:
        self._category_dict[key.id].by_id = category
        self._category_dict[key.id].by_id_keyset.add(key)

    def set_by_key(self, key: TensorKey, category: Category) -> None:
        self._category_dict[key.id].by_key[key] = category

    def setdefault_by_version(self, key: TensorKey, version: int, category: Category) -> None:
        self._category_dict[key.id].by_version.setdefault((key, version), category)

    def get(self, key: DeviceKey, version: int) -> Optional[Category]:
        """
        If a Tensor can be categorized by its ID, return by_id. If not, try categorizing
        by TensorKey, and if that fails, categorize by (TensorKey, version).
        """
        if isinstance(key, DeviceKey) and not isinstance(key, TensorKey):
            return None
        element = self._category_dict[key.id]
        return element.by_id or element.by_key.get(key) or element.by_version.get((key, version))


class MemoryProfile:
    def __init__(self, file_path: str) -> None:
        self._event_tree = EventTree(file_path)
        self._root_nodes = self._event_tree.get_root_nodes()
        self._data_flow_graph = DataFlowGraph(self._root_nodes)
        self._storage_size_dict = StorageSizeDict(self._event_tree.sorted_events)
        self._categories = CategoryDict()

        self._set_gradients_and_temporaries()
        self._set_parameters_using_python_tracer()
        self._set_inputs()
        self._set_parameters_using_data_flow()
        self._set_activations()
        self._set_optimizer_state()
        self._set_autograd_detail()

    @property
    def timeline(self) -> Tuple[Tuple[int, Action, DeviceKeyAndVersion, int], ...]:
        """
        Return memory timeline. The memory timeline records [timestamp,
        action, (key, version), size] for each allocation or free event.
        """
        output: List[Tuple[int, Action, DeviceKeyAndVersion, int]] = []
        allocation_times: Dict[Tuple[TensorKey, bool], int] = {}
        live_unknown: Dict[Tuple[int, int, int], bool] = {}

        # First, handle Allocation event. Stores timestamp of allcoation event
        # in `allocation_times`. Add storage to `output` that can't match Tensors.
        mem_event_list = [event for event in traverse_dfs(self._root_nodes) if event.tag == _EventType.Allocation]
        mem_event_list.sort(key=lambda x: x.start_time_ns)
        for event in mem_event_list:
            alloc_fields = event.extra_fields
            alloc_size = alloc_fields.alloc_size
            is_allocation = alloc_size > 0
            ts = event.start_time_ns

            tkey = TensorKey.from_allocation(alloc_fields)
            if tkey is not None:
                allocation_times[(tkey, is_allocation)] = ts
                continue

            device_key = DeviceKey(alloc_fields.device_type, alloc_fields.device_index)
            ptr_and_device = (alloc_fields.ptr, device_key)
            if is_allocation:
                if ptr_and_device in live_unknown:
                    output.append((ts, Action.INCREMENT_VERSION, (device_key, 0), alloc_size))
                else:
                    live_unknown[ptr_and_device] = True
                    output.append((ts, Action.CREATE, (device_key, 0), alloc_size))
            else:
                output.append((ts, Action.DESTROY, (device_key, 0), -alloc_size))
                if not live_unknown.pop(ptr_and_device, False):
                    output.append((-1, Action.PREEXISTING, (device_key, 0), -alloc_size))

        snapshot = self._category_snapshot()            # Get all (key, version) pairs
        last_version = dict(sorted(snapshot.keys()))    # For each key, get last version

        events: List[Tuple[int, Action, TensorKeyAndVersion]] = [
            (-1, Action.PREEXISTING, (key, version))
            for key, version in snapshot.keys()
            if (key, True) not in allocation_times and version == 0
        ]

        # Add storage of Tensors to `output` by traverse data flow graph.
        for node in self._data_flow_graph.flow_nodes:
            for key, edge in node._edges.items():
                if edge.is_allocation:
                    ts = allocation_times.get((key, True))
                    events.append((ts, Action.CREATE, (key, 0)))
                elif edge.mutated:
                    ts = node._event.start_time_ns
                    version = edge.input_version
                    events.append((ts, Action.INCREMENT_VERSION, (key, version)))
                if edge.is_deletion:
                    ts = allocation_times.get((key, False))
                    events.append((ts, Action.DESTROY, (key, last_version[key])))

        output.extend((time, action, (key, version), self._storage_size_dict[key])
                    for time, action, (key, version) in events)
        output.sort(key=lambda x: (x[0], x[1].value))
        return tuple(output)
    
    @property
    def memory_history(self) -> List[Tuple[DeviceKey, int, int, int]]:
        """
        Get memory usage history, to output the memory usage peak.
        """
        result: List[Tuple[DeviceKey, int, int, int]] = []
        for event in traverse_dfs(self._root_nodes):
            if event.tag == _EventType.Allocation:
                device = DeviceKey(event.extra_fields.device_type, event.extra_fields.device_index)
                result.append(tuple((device,
                                     event.extra_fields.total_active,
                                     event.extra_fields.total_allocated,
                                     event.extra_fields.total_reserved)))
        return result

    def _is_gradient(self, *args, **kwargs) -> bool:
        return self._categories.get(*args, **kwargs) == Category.GRADIENT
    
    @staticmethod
    def _is_backward(event: _ProfilerEvent) -> bool:
        if _RecordScope.BACKWARD_FUNCTION.value in get_scopes(event):
            return True
        return False

    def _category_snapshot(self) -> Dict[TensorKeyAndVersion, Optional[Category]]:
        """
        Get category for each (TensorKey, version) pair.
        """
        all_tensor_versions: Set[TensorKeyAndVersion] = set()
        for node in self._data_flow_graph.flow_nodes:
            all_tensor_versions.update(((key, version) for key, (_, version) in node.inputs.items()))
            all_tensor_versions.update((key, 0) for key in node.intermediates)
            all_tensor_versions.update(node.outputs.items())
        
        for category_element in self._categories._category_dict.values():
            all_tensor_versions.update((key, 0) for key in category_element.by_id_keyset)
        
        return {(key, version): self._categories.get(key, version)
                for key, version in sorted(all_tensor_versions)}

    def _any_version_depends_on_gradient(self) -> Set[int]:
        """
        Extract IDs of Tensors which depend or will depend on a gradient.
        Update the set of Tensor IDs that depend on or will depend on gradients during the loop.
        Tensors with GRADIENT and PARAMETER types directly depend on gradients, so if a tensor
        depends on a gradient or parameter, its ID should be added to `depends_on_gradient`.
        """
        model_relevant = {Category.GRADIENT, Category.PARAMETER}
        depends_on_gradient: Set[int] = set()
        while True:
            start_size = len(depends_on_gradient)
            for node in self._data_flow_graph.flow_nodes:
                ids = tuple(key.id for key, (_, version) in node.inputs.items()
                        if self._categories.get(key, version) in model_relevant or key.id in depends_on_gradient)
                if ids:
                    depends_on_gradient.update(ids)
                    depends_on_gradient.update(key.id for key in node.outputs)

            # Exit can be guaranteed because there is a finite set of TensorKeyAndVersion pairs.
            # In practice, looping twice is sufficient to capture the IDs of Tensors that depend
            # on gradients. In last loop, no more elements are added.
            if len(depends_on_gradient) == start_size:
                return depends_on_gradient

    def _set_gradients_and_temporaries(self) -> None:
        """
        Mark Tensors which are unambiguous and simple to recognize.
        """

        # Get gradient Tensors from `AccumulateGrad` ops directly.
        for event in traverse_dfs(self._root_nodes):
            for param_grad in extract_gradients(event):
                self._categories.set_by_id(param_grad, Category.GRADIENT)

        # Temporary Tensors denotes those Tensors only used in one operation.
        for node in self._data_flow_graph.flow_nodes:
            for intermediate in node.intermediates:
                self._categories.set_by_key(intermediate, Category.TEMPORARY)

    def _set_parameters_using_python_tracer(self) -> None:
        for event in traverse_dfs(self._root_nodes):
            for param in extract_parameters(event):
                if param is not None:
                    self._categories.set_by_id(param, Category.PARAMETER)

    def _set_inputs(self) -> None:
        """
        Mark inputs based on which Tensors are updated using gradients.
        """

        # Only annotate Tensors which actually contribute to the model calculation. 
        # Contributing to the model calculation means that the tensor is involved
        # in operators that include GRADIENT or PARAMETER tensors as well.
        model_relevant = {Category.GRADIENT, Category.PARAMETER}
        produces_gradient: Set[TensorKeyAndVersion] = set()
        for node in reversed(self._data_flow_graph.flow_nodes):
            tensors = {(key, version) for key, (_, version) in node.inputs.items()}
            tensors |= node.outputs.items()

            produced = False
            for tensor in tensors:
                if self._categories.get(*tensor) in model_relevant or tensor in produces_gradient:
                    produced = True
                    break
            if produced:
                produces_gradient |= tensors

        # Don't include Tensors created in the backward propagation.
        input_candidates = produces_gradient.copy()
        for node in self._data_flow_graph.flow_nodes:
            if self._is_backward(node._event):
                input_candidates -= set(node.outputs.items())

        # To distinct inputs with parts of the model, eliminate the Tensors which depend on gradient.
        depends_on_gradient = self._any_version_depends_on_gradient()
        for key, version in input_candidates:
            if key.id not in depends_on_gradient:
                self._categories.setdefault_by_version(key, version, Category.INPUT)

    def _set_parameters_using_data_flow(self) -> None:
        """Deduce which Tensors are parameters."""
        snapshot = self._category_snapshot()

        candidate_parameters: Set[TensorKeyAndVersion] = set()
        candidate_fwd_tensors: Set[TensorKeyAndVersion] = {i for i, category in snapshot.items()
                                                           if category == Category.INPUT}
        for node in self._data_flow_graph.flow_nodes:
            inputs = {(key, version) for key, (_, version) in node.inputs.items()}
            inputs_and_outputs = inputs | set(node.outputs.items())
            if (
                # Don't check nodes in the backward propagation
                not self._is_backward(node._event)
                # and don't check node produce gradient
                and not any(self._is_gradient(*i) for i in inputs_and_outputs)
                # and only check nodes which depend on an input
                and candidate_fwd_tensors.intersection(inputs)
            ):
                candidate_fwd_tensors |= node.outputs.items()
                candidate_parameters |= inputs.difference(candidate_fwd_tensors)

        # Require that each parameter eventually contributes to the value of a gradient
        used_for_gradient: Set[TensorKeyAndVersion] = set()
        for node in reversed(self._data_flow_graph.flow_nodes):
            if any(self._is_gradient(*output) or output in used_for_gradient for output in node.outputs.items()):
                used_for_gradient.update((key, version) for key, (_, version) in node.inputs.items())
        candidate_parameters.intersection_update(used_for_gradient)

        # and depends on a gradient
        parameter_keys = {key.id for key, _ in candidate_parameters}
        parameter_keys &= self._any_version_depends_on_gradient()

        for key, _ in snapshot.keys():
            if key.id in parameter_keys:
                self._categories.set_by_id(key, Category.PARAMETER)

    def _set_activations(self) -> None:
        """
        Flood the graph to identify activations.
        """
        required = {Category.INPUT, Category.ACTIVATION}
        also_allowed = {Category.PARAMETER, Category.TEMPORARY}
        for node in self._data_flow_graph.flow_nodes:
            inputs = {(key, version) for key, (_, version) in node.inputs.items()}
            input_categories = {self._categories.get(*input) for input in inputs}
            if (
                # Inputs must include INPUT or ACTIVATION
                (input_categories & required)
                # and inputs only include INPUT, ACTIVATION, PARAMETER and TEMPORARY.
                and not (input_categories - (required | also_allowed))
                # and the node is not in the backward propagation
                and not self._is_backward(node._event)
            ):
                for output in node.outputs.items():
                    self._categories.setdefault_by_version(*output, Category.ACTIVATION)

    def _set_optimizer_state(self) -> None:
        for event in traverse_dfs(self._root_nodes):
            if event.tag != _EventType.PyCall or event.extra_fields.optimizer_parameters is None:
                continue
            
            # Directly set OPTIMIZER_STATE in optimizer parameters.
            parameters = event.extra_fields.optimizer_parameters
            for _, tensor in it.chain(*[param.state for param in parameters]):
                key = TensorKey.from_tensor(tensor)
                if key is not None:
                    self._categories.set_by_id(key, Category.OPTIMIZER_STATE)

    def _set_autograd_detail(self) -> None:
        prior = {None, Category.AUTOGRAD_DETAIL}
        for node in self._data_flow_graph.flow_nodes:
            if not self._is_backward(node._event):
                continue
            
            # Directly set AUTOGRAD_DETAIL in the backward propagation.
            for key, version in node.outputs.items():
                if version == 0 or self._categories.get(key, version - 1) in prior:
                    self._categories.setdefault_by_version(key, version, Category.AUTOGRAD_DETAIL)


class MemoryProfileTimeline:
    def __init__(self, memory_profile: MemoryProfile):
        """
        The timeline contains elements of the form [timestamp, action, (TensorKey, version),
        numbytes], representing actions (pre-existing, create, destroy, or increment_version)
        on a memory chunk associated with a specific Tensor (identified by (TensorKey, version)).
        Categories map each (TensorKey, version) pair to a category, and the memory history
        tracks peak memory usage.
        """
        self.timeline = memory_profile.timeline
        self.categories = memory_profile._categories
        self.memory_history = memory_profile.memory_history
    
    @staticmethod
    def _parse_device_info(device_str: str) -> Optional[DeviceKey]:
        # If the device is "cpu".
        if device_str == "cpu":
            return DeviceKey(_DEVICE_DICT.get(device_str), -1)
        
        # If the device is "npu:0".
        device_str_list = device_str.strip().split(":")
        if len(device_str_list) != 2:
            print_error_msg(f"{device_str} is not in a valid format.")
            return None
        
        device_type = _DEVICE_DICT.get(device_str_list[0])
        if device_type is None:
            print_error_msg(f"{device_str} is not in a valid format.")
            return None
        
        try:
            device_index = int(device_str_list[1])
            return DeviceKey(device_type, device_index)
        except ValueError:
            return None

    def _update_sizes_by_category(self, sizes_by_category, key, version, delta):
        category = self.categories.get(key, version) if isinstance(key, TensorKey) else None
        category_index = _CATEGORY_TO_INDEX[category] + 1
        sizes_by_category[-1][category_index] += int(delta)

    def _get_category_index(self, key, version) -> int:
        category = self.categories.get(key, version) if isinstance(key, TensorKey) else None
        return _CATEGORY_TO_INDEX[category]
    
    def _construct_timeline(self, device_str: str) -> Tuple[List[int], List[List[int]]]:
        """
        For each timestamp in the `timesstamps`, compute the storage size for each category
        and store the results in the `sizes`. The `sizes_by_category` will have the same
        length as the `timestamps`, with each entry corresponding to a timestamp in timestamps.
        """
        timestamps: List[int] = []
        sizes_by_category: List[List[int]] = []

        device = self._parse_device_info(device_str)
        if device is None:
            return timestamps, sizes_by_category

        ts_min = -1
        for ts, action, (key, version), numbytes in self.timeline:
            if key.device_type != device.device_type or key.device_index != device.device_index:
                continue

            # Convert timestamps from ns to us.
            if ts != -1:
                ts = int(ts / Constant.NS_TO_US)
            
            # Save the smallest timestamp as the timestemp of pre-existing allocations.
            if ts_min == -1 or (ts < ts_min and ts > 0):
                ts_min = ts
            
            # Initialize the memory usage of the first timestamp.
            if len(timestamps) == 0:
                timestamps.append(ts)
                sizes_by_category.append([0 for _ in _CATEGORY_TO_INDEX] + [0])
            # Copy the memory usage of the last timestamp if the current timestamp
            # is later than the last timestamp.
            elif ts != timestamps[-1]:
                timestamps.append(ts)
                sizes_by_category.append(sizes_by_category[-1].copy())

            # For this timestamp, calculate storage size of each category according to action.
            if action in (Action.PREEXISTING, Action.CREATE):
                self._update_sizes_by_category(sizes_by_category, key, version, numbytes)
            elif action == Action.INCREMENT_VERSION:
                self._update_sizes_by_category(sizes_by_category, key, version, -numbytes)
                self._update_sizes_by_category(sizes_by_category, key, version + 1, numbytes)
            elif action == Action.DESTROY:
                self._update_sizes_by_category(sizes_by_category, key, version, -numbytes)

        timestamps = [ts_min if t < 0 else t for t in timestamps]
        return timestamps, sizes_by_category
    
    @staticmethod
    def _draw_memory_timeline(timestamps: List[int], stacked: List[List[int]], 
                             max_memory_allocated: int, max_memory_reserved: int) -> Optional[str]:
        # Import matplotlib.
        module_name = "matplotlib.pyplot"
        try:
            plt = importlib.import_module(module_name)
        except ModuleNotFoundError:
            print_error_msg(f"{module_name} was not found.")
            return None
        
        # Plot memory timeline as stacked data
        fig = plt.figure(figsize=(20, 12), dpi=80)
        axes = fig.gca()
        for category, color in _CATEGORY_TO_COLORS.items():
            idx = _CATEGORY_TO_INDEX[category]
            axes.fill_between(timestamps / Constant.US_TO_MS, stacked[:, idx], stacked[:, idx + 1],
                              color=color, alpha=0.7)
        fig.legend(["Unknown" if category is None else category.name for category in _CATEGORY_TO_COLORS])
        axes.set_xlabel("Time (ms)")
        axes.set_ylabel("Memory (GB)")
        title = "\n\n".join([f"Max memory allocated: {max_memory_allocated / Constant.B_TO_GB:.2f} GB \n"
                             f"Max memory reserved: {max_memory_reserved / Constant.B_TO_GB:.2f} GB"])
        axes.set_title(title)

        # Embed the memory timeline image into the HTML file
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        encoded = b64encode(buffer.read()).decode("utf-8")
        html = f"""<html>
<head><meta charset="utf-8" /><title>Memory Timeline HTML</title></head>
<body>
  <img src='data:image/png;base64,{encoded}'>
</body>
</html>"""
        return html

    def export_memory_timeline_json(self, output_path: str, device_str: str) -> None:
        """
        Saves `times` and `sizes` as a json file. `times` is a list of timestamp
        sorted in ascending order. For each timestamp, there is the memory usage
        of each category in `sizes` list at the time of the current timestamp.
        """
        # Get memory timeline data in specified device.
        timestamps, sizes_by_category = self._construct_timeline(device_str)
        if not timestamps:
            print_error_msg("No memory timeline data.")
            return
        
        realpath = ProfilerPathManager.get_realpath(output_path)
        if output_path.endswith(".gz"):
            FileManager.create_json_gz_file_by_path(realpath, [timestamps, sizes_by_category])
        else:
            FileManager.create_json_file_by_path(realpath, [timestamps, sizes_by_category])
    
    def export_memory_timeline_json_raw(self, output_path: str, device_str: str) -> None:
        """
        Saves raw memory events in a compressed json file. Each event consists of
        (timestamp, action, memory bytes, category).
        """
        device = self._parse_device_info(device_str)
        if device is None:
            return
        
        raw_events: List[Tuple[int, int, int, int]] = []
        for ts, action, (key, version), numbytes in self.timeline:
            if key.device_type != device.device_type or key.device_index != device.device_index:
                continue

            if action in (Action.PREEXISTING, Action.CREATE):
                raw_events.append((ts, _ACTION_TO_INDEX[action], numbytes, self._get_category_index(key, version)))
            elif action == Action.INCREMENT_VERSION:
                raw_events.append((ts, _ACTION_TO_INDEX[action], -numbytes, self._get_category_index(key, version)))
                raw_events.append((ts, _ACTION_TO_INDEX[action], numbytes, self._get_category_index(key, version + 1)))
            elif action == Action.DESTROY:
                raw_events.append((ts, _ACTION_TO_INDEX[action], -numbytes, self._get_category_index(key, version)))
        
        if not raw_events:
            print_error_msg("No memory timeline data.")
            return
        
        realpath = ProfilerPathManager.get_realpath(output_path)
        FileManager.create_json_gz_file_by_path(realpath, raw_events)

    def export_memory_timeline_html(self, output_path: str, device_str: str) -> None:
        """
        Stores the memory timeline plot as PNG format in an HTML file.
        """
        # Get memory timeline data.
        timestamps, sizes_by_category = self._construct_timeline(device_str)
        if not timestamps:
            print_error_msg("No memory timeline data.")
            return
        
        timestamps = np.array(timestamps)
        sizes_by_category = np.array(sizes_by_category)
        
        ts_min = min(timestamps)
        timestamps -= ts_min                                                # For this timeline, start at 0.
        stacked = np.cumsum(sizes_by_category, axis=1) / Constant.B_TO_GB   # Convert from B to GB.
        
        # Find max allocated size and max reserved size from memory history.
        device = self._parse_device_info(device_str)
        max_memory_allocated = max((allocated for key, _, allocated, _ in self.memory_history
                                    if key == device), default=0)
        max_memory_reserved = max((reserved for key, _, _, reserved in self.memory_history
                                   if key == device), default=0)

        html = self._draw_memory_timeline(timestamps, stacked, max_memory_allocated, max_memory_reserved)
        if html is None:
            return
        FileManager.create_text_file_by_path(ProfilerPathManager.get_realpath(output_path), html)


class MemoryTimelineParser(BaseParser):
    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._device = self._param_dict.get("device")
        ProfilerLogger.init(self._profiler_path, "MemoryTimelineParser")
        self.logger = ProfilerLogger.get_instance()

    def run(self, deps_data: dict):
        try:
            mem_profile = MemoryProfile(self._profiler_path)

            # Depending on the file suffix, save the data as html, json or raw.json.gz.
            mem_timeline = MemoryProfileTimeline(mem_profile)
            if self._output_path.endswith(".html"):
                mem_timeline.export_memory_timeline_html(self._output_path, self._device)
            elif self._output_path.endswith("raw.json.gz"):
                mem_timeline.export_memory_timeline_json_raw(self._output_path, self._device)
            else:
                mem_timeline.export_memory_timeline_json(self._output_path, self._device)
        except Exception as e:
            self.logger.error("Failed to generate %s, error: %s", self._output_path, str(e), exc_info=True)
            return Constant.FAIL, None
        return Constant.SUCCESS, None