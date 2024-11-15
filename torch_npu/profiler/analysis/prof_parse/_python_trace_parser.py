from collections import defaultdict
from enum import Enum
from ..prof_common_func._constant import contact_2num
from ..prof_common_func._trace_event_manager import TraceEventManager

__all__ = []


MODULE_NAME_DELIMITER = "######"


class TraceTag(Enum): 
    kPy_Call = 0
    kPy_Return = 1
    kC_Call = 2 
    kC_Return = 3


class CallType(Enum):
    kPyCall = 0
    kPyModuleCall = 1
    kCCall = 2


class PyTraceEvent:

    def __init__(self, start_time, end_time, name, thread_id, process_id, self_id,
                 parent_id, call_type, module_id, call_idx, return_idx, params):
        self._start_time = start_time
        self._end_time = end_time
        self._name = name
        self._tid = thread_id
        self._pid = process_id
        self._self_id = self_id
        self._parent_id = parent_id
        self._call_type = call_type
        self._module_id = module_id # Only set call_type == CallType.kPyModuleCall
        self._call_idx = call_idx
        self._return_idx = return_idx
        self._params = params       # Set when call_type in [kPyModuleCall, kPyOptimizerCall]

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, end_time):
        self._end_time = end_time

    @property
    def return_idx(self):
        return self._return_idx

    @return_idx.setter
    def return_idx(self, return_idx):
        self._return_idx = return_idx

    @property
    def parent_id(self):
        return self._parent_id
    
    @parent_id.setter
    def parent_id(self, parent_id):
        self._parent_id = parent_id

    @property
    def self_id(self):
        return self._self_id

    @property
    def args(self) -> dict:
        args = {"Python id": self._self_id, "Python parent id": self._parent_id}
        if self._call_type == CallType.kPyModuleCall:
            args['Python module id'] = self._module_id
        return args

    @property
    def name(self) -> str:
        if self._call_type == CallType.kPyModuleCall:
            return "%s %s_%s" % ("nn.Module:", self._name, self._module_id)
        return self._name

    @property
    def pid(self):
        return self._pid

    @property
    def tid(self):
        return self._tid

    @property
    def ts(self):
        return self._start_time

    @property
    def dur(self):
        return self._end_time - self._start_time
    
    @property
    def params(self):
        return self._params


class ReplayFrame:

    def __init__(self, call_bean, name, event_idx, call_type, module_id, cur_id, parent_id, params):
        self.trace_event = PyTraceEvent(
                        call_bean.start_ns, # start_time
                        -1,                 # end_time, place_holder
                        name,               # name
                        call_bean.tid,      # thread_id
                        call_bean.pid,      # process_id
                        cur_id,             # self_id
                        None,               # parent_id, place_holder
                        call_type,          # call_type
                        module_id,          # module_id
                        event_idx,          # call_idx
                        0,                  # return_idx, place_hodler
                        params)             # module or optimizer parameters
        self.id = cur_id
        self.parent_id = parent_id


class PythonTraceParser:

    def __init__(self, torch_tids: set, hash_data: list, python_call_data: list, param_data: list = None):
        self._torch_tids = torch_tids
        self._hash_data = hash_data
        self._python_call_data = python_call_data
        self._param_data = param_data
        self._hash_map = {}
        self._param_map = {}

    def get_python_trace_data(self) -> list:
        trace_event_list = self._gen_python_trace_event_data()
        if not trace_event_list:
            return []
        trace_data = [None] * len(trace_event_list)
        for i, event in enumerate(trace_event_list):
            trace_data[i] = TraceEventManager.create_x_event(event, "python_function")
        return trace_data

    def get_python_trace_api_data(self) -> list:
        trace_event_list = self._gen_python_trace_event_data()
        if not trace_event_list:
            return []
        trace_api_data = [None] * len(trace_event_list)
        for i, event in enumerate(trace_event_list):
            trace_api_data[i] = [event.ts, event.ts + event.dur, contact_2num(event.pid, event.tid), event.name]
        return trace_api_data
    
    def get_pycall_data(self) -> list:
        self._gen_param_map()
        return self._gen_python_trace_event_data()

    def _group_tarce_data_by_tid(self):
        trace_data_by_tid = defaultdict(lambda: [])
        for call_bean in self._python_call_data:
            if call_bean.tid in self._torch_tids:
                trace_data_by_tid[call_bean.tid].append(call_bean)
        return trace_data_by_tid

    def _gen_python_trace_event_data(self):
        self._gen_hash_map()
        trace_event_by_tid = self._group_tarce_data_by_tid()
        trace_event_list = []
        for thread_trace_event in trace_event_by_tid.values():
            trace_event_list.extend(self._replay_stack(thread_trace_event))
        return trace_event_list

    def _replay_stack(self, thread_trace_event) -> list:
        event_idx = 0
        max_call_end_time = 0
        replay_stack = []
        replay_result = []
        module_name_counter = {}
        module_uid_map = {}
        thread_trace_event.sort(key=lambda x: x.start_ns)
        for id_counter, call_bean in enumerate(thread_trace_event):
            name = str(self._hash_map.get(call_bean.key, call_bean.key))
            params = self._param_map.get(call_bean.key)
            frame_parent_id = replay_stack[-1].id if len(replay_stack) > 0 else 0
            if call_bean.trace_tag == TraceTag.kPy_Call.value:
                if MODULE_NAME_DELIMITER in name:
                    module_name, module_id = self.get_module_info_from_value(name, module_name_counter, module_uid_map)
                    replay_stack.append(ReplayFrame(
                        call_bean, module_name, event_idx, CallType.kPyModuleCall,
                        module_id, id_counter, frame_parent_id, params))
                else:
                    replay_stack.append(ReplayFrame(
                        call_bean, name, event_idx, CallType.kPyCall,
                        0, id_counter, frame_parent_id, params))
            elif call_bean.trace_tag == TraceTag.kC_Call.value:
                replay_stack.append(ReplayFrame(
                    call_bean, name, event_idx, CallType.kCCall,
                    0, id_counter, frame_parent_id, params))
            elif call_bean.trace_tag in (TraceTag.kPy_Return.value, TraceTag.kC_Return.value):
                if len(replay_stack) > 0:
                    replay_stack[-1].trace_event.end_time = call_bean.start_ns
                    replay_stack[-1].trace_event.return_idx = event_idx
                    replay_result.append(replay_stack[-1])
                    replay_stack.pop()
            event_idx += 1
            max_call_end_time = max(max_call_end_time, call_bean.start_ns)

        while len(replay_stack) > 0:
            replay_stack[-1].trace_event.end_time = max_call_end_time
            replay_stack[-1].trace_event.return_idx = event_idx
            replay_result.append(replay_stack[-1])
            replay_stack.pop()
            event_idx += 1

        event_id_map = {}
        output = []
        for replay_frame in replay_result:
            output.append(replay_frame.trace_event)
            event_id_map[replay_frame.id] = output[-1]
        for i, replay_frame in enumerate(replay_result):
            parent_event = event_id_map.get(replay_frame.parent_id, None)
            output[i].parent_id = parent_event.self_id if parent_event is not None else 0
        return output

    def _gen_hash_map(self):
        self._hash_map = {hash_bean.key: hash_bean.value for hash_bean in self._hash_data}
    
    def _gen_param_map(self):
        if self._param_data is not None:
            self._param_map = {param_bean.key: param_bean.params for param_bean in self._param_data}
    
    @staticmethod
    def get_module_info_from_value(name: str, module_name_counter: dict, module_uid_map: dict):
        module_name, module_uid = name.split(MODULE_NAME_DELIMITER)
        if module_uid not in module_uid_map:
            cur_count = module_name_counter.get(module_name, 0)
            module_uid_map[module_uid] = cur_count
            module_name_counter[module_name] = cur_count + 1
        return (module_name, module_uid_map.get(module_uid, module_uid))
