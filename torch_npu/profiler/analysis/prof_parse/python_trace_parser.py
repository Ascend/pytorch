from enum import Enum
from ..prof_common_func.constant import DbConstant, contact_2num
from ..prof_common_func.trace_event_manager import TraceEventManager

__all__ = []


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

    def __init__(self, start_time, end_time, name, thread_id, process_id, 
            self_id, parent_id, call_type, module_id, call_idx, return_idx):
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

    @property
    def end_time(self, end_time):
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
            return "%s_%s" % (self._name, self._module_id)
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


class ReplayFrame:

    def __init__(self, call_bean, name, event_idx, call_type, module_id, cur_id, parent_id):
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
                        0)                  # return_idx, place_hodler
        self.id = cur_id
        self.parent_id = parent_id


class PythonTraceParser:

    def __init__(self, module_call_data: list, python_call_data: list):
        self._module_call_data = module_call_data
        self._python_call_data = python_call_data
        self._module_name_map = {}
        self._module_id_map = {}

    def get_python_trace_data(self) -> list:
        self._gen_module_call_map()
        python_trace_event_list = self._replay_stack()
        if not python_trace_event_list:
            return []
        trace_data = [None] * len(python_trace_event_list)
        for i, event in enumerate(python_trace_event_list):
            trace_data[i] = TraceEventManager.create_x_event(event, "python_function")
        return trace_data

    def get_python_trace_api_data(self) -> list:
        self._gen_module_call_map()
        python_trace_event_list = self._replay_stack()
        if not python_trace_event_list:
            return []
        trace_api_data = [None] * len(python_trace_event_list)
        for i, event in enumerate(python_trace_event_list):
            trace_api_data[i] = [event.ts, event.ts + event.dur, contact_2num(event.pid, event.tid), event.name]
        return trace_api_data

    def _replay_stack(self) -> list:
        id_counter = 0
        event_idx = 0
        max_call_end_time = 0
        replay_stack = []
        replay_result = []
        for call_bean in self._python_call_data:
            frame_parent_id = replay_stack[-1].id if len(replay_stack) > 0 else 0
            if call_bean.trace_tag == TraceTag.kPy_Call.value:
                if event_idx in self._module_name_map:
                    replay_stack.append(ReplayFrame(
                        call_bean, self._module_name_map[event_idx], event_idx, CallType.kPyModuleCall,
                        self._module_id_map.get(event_idx, 0), id_counter, frame_parent_id))
                else:
                    replay_stack.append(ReplayFrame(
                        call_bean, call_bean.name, event_idx, CallType.kPyCall,
                        0, id_counter, frame_parent_id))
            elif call_bean.trace_tag == TraceTag.kC_Call.value:
                replay_stack.append(ReplayFrame(
                    call_bean, call_bean.name, event_idx, CallType.kCCall,
                    0, id_counter, frame_parent_id))
            elif call_bean.trace_tag in (TraceTag.kPy_Return.value, TraceTag.kC_Return.value):
                if len(replay_stack) > 0:
                    replay_stack[-1].trace_event.end_time = call_bean.start_ns
                    replay_stack[-1].trace_event.return_idx = event_idx
                    replay_result.append(replay_stack[-1])
                    replay_stack.pop()
            id_counter += 1
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

    def _gen_module_call_map(self):
        self._module_call_data.sort(key=lambda bean: bean.idx)
        self._module_name_map.clear()
        self._module_id_map.clear()
        module_name_counter = {}
        module_uid_map = {}
        for call_bean in self._module_call_data:
            self._module_name_map[call_bean.idx] = call_bean.module_name
            if call_bean.module_uid not in module_uid_map:
                cur_count = module_name_counter.get(call_bean.module_name, 0)
                module_uid_map[call_bean.module_uid] = cur_count
                module_name_counter[call_bean.module_name] = cur_count + 1
            self._module_id_map[call_bean.idx] = module_uid_map.get(call_bean.module_uid, call_bean.module_uid)
