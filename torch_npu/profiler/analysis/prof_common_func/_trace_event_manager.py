from ._constant import Constant
from ._constant import convert_ns2us_str
from ._constant import convert_ns2us_float

__all__ = []


class TraceEventManager:

    # Framework trace data sort index
    FWK_SORT_INDEX = 0
    GC_SORT_INDEX = 1

    # Pid format offset
    PID_OFFSET = 10
    INDEX_OFFSET = 5

    @classmethod
    def create_x_event(cls, event: any, cat: str) -> dict:
        return {"ph": "X", "name": event.name, "pid": event.pid, "tid": event.tid, "ts": convert_ns2us_str(event.ts),
                "dur": convert_ns2us_float(event.dur), "cat": cat, "args": event.args}

    @classmethod
    def create_m_event(cls, pid: int, tid_dict: dict) -> list:
        event_list = [
            {"ph": "M", "name": Constant.PROCESS_NAME, "pid": pid, "tid": 0, "args": {"name": "Python"}},
            {"ph": "M", "name": Constant.PROCESS_LABEL, "pid": pid, "tid": 0, "args": {"labels": "CPU"}},
            {"ph": "M", "name": Constant.PROCESS_SORT, "pid": pid, "tid": 0, "args": {"sort_index": cls.FWK_SORT_INDEX}}
        ]
        for tid, is_dequeue in tid_dict.items():
            if is_dequeue:
                sort_index = max(tid_dict.keys()) + 1
            else:
                sort_index = tid
            event_list.extend([{"ph": "M", "name": Constant.THREAD_NAME, "pid": pid, "tid": tid,
                                "args": {"name": f"Thread {tid}"}},
                               {"ph": "M", "name": Constant.THREAD_SORT, "pid": pid, "tid": tid,
                                "args": {"sort_index": sort_index}}])
        return event_list

    @classmethod
    def create_torch_to_npu_flow(cls, start_event: any, end_event: any) -> list:
        flow_id = end_event.ts
        return [{"ph": "s", "bp": "e", "name": "torch_to_npu", "id": flow_id, "pid": start_event.pid,
                 "tid": start_event.tid, "ts": convert_ns2us_str(start_event.ts), "cat": "async_npu"},
                {"ph": "f", "bp": "e", "name": "torch_to_npu", "id": flow_id, "pid": end_event.pid,
                 "tid": end_event.tid, "ts": convert_ns2us_str(end_event.ts), "cat": "async_npu"}]

    @classmethod
    def create_task_queue_flow(cls, ph: str, event: any) -> dict:
        return {"ph": ph, "bp": "e", "name": "enqueue_to_dequeue", "id": event.corr_id, "pid": event.pid,
                "tid": event.tid, "ts": convert_ns2us_str(event.ts), "cat": "async_task_queue"}

    @classmethod
    def create_fwd_flow(cls, event: any) -> list:
        fwd_list = []
        for fwd_id, node in event.items():
            if node.get('start') and node.get('end'):
                if node['start']['tid'] == node['end']['tid']:
                    continue
                flow_id = fwd_id
                fwd_list.extend([{"ph": "s", "bp": "e", "name": "fwdbwd", "id": flow_id, "pid": node['start']['pid'],
                    "tid": node['start']['tid'], "ts": convert_ns2us_str(node['start']['ts']), "cat": "fwdbwd"},
                    {"ph": "f", "bp": "e", "name": "fwdbwd", "id": flow_id, "pid": node['end']['pid'],
                    "tid": node['end']['tid'], "ts": convert_ns2us_str(node['end']['ts']), "cat": "fwdbwd"}])
        return fwd_list

    @classmethod
    def create_gc_m_event(cls, pid: int, tid: int) -> list:
        return [
            {"ph": "M", "name": Constant.PROCESS_NAME, "pid": pid, "tid": 0, "args": {"name": "Python GC"}},
            {"ph": "M", "name": Constant.PROCESS_LABEL, "pid": pid, "tid": 0, "args": {"labels": "CPU"}},
            {"ph": "M", "name": Constant.PROCESS_SORT, "pid": pid, "tid": 0, "args": {"sort_index": cls.GC_SORT_INDEX}},
            {"ph": "M", "name": Constant.THREAD_NAME, "pid": pid, "tid": tid, "args": {"name": "Python GC"}},
            {"ph": "M", "name": Constant.THREAD_SORT, "pid": pid, "tid": tid, "args": {"sort_index": tid}}
        ]

    @classmethod
    def get_pid_format(cls, pid: int, sort_index: int, device_id: int):
        return (pid << cls.PID_OFFSET) | (sort_index << cls.INDEX_OFFSET) | device_id
