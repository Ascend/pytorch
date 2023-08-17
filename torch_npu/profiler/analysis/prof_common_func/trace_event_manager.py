from .constant import Constant


class TraceEventManager:
    @classmethod
    def create_x_event(cls, event: any, cat: str) -> dict:
        return {"ph": "X", "name": event.name, "pid": event.pid, "tid": event.tid, "ts": event.ts,
                "dur": event.dur, "cat": cat, "args": event.args}

    @classmethod
    def create_m_event(cls, pid: int, tid_dict: dict) -> list:
        event_list = [
            {"ph": "M", "name": Constant.PROCESS_NAME, "pid": pid, "tid": 0, "args": {"name": "Python"}},
            {"ph": "M", "name": Constant.PROCESS_LABEL, "pid": pid, "tid": 0, "args": {"labels": "CPU"}},
            {"ph": "M", "name": Constant.PROCESS_SORT, "pid": pid, "tid": 0, "args": {"sort_index": 0}}
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
                 "tid": start_event.tid, "ts": start_event.ts, "cat": "async_npu"},
                {"ph": "f", "bp": "e", "name": "torch_to_npu", "id": flow_id, "pid": end_event.pid,
                 "tid": end_event.tid, "ts": end_event.ts, "cat": "async_npu"}]

    @classmethod
    def create_task_queue_flow(cls, ph: str, event: any) -> dict:
        return {"ph": ph, "bp": "e", "name": "enqueue_to_dequeue", "id": event.corr_id, "pid": event.pid,
                "tid": event.tid, "ts": event.ts, "cat": "async_task_queue"}
