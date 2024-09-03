import os
import re
from collections import defaultdict

from ..prof_bean._torch_op_bean import TorchOpBean
from ..prof_common_func._binary_decoder import BinaryDecoder
from ..prof_common_func._constant import Constant, DbConstant, contact_2num, print_warn_msg
from ..prof_common_func._file_manager import FileManager
from ..prof_common_func._file_tag import FileTag
from ..prof_common_func._path_manager import ProfilerPathManager
from ..prof_common_func._tlv_decoder import TLVDecoder
from ..prof_common_func._trace_event_manager import TraceEventManager
from ..prof_common_func._tree_builder import TreeBuilder
from ..prof_config._fwk_file_parser_config import FwkFileParserConfig
from ._python_trace_parser import PythonTraceParser


__all__ = []


class FwkFileParser:
    def __init__(self, profiler_path: str):
        self._fwk_path = ProfilerPathManager.get_fwk_path(profiler_path)
        self._profiler_path = profiler_path
        self._file_list = {}
        self._file_dispatch()

    def get_file_data_by_tag(self, file_tag: int) -> list:
        file_path = self._file_list.get(file_tag)
        if not file_path:
            return []
        all_bytes = FileManager.file_read_all(file_path, "rb")
        file_bean = FwkFileParserConfig.FILE_BEAN_MAP.get(file_tag, {}).get("bean")
        is_tlv = FwkFileParserConfig.FILE_BEAN_MAP.get(file_tag, {}).get("is_tlv")
        struct_size = FwkFileParserConfig.FILE_BEAN_MAP.get(file_tag, {}).get("struct_size")
        if is_tlv:
            return TLVDecoder.decode(all_bytes, file_bean, struct_size)
        else:
            return BinaryDecoder.decode(all_bytes, file_bean, struct_size)

    def get_enqueue_data(self) -> list:
        enqueue_data_list = []
        op_mark_data = self.get_file_data_by_tag(FileTag.OP_MARK)
        if not op_mark_data:
            return enqueue_data_list
        op_mark_data.sort(key=lambda x: x.time_ns)
        tid_op_dict = defaultdict(lambda: defaultdict(list))
        for op_mark in op_mark_data:
            if not op_mark.is_enqueue:
                continue
            if op_mark.is_enqueue_start:
                tid_op_dict[op_mark.tid][op_mark.origin_name].append(op_mark)
                continue
            start_op_list = tid_op_dict.get(op_mark.tid, {}).get(op_mark.origin_name, [])
            if not start_op_list:
                print_warn_msg("Enquque data match failed")
                continue
            start_op = start_op_list.pop()
            op_mark.ts = start_op.time_ns
            op_mark.dur = op_mark.time_ns - start_op.time_ns
            enqueue_data_list.append(op_mark)
            start_op_list.clear()
        return enqueue_data_list

    def get_dequeue_data(self) -> list:
        dequeue_data_list = []
        op_mark_data = self.get_file_data_by_tag(FileTag.OP_MARK)
        if not op_mark_data:
            return dequeue_data_list
        op_mark_data.sort(key=lambda x: x.time_ns)
        tid_op_dict = defaultdict(lambda: defaultdict(list))
        for op_mark in op_mark_data:
            if not op_mark.is_dequeue:
                continue
            if op_mark.is_dequeue_start:
                tid_op_dict[op_mark.tid][op_mark.origin_name].append(op_mark)
                continue
            start_op_list = tid_op_dict.get(op_mark.tid, {}).get(op_mark.origin_name, [])
            if not start_op_list:
                print_warn_msg("Dequque data match failed")
                continue
            start_op = start_op_list.pop()
            op_mark.ts = start_op.time_ns
            op_mark.dur = op_mark.time_ns - start_op.time_ns
            dequeue_data_list.append(op_mark)
            start_op_list.clear()
        return dequeue_data_list

    def get_task_queue_data(self) -> any:
        enqueue_data_list, dequeue_data_list = [], []
        op_mark_data = self.get_file_data_by_tag(FileTag.OP_MARK)
        if not op_mark_data:
            return [], []
        op_mark_data.sort(key=lambda x: x.time_ns)
        enqueue_tid_op_dict = defaultdict(lambda: defaultdict(list))
        dequeue_tid_op_dict = defaultdict(lambda: defaultdict(list))
        for op_mark in op_mark_data:
            if op_mark.is_enqueue_start:
                enqueue_tid_op_dict[op_mark.tid][op_mark.origin_name].append(op_mark)
                continue
            if op_mark.is_dequeue_start:
                dequeue_tid_op_dict[op_mark.tid][op_mark.origin_name].append(op_mark)
                continue
            if op_mark.is_enqueue_end:
                start_op_list = enqueue_tid_op_dict.get(op_mark.tid, {}).get(op_mark.origin_name, [])
                if not start_op_list:
                    print_warn_msg("Enquque data match failed")
                    continue
                start_op = start_op_list.pop()
                op_mark.ts = start_op.time_ns
                op_mark.dur = op_mark.time_ns - start_op.time_ns
                enqueue_data_list.append(op_mark)
                start_op_list.clear()
                continue
            if op_mark.is_dequeue_end:
                start_op_list = dequeue_tid_op_dict.get(op_mark.tid, {}).get(op_mark.origin_name, [])
                if not start_op_list:
                    print_warn_msg("Dequque data match failed")
                    continue
                start_op = start_op_list.pop()
                op_mark.ts = start_op.time_ns
                op_mark.dur = op_mark.time_ns - start_op.time_ns
                dequeue_data_list.append(op_mark)
                start_op_list.clear()
        return enqueue_data_list, dequeue_data_list

    def get_torch_op_tree_node(self, only_fwk: bool = False) -> list:
        torch_op_list = self.get_file_data_by_tag(FileTag.TORCH_OP)
        if not torch_op_list:
            return []
        enqueue_data_list = []
        if not only_fwk:
            enqueue_data_list = self.get_enqueue_data()
        result_data = TreeBuilder.build_tree(torch_op_list, enqueue_data_list)
        return result_data

    def get_fwk_trace_data(self):
        torch_op_data = self.get_file_data_by_tag(FileTag.TORCH_OP)
        if not torch_op_data:
            return []
        enqueue_data_list, dequeue_data_list = self.get_task_queue_data()
        pid = torch_op_data[0].pid
        tid_dict = {}
        fwk_x_event_list = [None] * (
                len(torch_op_data) + len(enqueue_data_list) * 2 + len(dequeue_data_list) * 2)
        index = 0
        fwd_dict = {}
        for torch_op in torch_op_data:
            self.filter_fwd_bwd_event(fwd_dict, torch_op)
            tid_dict[torch_op.tid] = False
            fwk_x_event_list[index] = TraceEventManager.create_x_event(torch_op, "cpu_op")
            index += 1
        for enqueue_data in enqueue_data_list:
            tid_dict[enqueue_data.tid] = False
            fwk_x_event_list[index] = TraceEventManager.create_x_event(enqueue_data, "enqueue")
            index += 1
            fwk_x_event_list[index] = TraceEventManager.create_task_queue_flow(Constant.FLOW_START_PH, enqueue_data)
            index += 1
        for dequeue_data in dequeue_data_list:
            tid_dict[dequeue_data.tid] = True
            fwk_x_event_list[index] = TraceEventManager.create_x_event(dequeue_data, "dequeue")
            index += 1
            fwk_x_event_list[index] = TraceEventManager.create_task_queue_flow(Constant.FLOW_END_PH, dequeue_data)
            index += 1
        other_event_list = TraceEventManager.create_m_event(pid, tid_dict)
        other_event_list.extend(TraceEventManager.create_fwd_flow(fwd_dict))
        fwk_x_event_list.extend(other_event_list)
        python_trace_data = self.get_python_trace_data()
        if python_trace_data:
            fwk_x_event_list.extend(python_trace_data)
        gc_record_data = self.get_gc_record_trace_data()
        if gc_record_data:
            fwk_x_event_list.extend(gc_record_data)
        return fwk_x_event_list

    def get_python_trace_data(self) -> list:
        trace_hash_data = self.get_file_data_by_tag(FileTag.PYTHON_TRACER_HASH)
        func_call_data = self.get_file_data_by_tag(FileTag.PYTHON_TRACER_FUNC)
        python_trace_parser = PythonTraceParser(trace_hash_data, func_call_data)
        return python_trace_parser.get_python_trace_data()

    @classmethod
    def filter_fwd_bwd_event(cls, fwd_dict: dict, torch_op: TorchOpBean):
        seq_num = torch_op.args.get("Sequence number", -1)
        if seq_num < 0:
            return
        fwd_event = fwd_dict.get(seq_num, {})
        mode = "start" if torch_op.args.get("Fwd thread id") == 0 else "end"
        if fwd_event.get(mode, {}).get("ts", -float('inf')) < torch_op.ts:
            node = {mode: {'pid': torch_op.pid, 'tid': torch_op.tid, 'ts': torch_op.ts}}
            fwd_dict.setdefault(seq_num, {}).update(node)

    def has_task_queue_data(self):
        return bool(self._file_list.get(FileTag.OP_MARK))

    def _file_dispatch(self):
        for file_name in os.listdir(os.path.realpath(self._fwk_path)):
            file_path = os.path.join(self._fwk_path, file_name)
            if not os.path.isfile(file_path):
                continue
            for file_tag, pattern in FwkFileParserConfig.FILE_DISPATCH_MAP.items():
                if re.match(pattern, file_name):
                    self._file_list.setdefault(file_tag, file_path)

    def filter_fwd_bwd_api(self, fwd_bwd_dict: dict, torch_op: TorchOpBean, torch_op_idx: int):
        seq_num = torch_op.args.get("Sequence number", -1)
        if seq_num < 0:
            return
        fwd_event = fwd_bwd_dict.get(seq_num, {})
        mode = "start" if torch_op.args.get("Fwd thread id") == 0 else "end"
        if fwd_event.get(mode, {}).get("ts", -float('inf')) < torch_op.ts:
            node = {mode: {'idx': torch_op_idx}}
            fwd_bwd_dict.setdefault(seq_num, {}).update(node)

    def update_fwd_bwd_connection_id(self, fwd_dict: dict, torch_op_apis: list, start_connection_id: int):
        nodes = fwd_dict.values()
        for node in nodes:
            if node.get('start') and node.get('end'):
                fwb_op_id = node['start']['idx']
                bwd_op_id = node['end']['idx']
                torch_op_apis[fwb_op_id][3].append(start_connection_id)
                torch_op_apis[bwd_op_id][3].append(start_connection_id)
                
                start_connection_id += 1

    def get_fwk_api(self) -> dict:
        torch_op_data = self.get_file_data_by_tag(FileTag.TORCH_OP)
        if not torch_op_data:
            return {}
        pid = torch_op_data[0].pid
        torch_op_apis = []
        fwd_bwd_dict = {}
        torch_op_idx = 0
        mstx_mark_apis = []

        for torch_op in torch_op_data:
            api = [torch_op.ts, torch_op.end_ns, contact_2num(pid, torch_op.tid), [], torch_op.name,
                   torch_op.args.get(Constant.SEQUENCE_NUMBER, -1), torch_op.args.get(Constant.FORWARD_THREAD_ID),
                   torch_op.args.get(Constant.INPUT_SHAPES), torch_op.args.get(Constant.INPUT_DTYPES), torch_op.call_stack]
            if torch_op.name == "mstx_mark_op":
                mstx_mark_apis.append(api)
            else:
                torch_op_apis.append(api)
                self.filter_fwd_bwd_api(fwd_bwd_dict, torch_op, torch_op_idx)
                torch_op_idx += 1

        connection_ids = []
        task_enqueues = []
        task_dequeues = []
        enqueue_data_list, dequeue_data_list = self.get_task_queue_data()
        for enqueue_data in enqueue_data_list:
            task_enqueues.append(
                [enqueue_data.ts, enqueue_data.ts + enqueue_data.dur, contact_2num(pid, enqueue_data.tid),
                 enqueue_data.corr_id, enqueue_data.name])
            connection_ids.append(enqueue_data.corr_id)
        for dequeue_data in dequeue_data_list:
            task_dequeues.append(
                [dequeue_data.ts, dequeue_data.ts + dequeue_data.dur, contact_2num(pid, dequeue_data.tid),
                 dequeue_data.corr_id, dequeue_data.name])
        
        start_connection_id = max(connection_ids) + 1 if connection_ids else 0
        self.update_fwd_bwd_connection_id(fwd_bwd_dict, torch_op_apis, start_connection_id)

        trace_hash_data = self.get_file_data_by_tag(FileTag.PYTHON_TRACER_HASH)
        func_call_data = self.get_file_data_by_tag(FileTag.PYTHON_TRACER_FUNC)
        python_trace_parser = PythonTraceParser(trace_hash_data, func_call_data)
        python_trace_apis = python_trace_parser.get_python_trace_api_data()
        return {"torch_op": torch_op_apis, "task_enqueues": task_enqueues, "task_dequeues": task_dequeues,
                "python_trace": python_trace_apis, "mstx_op": mstx_mark_apis}

    def get_first_fwk_op(self):
        torch_op_data = self.get_file_data_by_tag(FileTag.TORCH_OP)
        if not torch_op_data:
            return None
        return min(torch_op_data, key=lambda op: op.ts)

    def get_gc_record_db_data(self):
        gc_events = self.get_file_data_by_tag(FileTag.GC_RECORD)
        if not gc_events:
            return []
        return [[event.ts, event.ts + event.dur, contact_2num(event.pid, event.tid)] for event in gc_events]

    def get_gc_record_trace_data(self):
        gc_events = self.get_file_data_by_tag(FileTag.GC_RECORD)
        if not gc_events:
            return []
        cann_path = ProfilerPathManager.get_cann_path(self._profiler_path)
        device_id = ProfilerPathManager.get_device_id(cann_path)
        event_list = [None] * len(gc_events)
        for idx, event in enumerate(gc_events):
            event.pid = TraceEventManager.get_pid_format(event.pid, TraceEventManager.GC_SORT_INDEX, device_id)
            event_list[idx] = TraceEventManager.create_x_event(event, "GC")
        event_list.extend(TraceEventManager.create_gc_m_event(gc_events[0].pid, gc_events[0].tid))
        return event_list
