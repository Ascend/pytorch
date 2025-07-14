import copy
from collections import defaultdict
from enum import Enum
from ._base_parser import BaseParser
from ..prof_common_func._constant import Constant
from ..prof_common_func._file_manager import FileManager
from ..prof_common_func._constant import convert_ns2us_float
from ..prof_common_func._log import ProfilerLogger
from ..prof_parse._cann_file_parser import CANNFileParser
from ..prof_parse._fwk_cann_relation_parser import FwkCANNRelationParser
from ..prof_parse._fwk_file_parser import FwkFileParser

__all__ = []


def default_time():
    return {
        'compute': 0,
        'comunNotOverlp': 0,
        'Overlp': 0,
        'comun': 0,
        'free': 0,
        'stage': 0,
        'bubble': 0,
        'comunNotOverlpRec': 0,
        'prepare': 0
    }


def step_time_dict():
    return defaultdict(default_time)


class _StepInfoIndex(Enum):
    ID = 0
    START_TS = 1
    END_TS = 2
    E2E_START_TS = 3
    E2E_END_TS = 4
    FWK_START_TS = 5
    FIRST_TASK_TS = 6


class TraceStepTimeParser(BaseParser):
    STEP_TRACE = "step_trace_time.csv"
    timeflag = {'Communication': 'comun', 'Computing': 'compute', 'Free': 'free',
                'Communication(Not Overlapped)': 'comunNotOverlp', 'hcom_receive': 'bubble'}
    title = ['Device_id', 'Step', 'Computing', 'Communication(Not Overlapped)', 'Overlapped', 'Communication',
             'Free', 'Stage', 'Bubble', 'Communication(Not Overlapped and Exclude Receive)', 'Preparing']

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self.step_range = []

    @classmethod
    def is_float_num(cls, num):
        try:
            float(num)
            return True
        except ValueError:
            return False

    @classmethod
    def count_time(cls, add_type, data, step_list, save_time, pid_device_map):
        start_time = data.get('ts', 0)
        duration = data.get('dur', 0)
        device_id = pid_device_map[data['pid']]
        cur_step = None
        if not cls.is_float_num(start_time) or not cls.is_float_num(duration):
            print('Ts or dur format error!')
            return
        start_time = float(start_time)
        duration = float(duration)
        for step in step_list.get(device_id, []):
            if step[_StepInfoIndex.START_TS.value] <= start_time < step[_StepInfoIndex.END_TS.value]:
                cur_step = step[_StepInfoIndex.ID.value]
                break
        for step in step_list.get(device_id, []):
            if cur_step == step[_StepInfoIndex.ID.value]:
                if start_time < step[_StepInfoIndex.E2E_START_TS.value] or \
                    step[_StepInfoIndex.E2E_START_TS.value] == -1:
                    step[_StepInfoIndex.E2E_START_TS.value] = start_time
                if start_time + duration > step[_StepInfoIndex.E2E_END_TS.value] or \
                    step[_StepInfoIndex.E2E_END_TS.value] == -1:
                    step[_StepInfoIndex.E2E_END_TS.value] = start_time + duration
                if add_type in {'Communication', 'Computing'}:
                    if start_time < step[_StepInfoIndex.FIRST_TASK_TS.value] or \
                       step[_StepInfoIndex.FIRST_TASK_TS.value] == -1:
                        step[_StepInfoIndex.FIRST_TASK_TS.value] = start_time
                break
        save_time[device_id][cur_step][cls.timeflag.get(add_type)] += duration

    @classmethod
    def get_e2e_time(cls, step, step_list):
        for cur_step in step_list:
            if cur_step[_StepInfoIndex.ID.value] == step:
                return cur_step[_StepInfoIndex.E2E_END_TS.value] - cur_step[_StepInfoIndex.E2E_START_TS.value]
        return 0

    def get_prepare_time(self, step, step_list):
        for cur_step in step_list:
            if cur_step[_StepInfoIndex.ID.value] == step:
                first_task_start_ts = cur_step[_StepInfoIndex.FIRST_TASK_TS.value]
                if step is None:
                    first_fwk_op = FwkFileParser(self._profiler_path).get_first_fwk_op()
                    return (first_task_start_ts - convert_ns2us_float(first_fwk_op.ts)) if first_fwk_op else 0
                return first_task_start_ts - cur_step[_StepInfoIndex.FWK_START_TS.value]
        return 0

    def create_step_file(self, output_path: str, json_str: list, file_name: str) -> None:
        step_list = []
        save_time = defaultdict(step_time_dict)
        if not json_str:
            return
        # obtain the mapping between pid and device_id(rank_id)
        pid_device_map = {}
        for data in json_str:
            if data.get('name') == 'process_labels' and data.get('args', {}).get('labels', '').startswith('NPU'):
                label = data['args']['labels']
                pid_device_map[data.get('pid')] = -1 if label == 'NPU' else int(label.split(' ')[1]) # "labels": "NPU 0"
        # get initial step time
        for cur_step in self.step_range:
            step_list.append(
                [cur_step.get(Constant.STEP_ID), convert_ns2us_float(cur_step.get(Constant.START_TS)),
                 convert_ns2us_float(cur_step.get(Constant.END_TS)), -1, -1,
                 convert_ns2us_float(cur_step.get(Constant.FWK_START_TS)), -1])
        if not self.step_range:
            step_list.append([None, -1, -1, -1, -1, -1, -1])
        # every device should have its own step_list
        step_dict = {}
        for device in set(pid_device_map.values()):
            step_dict[device] = copy.deepcopy(step_list)
        has_analysis_data_flag = False
        bubble_data = []
        # traverse json and calculate time
        for data in json_str:
            if data.get('name') in {'Communication', 'Computing', 'Free', 'Communication(Not Overlapped)'}:
                self.count_time(data.get('name'), data, step_dict, save_time, pid_device_map)
                has_analysis_data_flag = True
            elif str(data.get('name')).startswith('hcom_receive'):
                bubble_data.append(data)
                self.count_time('hcom_receive', data, step_dict, save_time, pid_device_map)
        if not has_analysis_data_flag:
            return
        print_time = []
        for device, device_time in save_time.items():
            for step, step_time in device_time.items():
                if self.step_range and step is None:
                    continue
                step_time['comunNotOverlpRec'] = step_time['comunNotOverlp'] - step_time['bubble']
                step_time['Overlp'] = step_time['comun'] - step_time['comunNotOverlp']
                step_time['stage'] = self.get_e2e_time(step, step_dict.get(device, [])) - step_time['bubble']
                step_time['prepare'] = self.get_prepare_time(step, step_dict.get(device, []))
                print_time.append(
                    [device, step, step_time['compute'], step_time['comunNotOverlp'], step_time['Overlp'],
                     step_time['comun'], step_time['free'], step_time['stage'], step_time['bubble'],
                     step_time['comunNotOverlpRec'], step_time['prepare']])
        if print_time:
            if self.step_range:
                print_time.sort(key=lambda x: (x[0], int(x[1])))  # step is a string
            else:
                print_time.sort(key=lambda x: x[0])  # step is None
        FileManager.create_csv_file(output_path, print_time, file_name, self.title)

    def run(self, deps_data: dict):
        ProfilerLogger.init(self._profiler_path, "TraceStepTimeParser")
        self.logger = ProfilerLogger.get_instance()
        try:
            self._init_step_range(deps_data)
            self.generate_view()
        except Exception as e:
            self.logger.error("Failed to generate step_trace_time.csv, error: %s", str(e), exc_info=True)
            return Constant.FAIL, None
        return Constant.SUCCESS, None

    def generate_view(self) -> None:
        trace_data = CANNFileParser(self._profiler_path).get_timeline_all_data()
        self.create_step_file(self._output_path, trace_data, self.STEP_TRACE)

    def _init_step_range(self, deps_data: dict):
        torch_op_node = deps_data.get(Constant.TREE_BUILD_PARSER, [])
        if torch_op_node:
            self.step_range = FwkCANNRelationParser(self._profiler_path).get_step_range(torch_op_node[0], deps_data.get(
                Constant.RELATION_PARSER, {}))
