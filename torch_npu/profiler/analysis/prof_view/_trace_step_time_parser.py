from enum import Enum
from ._base_parser import BaseParser
from ..prof_common_func._constant import Constant, print_error_msg
from ..prof_common_func._file_manager import FileManager
from ..prof_common_func._constant import convert_ns2us_float
from ..prof_parse._cann_file_parser import CANNFileParser
from ..prof_parse._fwk_cann_relation_parser import FwkCANNRelationParser
from ..prof_parse._fwk_file_parser import FwkFileParser

__all__ = []


class StepInfoIndex(Enum):
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
    title = ['Step', 'Computing', 'Communication(Not Overlapped)', 'Overlapped', 'Communication', 'Free', 'Stage',
             'Bubble', 'Communication(Not Overlapped and Exclude Receive)', 'Preparing']

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
    def count_time(cls, add_type, start_time, duration, step_list, save_time):
        cur_step = None
        if not cls.is_float_num(start_time) or not cls.is_float_num(duration):
            print('Ts or dur format error!')
            return
        start_time = float(start_time)
        duration = float(duration)
        for step in step_list:
            if step[StepInfoIndex.START_TS.value] <= start_time < step[StepInfoIndex.END_TS.value]:
                cur_step = step[StepInfoIndex.ID.value]
                break
        for step in step_list:
            if cur_step == step[StepInfoIndex.ID.value]:
                if start_time < step[StepInfoIndex.E2E_START_TS.value] or \
                    step[StepInfoIndex.E2E_START_TS.value] == -1:
                    step[StepInfoIndex.E2E_START_TS.value] = start_time
                if start_time + duration > step[StepInfoIndex.E2E_END_TS.value] or \
                    step[StepInfoIndex.E2E_END_TS.value] == -1:
                    step[StepInfoIndex.E2E_END_TS.value] = start_time + duration
                if add_type in {'Communication', 'Computing'}:
                    if start_time < step[StepInfoIndex.FIRST_TASK_TS.value] or \
                       step[StepInfoIndex.FIRST_TASK_TS.value] == -1:
                        step[StepInfoIndex.FIRST_TASK_TS.value] = start_time
                break
        for cur_save in save_time:
            if cur_save.get('step') == cur_step:
                cur_save[cls.timeflag.get(add_type)] += duration
                break

    @classmethod
    def get_e2e_time(cls, step, step_list):
        for cur_step in step_list:
            if cur_step[StepInfoIndex.ID.value] == step:
                return cur_step[StepInfoIndex.E2E_END_TS.value] - cur_step[StepInfoIndex.E2E_START_TS.value]
        return 0

    def get_prepare_time(self, step, step_list):
        for cur_step in step_list:
            if cur_step[StepInfoIndex.ID.value] == step:
                first_task_start_ts = cur_step[StepInfoIndex.FIRST_TASK_TS.value]
                if step is None:
                    first_fwk_op = FwkFileParser(self._profiler_path).get_first_fwk_op()
                    return (first_task_start_ts - convert_ns2us_float(first_fwk_op.ts)) if first_fwk_op else 0
                return first_task_start_ts - cur_step[StepInfoIndex.FWK_START_TS.value]
        return 0

    def create_step_file(self, output_path: str, json_str: list, file_name: str) -> None:
        step_list = []
        save_time = []
        if not json_str:
            return
        # get step time
        for cur_step in self.step_range:
            step_list.append(
                [cur_step.get(Constant.STEP_ID), convert_ns2us_float(cur_step.get(Constant.START_TS)),
                 convert_ns2us_float(cur_step.get(Constant.END_TS)), -1, -1,
                 convert_ns2us_float(cur_step.get(Constant.FWK_START_TS)), -1])
            save_time.append(
                {'step': cur_step.get(Constant.STEP_ID), 'compute': 0, 'comunNotOverlp': 0, 'Overlp': 0, 'comun': 0,
                 'free': 0, 'stage': 0, 'bubble': 0, 'comunNotOverlpRec': 0, 'prepare': 0})
        if not self.step_range:
            save_time.append(
                {'step': None, 'compute': 0, 'comunNotOverlp': 0, 'Overlp': 0, 'comun': 0, 'free': 0, 'stage': 0,
                 'bubble': 0, 'comunNotOverlpRec': 0, 'prepare': 0})
            step_list.append([None, -1, -1, -1, -1, -1, -1])

        has_analysis_data_flag = False
        for data in json_str:
            if data.get('name') in {'Communication', 'Computing', 'Free', 'Communication(Not Overlapped)'}:
                self.count_time(data.get('name'), data.get('ts', 0), data.get('dur', 0), step_list, save_time)
                has_analysis_data_flag = True
            elif str(data.get('name')).startswith('hcom_receive'):
                self.count_time('hcom_receive', data.get('ts', 0), data.get('dur', 0), step_list, save_time)
        if not has_analysis_data_flag:
            return
        for calc_time in save_time:
            calc_time['comunNotOverlpRec'] = calc_time['comunNotOverlp'] - calc_time['bubble']
            calc_time['Overlp'] = calc_time['comun'] - calc_time['comunNotOverlp']
            calc_time['stage'] = self.get_e2e_time(calc_time['step'], step_list) - calc_time['bubble']
            calc_time['prepare'] = self.get_prepare_time(calc_time['step'], step_list)
        print_time = []
        for step in save_time:
            print_time.append(
                [step['step'], step['compute'], step['comunNotOverlp'], step['Overlp'], step['comun'], step['free'],
                 step['stage'], step['bubble'], step['comunNotOverlpRec'], step['prepare']])
        FileManager.create_csv_file(output_path, print_time, file_name, self.title)

    def run(self, deps_data: dict):
        try:
            self._init_step_range(deps_data)
            self.generate_view()
        except Exception:
            print_error_msg("Failed to generate step_trace_time.csv.")
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
