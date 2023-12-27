from .base_parser import BaseParser
from ..prof_common_func.constant import Constant, print_error_msg
from ..prof_common_func.file_manager import FileManager
from ..prof_common_func.constant import convert_ns2us_float
from ..prof_parse.cann_file_parser import CANNFileParser
from ..prof_parse.fwk_cann_relation_parser import FwkCANNRelationParser


class TraceStepTimeParser(BaseParser):
    STEP_TRACE = "step_trace_time.csv"
    timeflag = {'Communication': 'comun', 'Computing': 'compute', 'Free': 'free',
                'Communication(Not Overlapped)': 'comunNotOverlp', 'hcom_receive': 'bubble'}
    title = ['Step', 'Computing', 'Communication(Not Overlapped)', 'Overlapped', 'Communication', 'Free', 'Stage',
             'Bubble', 'Communication(Not Overlapped and Exclude Receive)']

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
    def count_time(cls, addtype, addtime, durtime, step_list, save_time):
        cur_step = None
        if not cls.is_float_num(addtime) or not cls.is_float_num(durtime):
            print('Ts or dur format error!')
            return
        addtime = float(addtime)
        durtime = float(durtime)
        for step in step_list:
            if step[1] <= addtime <= step[2]:
                cur_step = step[0]
                break
        for step in step_list:
            if cur_step == step[0]:
                if addtime < step[3] or step[3] == -1:
                    step[3] = addtime
                if addtime + durtime > step[4] or step[4] == -1:
                    step[4] = addtime + durtime
                break
        for cur_save in save_time:
            if cur_save.get('step') == cur_step:
                cur_save[cls.timeflag.get(addtype)] += durtime
                break

    @classmethod
    def getE2ETime(cls, step, step_list):
        for curStep in step_list:
            if curStep[0] == step:
                return curStep[4] - curStep[3]
        return None

    def create_step_file(self, output_path: str, json_str: list, file_name: str) -> None:
        step_list = []
        save_time = []
        if not json_str:
            return
        # get step time
        for curStep in self.step_range:
            step_list.append(
                [curStep.get(Constant.STEP_ID), convert_ns2us_float(curStep.get(Constant.START_TS)),
                 convert_ns2us_float(curStep.get(Constant.END_TS)), -1, -1])
            save_time.append(
                {'step': curStep.get(Constant.STEP_ID), 'compute': 0, 'comunNotOverlp': 0, 'Overlp': 0, 'comun': 0,
                 'free': 0, 'stage': 0, 'bubble': 0, 'comunNotOverlpRec': 0})
        if not self.step_range:
            save_time.append(
                {'step': None, 'compute': 0, 'comunNotOverlp': 0, 'Overlp': 0, 'comun': 0, 'free': 0, 'stage': 0,
                 'bubble': 0, 'comunNotOverlpRec': 0})
            step_list.append([None, -1, -1, -1, -1])

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
            calc_time['stage'] = self.getE2ETime(calc_time['step'], step_list) - calc_time['bubble']
        print_time = []
        for step in save_time:
            print_time.append(
                [step['step'], step['compute'], step['comunNotOverlp'], step['Overlp'], step['comun'], step['free'],
                 step['stage'], step['bubble'], step['comunNotOverlpRec']])
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
