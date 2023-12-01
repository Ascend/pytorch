from .base_parser import BaseParser
from ..prof_common_func.constant import Constant, print_error_msg, convert_ns2us_str
from ..prof_common_func.csv_headers import CsvHeaders
from ..prof_common_func.file_manager import FileManager
from ..prof_bean.op_summary_bean import OpSummaryBean
from ..prof_parse.cann_file_parser import CANNFileParser, CANNDataEnum
from ..prof_parse.fwk_cann_relation_parser import FwkCANNRelationParser
from ..profiler_config import ProfilerConfig


class KernelViewParser(BaseParser):
    KERNEL_VIEW = "kernel_details.csv"

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self.step_range = []

    @classmethod
    def _project_map_for_headers(cls, input_headers: list):
        project_map_dict = {CsvHeaders.OP_SUMMARY_SHOW_HEADERS[i]: CsvHeaders.OP_SUMMARY_KERNEL_BASE_HEADERS[i] for i in
                            range(len(CsvHeaders.OP_SUMMARY_SHOW_HEADERS))}

        output_headers = []
        for header in input_headers:
            if header in project_map_dict:
                output_headers.append(project_map_dict.get(header))
            else:
                output_headers.append(header)
        return output_headers

    def run(self, deps_data: dict):
        try:
            ProfilerConfig().load_info(self._profiler_path)
            self._init_step_range(deps_data)
            self.generate_view()
        except Exception:
            print_error_msg("Failed to generate kernel_details.csv.")
            return Constant.FAIL, None
        return Constant.SUCCESS, None

    def generate_view(self) -> None:
        op_summary_file_set = CANNFileParser(self._profiler_path).get_file_list_by_type(CANNDataEnum.OP_SUMMARY)
        summary_data = []
        output_headers = CsvHeaders.OP_SUMMARY_KERNEL_BASE_HEADERS
        for file_path in op_summary_file_set:
            all_data = FileManager.read_csv_file(file_path, OpSummaryBean)
            if all_data:
                OpSummaryBean.headers = all_data[
                    0].all_headers if ProfilerConfig().is_all_kernel_headers() else CsvHeaders.OP_SUMMARY_SHOW_HEADERS
                output_headers = self._project_map_for_headers(OpSummaryBean.headers)
            if not self.step_range:
                summary_data.extend([data.row for data in all_data])
                continue
            for data in all_data:
                step_id = None
                for step_data in self.step_range:
                    if step_data.get(Constant.START_TS) <= data.ts <= step_data.get(Constant.END_TS):
                        step_id = step_data.get(Constant.STEP_ID)
                        break
                summary_data.append([step_id] + data.row)

        headers = ["Step Id"] + output_headers if self.step_range else output_headers
        FileManager.create_csv_file(self._output_path, summary_data, self.KERNEL_VIEW, headers)

    def _init_step_range(self, deps_data: dict):
        torch_op_node = deps_data.get(Constant.TREE_BUILD_PARSER, [])
        if torch_op_node:
            step_range = FwkCANNRelationParser(self._profiler_path).get_step_range(torch_op_node[0], deps_data.get(
                Constant.RELATION_PARSER, {}))
            for step_data in step_range:
                step_id = step_data.get(Constant.STEP_ID)
                step_start = convert_ns2us_str(step_data.get(Constant.START_TS, 0))
                step_end = convert_ns2us_str(step_data.get(Constant.END_TS, 0))
                self.step_range.append(
                    {Constant.STEP_ID: step_id, Constant.START_TS: step_start, Constant.END_TS: step_end})
