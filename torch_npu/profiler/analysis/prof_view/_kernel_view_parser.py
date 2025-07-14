from ._base_parser import BaseParser
from ..prof_common_func._constant import Constant, convert_ns2us_str
from ..prof_common_func._csv_headers import CsvHeaders
from ..prof_common_func._file_manager import FileManager
from ..prof_common_func._log import ProfilerLogger
from ..prof_bean._op_summary_bean import OpSummaryBean
from ..prof_parse._cann_file_parser import CANNFileParser, CANNDataEnum
from ..prof_parse._fwk_cann_relation_parser import FwkCANNRelationParser
from .._profiler_config import ProfilerConfig

__all__ = []


class KernelViewParser(BaseParser):
    KERNEL_VIEW = "kernel_details.csv"

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self.step_range = []

    @classmethod
    def _project_map_for_headers(cls, input_headers: list):
        project_map_dict = {}
        for index, header in enumerate(CsvHeaders.OP_SUMMARY_SHOW_HEADERS):
            project_map_dict[header] = CsvHeaders.OP_SUMMARY_KERNEL_BASE_HEADERS[index]

        output_headers = []
        for header in input_headers:
            if header in project_map_dict:
                output_headers.append(project_map_dict.get(header))
            else:
                output_headers.append(header)
        return output_headers

    def run(self, deps_data: dict):
        ProfilerLogger.init(self._profiler_path, "KernelViewParser")
        self.logger = ProfilerLogger.get_instance()
        try:
            ProfilerConfig().load_info(self._profiler_path)
            self._init_step_range(deps_data)
            self.generate_view()
        except Exception as e:
            self.logger.error("Failed to generate kernel_details.csv, error: %s", str(e), exc_info=True)
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
            kernel_dict = deps_data.get(Constant.RELATION_PARSER, {})
            if not kernel_dict:
                self.logger.error("Kernel view get step range failed, the kernel dict is empty.")
                return
            step_range = FwkCANNRelationParser(self._profiler_path).get_step_range(torch_op_node[0], kernel_dict)
            if not step_range:
                self.logger.warning("Kernel view get step range failed, the step range is empty.")
            for step_data in step_range:
                step_id = step_data.get(Constant.STEP_ID)
                step_start = convert_ns2us_str(step_data.get(Constant.START_TS, 0))
                step_end = convert_ns2us_str(step_data.get(Constant.END_TS, 0))
                self.step_range.append(
                    {Constant.STEP_ID: step_id, Constant.START_TS: step_start, Constant.END_TS: step_end})
