from ..prof_common_func.csv_headers import CsvHeaders
from ..prof_common_func.file_manager import FileManager
from ..prof_bean.op_summary_bean import OpSummaryBean
from ..prof_common_func.global_var import GlobalVar
from ..prof_parse.cann_file_parser import CANNFileParser, CANNDataEnum
from ..prof_view.base_view_parser import BaseViewParser
from ..profiler_config import ProfilerConfig


class KernelViewParser(BaseViewParser):
    KERNEL_VIEW = "kernel_details.csv"

    def __init__(self, profiler_path: str):
        super().__init__(profiler_path)

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

    def generate_view(self, output_path: str, **kwargs) -> None:
        op_summary_file_set = CANNFileParser(self._profiler_path).get_file_list_by_type(CANNDataEnum.OP_SUMMARY)
        summary_data = []
        output_headers = CsvHeaders.OP_SUMMARY_KERNEL_BASE_HEADERS
        for file_path in op_summary_file_set:
            all_data = FileManager.read_csv_file(file_path, OpSummaryBean)
            if all_data:
                OpSummaryBean.headers = all_data[
                    0].all_headers if ProfilerConfig().is_all_kernel_headers() else CsvHeaders.OP_SUMMARY_SHOW_HEADERS
                output_headers = self._project_map_for_headers(OpSummaryBean.headers)
            if not GlobalVar.step_range:
                summary_data.extend([data.row for data in all_data])
                continue
            for data in all_data:
                step_id = None
                for step_data in GlobalVar.step_range:
                    if step_data[1] <= data.ts <= step_data[2]:
                        step_id = step_data[0]
                        break
                summary_data.append([step_id] + data.row)

        headers = ["Step Id"] + output_headers if GlobalVar.step_range else output_headers
        FileManager.create_csv_file(output_path, summary_data, self.KERNEL_VIEW, headers)
