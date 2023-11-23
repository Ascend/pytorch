from .base_parser import BaseParser
from ..prof_common_func.constant import Constant, print_error_msg
from ..prof_common_func.file_manager import FileManager
from ..prof_parse.cann_file_parser import CANNFileParser, CANNDataEnum
from ..profiler_config import ProfilerConfig


class IntegrateParser(BaseParser):
    """
    copy and integrate files from cann
    """
    CSV_FILENAME_MAP = {
        CANNDataEnum.AI_CPU: "data_preprocess.csv",
        CANNDataEnum.L2_CACHE: "l2_cache.csv",
        CANNDataEnum.OP_STATISTIC: "op_statistic.csv",
        CANNDataEnum.NPU_MODULE_MEM: "npu_module_mem.csv"
    }

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)

    def run(self, deps_data: dict):
        try:
            ProfilerConfig().load_info(self._profiler_path)
            self.generate_view()
        except Exception:
            print_error_msg("Failed to generate data_preprocess.csv or l2_cache.csv.")
            return Constant.FAIL, None
        return Constant.SUCCESS, None

    def generate_view(self) -> None:
        for cann_data_enum, parser_bean in ProfilerConfig().get_parser_bean():
            self.generate_csv(cann_data_enum, parser_bean, self._output_path)

    def generate_csv(self, cann_data_enum: int, parser_bean: any, output_path: str) -> None:
        """
        summarize data to generate csv files
        Returns: None
        """
        file_set = CANNFileParser(self._profiler_path).get_file_list_by_type(cann_data_enum)
        summary_data = []
        output_headers = []
        for file in file_set:
            all_data = FileManager.read_csv_file(file, parser_bean)
            for data in all_data:
                summary_data.append(data.row)
            if all_data and not output_headers:
                output_headers = all_data[0].headers
        FileManager.create_csv_file(output_path, summary_data,
                                    self.CSV_FILENAME_MAP.get(cann_data_enum, "none"), output_headers)
