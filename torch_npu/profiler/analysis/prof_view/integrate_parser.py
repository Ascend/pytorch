from ..prof_common_func.file_manager import FileManager
from ..prof_view.base_view_parser import BaseViewParser
from ..prof_parse.cann_file_parser import CANNFileParser, CANNDataEnum
from ..profiler_config import ProfilerConfig


class IntegrateParser(BaseViewParser):
    """
    copy and integrate files from cann
    """
    CSV_FILENAME_MAP = {
        CANNDataEnum.AI_CPU: "data_preprocess.csv",
        CANNDataEnum.L2_CACHE: "l2_cache.csv"
    }

    def __init__(self, profiler_path: str):
        self._profiler_path = profiler_path

    def generate_view(self, output_path: str, **kwargs) -> None:
        for cann_data_enum, parser_bean in ProfilerConfig().get_parser_bean():
            self.generate_csv(cann_data_enum, parser_bean, output_path)

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
