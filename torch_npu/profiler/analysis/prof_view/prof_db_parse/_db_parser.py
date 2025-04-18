import os

from ._basic_db_parser import BasicDbParser
from ._communication_db_parser import CommunicationDbParser
from ._fwk_api_db_parser import FwkApiDbParser
from ._gc_record_db_parser import GCRecordDbParser
from ._memory_db_parser import MemoryDbParser
from ._step_info_db_parser import StepInfoDbParser
from ._trace_step_time_db_parser import TraceStepTimeDbParser
from ...prof_common_func._constant import Constant, DbConstant, print_error_msg
from ...prof_common_func._db_manager import TorchDb, AnalysisDb
from .._base_parser import BaseParser
from ..._profiler_config import ProfilerConfig
from ...prof_common_func._log import ProfilerLogger
from ...prof_common_func._path_manager import ProfilerPathManager

__all__ = []


class DbParser(BaseParser):
    PYTORCH_DB_MAP = {
        Constant.BASIC_DB_PARSER: BasicDbParser,
        Constant.FWK_API_DB_PARSER: FwkApiDbParser,
        Constant.MEMORY_DB_PARSER: MemoryDbParser,
        Constant.GC_RECORD_DB_PARSER: GCRecordDbParser,
    }

    ANALYSIS_DB_MAP = {
        Constant.STEP_INFO_DB_PARSER: StepInfoDbParser,
        Constant.COMMUNICATION_DB_PARSER: CommunicationDbParser,
        Constant.TRACE_STEP_TIME_DB_PARSER: TraceStepTimeDbParser,
    }

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        ProfilerLogger.init(self._profiler_path, "DbParser")
        self.logger = ProfilerLogger.get_instance()

    def run(self, deps_data: dict):
        ProfilerConfig().load_info(self._profiler_path)
        torch_db_path = DbConstant.DB_ASCEND_PYTORCH_PROFILER
        if ProfilerConfig().rank_id != -1:
            torch_db_path = f"ascend_pytorch_profiler_{ProfilerConfig().rank_id}.db"
        TorchDb().init(os.path.join(self._output_path, torch_db_path))
        AnalysisDb().init(os.path.join(self._output_path, DbConstant.DB_ANALYSIS))

        parser_db_map = self.PYTORCH_DB_MAP
        if ProfilerPathManager.get_cann_path(self._profiler_path) and ProfilerConfig().get_level() != Constant.LEVEL_NONE:
            parser_db_map = {**self.PYTORCH_DB_MAP, **self.ANALYSIS_DB_MAP}
        try:
            for name, parser in parser_db_map.items():
                parser_obj = parser(name, self._param_dict)
                parser_status, parser_data = parser_obj.run(deps_data)
                deps_data.setdefault(name, parser_data)
                if parser_status != Constant.SUCCESS:
                    print_error_msg(f"DB parser failed, parser is {name}")
        except Exception as error:
            self.logger.error("Failed to generate ascend_pytorch_profiler db file, error: %s", str(error), exc_info=True)
            return Constant.FAIL, ""
        finally:
            TorchDb().close()
            AnalysisDb().close()
        return Constant.SUCCESS, ""
