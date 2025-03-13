import os
import re
import shutil
import json

from ...prof_common_func._log import ProfilerLogger
from ...prof_common_func._utils import collect_env_vars
from ...prof_common_func._path_manager import ProfilerPathManager
from ...prof_common_func._file_manager import FileManager
from ...prof_common_func._constant import Constant, DbConstant, TableColumnsManager
from ...prof_common_func._db_manager import TorchDb
from ...prof_common_func._host_info import get_host_info
from .._base_parser import BaseParser
from ..._profiler_config import ProfilerConfig

__all__ = []


class BasicDbParser(BaseParser):
    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._cann_path = ProfilerPathManager.get_cann_path(self._profiler_path)
        ProfilerLogger.init(self._profiler_path, "BasicDbParser")
        self.logger = ProfilerLogger.get_instance()

    def run(self, deps_data: dict):
        try:
            cann_db_path = self.get_cann_db_path()
            if cann_db_path:
                shutil.move(cann_db_path, TorchDb().get_db_path())
            self.create_ascend_db()
            self.save_rank_info_to_db()
            self.save_host_info_to_db()
            self.save_env_vars_info_to_db()
            self.save_profiler_metadata_to_db()
        except Exception as error:
            self.logger.error("Failed to generate basic db file. Error: %s", str(error), exc_info=True)
            return Constant.FAIL, ""
        return Constant.SUCCESS, ""

    def get_cann_db_path(self):
        if not self._cann_path:
            return ""
        db_patten = '^msprof_\d+\.db$'
        for cann_file in os.listdir(self._cann_path):
            file_path = os.path.join(self._cann_path, cann_file)
            if re.match(db_patten, cann_file):
                try:
                    FileManager.check_db_file_vaild(file_path)
                except RuntimeError:
                    self.logger.warning("Invalid cann db file. file name is: %s", cann_file)
                    continue
                return file_path
        return ""
    
    def create_ascend_db(self):
        if not TorchDb().create_connect_db():
            raise RuntimeError(f"Failed to connect to db file: {TorchDb().get_db_path()}")

    def save_rank_info_to_db(self):
        if ProfilerConfig().rank_id == -1:
            return
        TorchDb().create_table_with_headers(DbConstant.TABLE_RANK_DEVICE_MAP,
                                            TableColumnsManager.TableColumns.get(DbConstant.TABLE_RANK_DEVICE_MAP))
        TorchDb().insert_data_into_table(DbConstant.TABLE_RANK_DEVICE_MAP,
                                         [[ProfilerConfig().rank_id, ProfilerPathManager.get_device_id(self._cann_path)]])
    
    def save_host_info_to_db(self):
        if TorchDb().judge_table_exist(DbConstant.TABLE_HOST_INFO):
            return
        host_info = get_host_info()
        TorchDb().create_table_with_headers(DbConstant.TABLE_HOST_INFO,
                                            TableColumnsManager.TableColumns.get(DbConstant.TABLE_HOST_INFO))
        TorchDb().insert_data_into_table(DbConstant.TABLE_HOST_INFO,
                                         [[host_info.get('host_uid'), host_info.get('host_name')]])

    def save_env_vars_info_to_db(self):
        env_vars_dict = collect_env_vars()
        TorchDb().create_table_with_headers(DbConstant.TABLE_META_DATA,
                                            TableColumnsManager.TableColumns.get(DbConstant.TABLE_META_DATA))
        TorchDb().insert_data_into_table(DbConstant.TABLE_META_DATA,
                                         [['ENV_VARIABLES', json.dumps(env_vars_dict.get('ENV_VARIABLES'))]])

    def save_profiler_metadata_to_db(self):
        profiler_metadata_path = os.path.join(self._profiler_path, Constant.PROFILER_META_DATA)
        if not os.path.exists(profiler_metadata_path):
            self.logger.warning("Can not find profiler_metadata.json, path is: %s", profiler_metadata_path)
            return
        profiler_metadata = FileManager.file_read_all(profiler_metadata_path)
        try:
            profiler_metadata = json.loads(profiler_metadata)
        except json.JSONDecodeError as e:
            self.logger.warning("profiler_metadata.json parse failed, error is: %s", str(e))
            return
        data = [
            [str(key), json.dumps(value)] for key, value in profiler_metadata.items()
        ]
        TorchDb().create_table_with_headers(DbConstant.TABLE_META_DATA,
                                            TableColumnsManager.TableColumns.get(DbConstant.TABLE_META_DATA))
        TorchDb().insert_data_into_table(DbConstant.TABLE_META_DATA, data)
