import os
import re
import shutil

from ...prof_common_func.path_manager import ProfilerPathManager
from ...prof_common_func.file_manager import FileManager
from ...prof_common_func.constant import Constant, DbConstant, TableColumnsManager, print_error_msg, print_warn_msg
from ...prof_common_func.db_manager import DbManager
from ...prof_common_func.host_info import _get_host_info
from ..base_parser import BaseParser
from ...profiler_config import ProfilerConfig

__all__ = []


class DbParser(BaseParser):
    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._cann_path = ProfilerPathManager.get_cann_path(self._profiler_path)
        self._ascend_db_path = os.path.join(self._output_path, DbConstant.DB_ASCEND_PYTORCH_PROFILER)
        self._conn = None
        self._cur = None

    def run(self, depth_data: dict):
        try:
            ProfilerConfig().load_info(self._profiler_path)
            if ProfilerConfig().rank_id != -1:
                self._ascend_db_path = os.path.join(self._output_path, f"ascend_pytorch_profiler_{ProfilerConfig().rank_id}.db")
            cann_db_path = self.get_cann_db_path()
            if cann_db_path:
                shutil.move(cann_db_path, self._ascend_db_path)
            self.create_ascend_db()
            self.save_rank_info_to_db()
            self.save_host_info_to_db()
            DbManager.destroy_db_connect(self._conn, self._cur)
        except RuntimeError:
            print_error_msg("Failed to generate ascend_pytorch_profiler db file.")
            DbManager.destroy_db_connect(self._conn, self._cur)
            return Constant.FAIL, ""
        return Constant.SUCCESS, self._ascend_db_path

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
                    print_warn_msg("Invalid cann db file.")
                    continue
                return file_path
        return ""
    
    def create_ascend_db(self):
        self._conn, self._cur = DbManager.create_connect_db(self._ascend_db_path)
        if not (self._conn and self._cur):
            raise RuntimeError(f"Failed to connect to db file: {self._ascend_db_path}")

    def save_rank_info_to_db(self):
        if ProfilerConfig().rank_id == -1:
            return
        DbManager.create_table_with_headers(self._conn, self._cur, DbConstant.TABLE_RANK_DEVICE_MAP, TableColumnsManager.TableColumns.get(DbConstant.TABLE_RANK_DEVICE_MAP))
        DbManager.insert_data_into_table(self._conn, DbConstant.TABLE_RANK_DEVICE_MAP, [[ProfilerConfig().rank_id, ProfilerPathManager.get_device_id(self._cann_path)]])
    
    def save_host_info_to_db(self):
        if DbManager.judge_table_exist(self._cur, DbConstant.TABLE_HOST_INFO):
            return
        host_info = _get_host_info()
        DbManager.create_table_with_headers(self._conn, self._cur, DbConstant.TABLE_HOST_INFO, TableColumnsManager.TableColumns.get(DbConstant.TABLE_HOST_INFO))
        DbManager.insert_data_into_table(self._conn, DbConstant.TABLE_HOST_INFO, [[host_info.get('host_uid'), host_info.get('host_name')]])
