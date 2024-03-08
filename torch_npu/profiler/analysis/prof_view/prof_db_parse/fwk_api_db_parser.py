import os

from enum import Enum
from ...prof_common_func.db_manager import DbManager
from ...prof_common_func.id_manager import Str2IdManager
from ...prof_common_func.constant import Constant, DbConstant, TableColumnsManager, print_error_msg
from ..base_parser import BaseParser
from ...prof_parse.fwk_file_parser import FwkFileParser


class FwkApiTableRow(Enum):
    START_NS = 0
    END_NS = 1
    TYPE = 2
    GLOBAL_TID = 3
    CONNECTION_ID = 4
    NAME = 5
    API_ID = 6


class FwkApiInfoTableRow(Enum):
    API_ID = 0
    SEQUENCE_NUM = 1
    FWD_THREAD_ID = 2
    INPUT_DIMS = 3
    INPUT_SHAPES = 4
    STACK = 5


class FwkApiDbParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._fwk_api = []
        self._fwk_api_info = []
        self._conn = None
        self._cur = None
        self._db_path = ""
        self._max_cann_api_id = 0
        self._max_cann_connection_id = 0
 
    def run(self, depth_data: dict):
        try:
            self.init_db_connect()
            self.set_start_string_id()
            self.get_max_cann_id()
            self._fwk_api_data = FwkFileParser(self._profiler_path).get_fwk_api()
            self._fwk_api = self.update_fwk_api(self._fwk_api_data.get("fwk_api", []))
            self._fwk_api_info = self.update_fwk_api_info(self._fwk_api_data.get("fwk_api_info", []))
            self.save_api_data_to_db()
        except Exception:
            print_error_msg("Failed to generate framework api table.")
            DbManager.destroy_db_connect(self._conn, self._cur)
            return Constant.FAIL, None
        return Constant.SUCCESS, None

    def update_fwk_api(self, fwk_apis: list):
        if not fwk_apis:
            return []
        for fwk_api in fwk_apis:
            fwk_api.insert(FwkApiTableRow.TYPE.value, DbConstant.PYTORCH_API_TYPE_ID)
            if fwk_api[FwkApiTableRow.CONNECTION_ID.value] != DbConstant.DB_INVALID_VALUE:
                fwk_api[FwkApiTableRow.CONNECTION_ID.value] += self._max_cann_connection_id
            fwk_api[FwkApiTableRow.NAME.value] = Str2IdManager().get_id_from_str(fwk_api[FwkApiTableRow.NAME.value])
            fwk_api[FwkApiTableRow.API_ID.value] += self._max_cann_api_id
        return fwk_apis

    def update_call_stack(self, call_stack: str) -> str:
        if not call_stack:
            return ""
        stacks = call_stack.split(";\r\n")
        stack_ids = []
        for stack in stacks:
            stack_ids.append(Str2IdManager().get_id_from_str(stack))
        return ";".join(str(stack_id) for stack_id in stack_ids)

    def update_fwk_api_info(self, fwk_api_infos: list):
        if not fwk_api_infos:
            return []
        for fwk_api_info in fwk_api_infos:
            fwk_api_info[FwkApiInfoTableRow.API_ID.value] += self._max_cann_api_id
            fwk_api_info[FwkApiInfoTableRow.INPUT_DIMS.value] = DbConstant.DB_INVALID_VALUE if not fwk_api_info[FwkApiInfoTableRow.INPUT_DIMS.value] else Str2IdManager().get_id_from_str(fwk_api_info[FwkApiInfoTableRow.INPUT_DIMS.value])
            fwk_api_info[FwkApiInfoTableRow.INPUT_SHAPES.value] = DbConstant.DB_INVALID_VALUE if not fwk_api_info[FwkApiInfoTableRow.INPUT_SHAPES.value] else Str2IdManager().get_id_from_str(fwk_api_info[FwkApiInfoTableRow.INPUT_SHAPES.value])
            fwk_api_info[FwkApiInfoTableRow.STACK.value] = "" if not fwk_api_info[FwkApiInfoTableRow.STACK.value] else self.update_call_stack(fwk_api_info[FwkApiInfoTableRow.STACK.value])
        return fwk_api_infos

    def set_start_string_id(self):
        Str2IdManager().set_start_id(DbConstant.START_STRING_ID_FWK_API)
    
    def get_max_cann_id(self):
        if not DbManager.judge_table_exit(self._cur, DbConstant.TABLE_API):
            return
        sql = "select max(apiId) from {}".format(DbConstant.TABLE_API)
        apiIds = DbManager.fetch_one_data(self._cur, sql)
        if apiIds and apiIds[0]:
            self._max_cann_api_id = apiIds[0] + 1
        sql = "select max(connectionId) from {}".format(DbConstant.TABLE_API)
        connectionIds = DbManager.fetch_one_data(self._cur, sql)
        if connectionIds and connectionIds[0]:
            self._max_cann_connection_id = connectionIds[0] + 1
  
    def init_db_connect(self) -> None:
        self._db_path = os.path.join(self._output_path, DbConstant.DB_ASCEND_PYTORCH)
        self._conn, self._cur = DbManager.create_connect_db(self._db_path)
        if not (self._conn and self._cur):
            raise RuntimeError(f"Failed to connect to db file: {self._db_path}")

    def save_api_type(self):
        DbManager.create_table_with_headers(self._conn, self._cur, DbConstant.TABLE_ENUM_API_TYPE, TableColumnsManager.TableColumns.get(DbConstant.TABLE_ENUM_API_TYPE))
        DbManager.insert_data_into_table(self._conn, DbConstant.TABLE_ENUM_API_TYPE, [[DbConstant.PYTORCH_API_TYPE_ID, DbConstant.PYTORCH_API_TYPE]])
    
    def save_fwk_api(self):
        if not self._fwk_api:
            return
        DbManager.create_table_with_headers(self._conn, self._cur, DbConstant.TABLE_API, TableColumnsManager.TableColumns.get(DbConstant.TABLE_API))
        DbManager.insert_data_into_table(self._conn, DbConstant.TABLE_API, self._fwk_api)
    
    def save_fwk_api_info(self):
        if not self._fwk_api_info:
            return
        DbManager.create_table_with_headers(self._conn, self._cur, DbConstant.TABLE_API_INFO, TableColumnsManager.TableColumns.get(DbConstant.TABLE_API_INFO))
        DbManager.insert_data_into_table(self._conn, DbConstant.TABLE_API_INFO, self._fwk_api_info)

    def save_strings_id(self):
        DbManager.create_table_with_headers(self._conn, self._cur, DbConstant.TABLE_STRING_IDS, TableColumnsManager.TableColumns.get(DbConstant.TABLE_STRING_IDS))
        DbManager.insert_data_into_table(self._conn, DbConstant.TABLE_STRING_IDS, Str2IdManager().get_all_string_2_id_data())

    def save_api_data_to_db(self):
        self.save_api_type()
        self.save_fwk_api()
        self.save_fwk_api_info()
        self.save_strings_id()
        DbManager.destroy_db_connect(self._conn, self._cur)