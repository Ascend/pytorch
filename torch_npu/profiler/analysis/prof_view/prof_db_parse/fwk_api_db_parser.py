import os

from enum import Enum
from ...prof_common_func.db_manager import DbManager
from ...prof_common_func.id_manager import Str2IdManager, ConnectionIdManager, CallChainIdManager
from ...prof_common_func.constant import Constant, DbConstant, TableColumnsManager, print_error_msg
from ...prof_common_func.path_manager import ProfilerPathManager
from ..base_parser import BaseParser
from ...prof_parse.fwk_file_parser import FwkFileParser


class FwkApiTableRow(Enum):
    START_NS = 0
    END_NS = 1
    GLOBAL_TID = 2
    CONNECTION_ID = 3
    NAME = 4
    SEQUENCE_NUM = 5
    FWD_THREAD_ID = 6
    INPUT_DIMS = 7
    INPUT_SHAPES = 8
    CALLCHAIN_ID = 9


class TorchOpDataOri(Enum):
    START_NS = 0
    END_NS = 1
    GLOBAL_TID = 2
    CONNECTION_ID = 3
    NAME = 4
    SEQUENCE_NUM = 5
    FWD_THREAD_ID = 6
    INPUT_DIMS = 7
    INPUT_SHAPES = 8
    CALL_STACK = 9


class TaskQueueDataOri(Enum):
    START_NS = 0
    END_NS = 1
    GLOBAL_TID = 2
    CORRELATION_ID = 3
    NAME = 4


class PythonTraceApiDataOri(Enum):
    START_NS = 0
    END_NS = 1
    GLOBAL_TID = 2
    NAME = 3


class FwkApiDbParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._conn = None
        self._cur = None
        self._db_path = ""
        self._max_cann_connection_id = 0
        self._fwk_apis = []
 
    def run(self, depth_data: dict):
        try:
            self._db_path = depth_data.get(Constant.DB_PARSER, "")
            self.init_db_connect()
            self.set_start_string_id()
            self.get_max_cann_id()
            fwk_api_data = FwkFileParser(self._profiler_path).get_fwk_api()
            self.get_api_data_for_db(fwk_api_data)
            self.save_api_data_to_db()
        except Exception:
            print_error_msg("Failed to generate framework api table.")
            DbManager.destroy_db_connect(self._conn, self._cur)
            return Constant.FAIL, None
        return Constant.SUCCESS, None

    def get_api_data_for_db(self, fwk_api_data: dict):
        if not fwk_api_data:
            return
        task_queues = fwk_api_data.get("task_queue", [])
        for queue in task_queues:
            self._fwk_apis.append([queue[TaskQueueDataOri.START_NS.value],
                                   queue[TaskQueueDataOri.END_NS.value],
                                   queue[TaskQueueDataOri.GLOBAL_TID.value],
                                   ConnectionIdManager().get_id_from_connection_ids([queue[TaskQueueDataOri.CORRELATION_ID.value] + self._max_cann_connection_id]),
                                   Str2IdManager().get_id_from_str(queue[TaskQueueDataOri.NAME.value]),
                                   None, None, None, None, None])
        python_trace_apis = fwk_api_data.get("python_trace", [])
        for python_trace_api in python_trace_apis:
            self._fwk_apis.append([python_trace_api[PythonTraceApiDataOri.START_NS.value],
                                   python_trace_api[PythonTraceApiDataOri.END_NS.value],
                                   python_trace_api[PythonTraceApiDataOri.GLOBAL_TID.value],
                                   None,
                                   Str2IdManager().get_id_from_str(python_trace_api[PythonTraceApiDataOri.NAME.value]),
                                   None, None, None, None, None])
        torch_op_apis = fwk_api_data.get("torch_op", [])
        for torch_op_api in torch_op_apis:
            self._fwk_apis.append([torch_op_api[TorchOpDataOri.START_NS.value],
                                   torch_op_api[TorchOpDataOri.END_NS.value],
                                   torch_op_api[TorchOpDataOri.GLOBAL_TID.value],
                                   None if not torch_op_api[TorchOpDataOri.CONNECTION_ID.value] else ConnectionIdManager().get_id_from_connection_ids([torch_op_api[TorchOpDataOri.CONNECTION_ID.value] + self._max_cann_connection_id]),
                                   Str2IdManager().get_id_from_str(torch_op_api[TorchOpDataOri.NAME.value]),
                                   torch_op_api[TorchOpDataOri.SEQUENCE_NUM.value],
                                   torch_op_api[TorchOpDataOri.FWD_THREAD_ID.value],
                                   None if not torch_op_api[TorchOpDataOri.INPUT_DIMS.value] else Str2IdManager().get_id_from_str(torch_op_api[TorchOpDataOri.INPUT_DIMS.value]),
                                   None if not torch_op_api[TorchOpDataOri.INPUT_SHAPES.value] else Str2IdManager().get_id_from_str(torch_op_api[TorchOpDataOri.INPUT_SHAPES.value]),
                                   None if not torch_op_api[TorchOpDataOri.CALL_STACK.value] else CallChainIdManager().get_callchain_id_from_callstack(torch_op_api[TorchOpDataOri.CALL_STACK.value])])

    def set_start_string_id(self):
        Str2IdManager().set_start_id(DbConstant.START_STRING_ID_FWK_API)
    
    def get_max_cann_id(self):
        if not DbManager.judge_table_exist(self._cur, DbConstant.TABLE_CANN_API):
            return
        sql = "select max(connectionId) from {}".format(DbConstant.TABLE_CANN_API)
        connectionIds = DbManager.fetch_one_data(self._cur, sql)
        if connectionIds and connectionIds[0]:
            self._max_cann_connection_id = connectionIds[0] + 1
  
    def init_db_connect(self) -> None:
        self._conn, self._cur = DbManager.create_connect_db(self._db_path)
        if not (self._conn and self._cur):
            raise RuntimeError(f"Failed to connect to db file: {self._db_path}")

    def save_fwk_api(self):
        if not self._fwk_apis:
            return
        DbManager.create_table_with_headers(self._conn, self._cur, DbConstant.TABLE_PYTORCH_API, TableColumnsManager.TableColumns.get(DbConstant.TABLE_PYTORCH_API))
        DbManager.insert_data_into_table(self._conn, DbConstant.TABLE_PYTORCH_API, self._fwk_apis)

    def save_string_ids(self):
        DbManager.create_table_with_headers(self._conn, self._cur, DbConstant.TABLE_STRING_IDS, TableColumnsManager.TableColumns.get(DbConstant.TABLE_STRING_IDS))
        DbManager.insert_data_into_table(self._conn, DbConstant.TABLE_STRING_IDS, Str2IdManager().get_all_string_2_id_data())

    def sava_connection_ids(self):
        connection_ids = ConnectionIdManager().get_all_connection_ids()
        if not connection_ids:
            return
        save_connection_ids = []
        for index, conn_ids in connection_ids.items():
            for conn_id in conn_ids:
                save_connection_ids.append([index, conn_id])
        DbManager.create_table_with_headers(self._conn, self._cur, DbConstant.TABLE_CONNECTION_IDS, TableColumnsManager.TableColumns.get(DbConstant.TABLE_CONNECTION_IDS))
        DbManager.insert_data_into_table(self._conn, DbConstant.TABLE_CONNECTION_IDS, save_connection_ids)

    def save_callchain_ids(self):
        callchain_ids = CallChainIdManager().get_all_callchain_id()
        if not callchain_ids:
            return
        save_callchain_ids = []
        for index, callstack_ids in callchain_ids.items():
            for callstack_id in callstack_ids:
                save_callchain_ids.append([index] + callstack_id)
        DbManager.create_table_with_headers(self._conn, self._cur, DbConstant.TABLE_PYTORCH_CALLCHAINS, TableColumnsManager.TableColumns.get(DbConstant.TABLE_PYTORCH_CALLCHAINS))
        DbManager.insert_data_into_table(self._conn, DbConstant.TABLE_PYTORCH_CALLCHAINS, save_callchain_ids)

    def save_api_data_to_db(self):
        self.save_fwk_api()
        self.save_string_ids()
        self.sava_connection_ids()
        self.save_callchain_ids()
        DbManager.destroy_db_connect(self._conn, self._cur)