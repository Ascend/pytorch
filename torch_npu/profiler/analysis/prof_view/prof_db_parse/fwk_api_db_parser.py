import os

from enum import Enum
from ...prof_common_func.db_manager import DbManager
from ...prof_common_func.id_manager import Str2IdManager, ConnectionIdManager, CallChainIdManager
from ...prof_common_func.constant import Constant, DbConstant, TableColumnsManager, print_error_msg
from ..base_parser import BaseParser
from ...prof_parse.fwk_file_parser import FwkFileParser

__all__ = []


class _TorchOpDataOri(Enum):
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


class _TaskQueueDataOri(Enum):
    START_NS = 0
    END_NS = 1
    GLOBAL_TID = 2
    CORRELATION_ID = 3
    NAME = 4


class _PythonTraceApiDataOri(Enum):
    START_NS = 0
    END_NS = 1
    GLOBAL_TID = 2
    NAME = 3


class _CannNodeLaunchApiOri(Enum):
    START_NS = 0
    END_NS = 1
    GLOBAL_TID = 2
    CORRELATION_ID = 3


class FwkApiDbParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._conn = None
        self._cur = None
        self._db_path = ""
        self._max_cann_connection_id = 0
        self._fwk_apis = []
 
    def run(self, deps_data: dict):
        try:
            self._db_path = deps_data.get(Constant.DB_PARSER, "")
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
            self._fwk_apis.append([queue[_TaskQueueDataOri.START_NS.value],
                                   queue[_TaskQueueDataOri.END_NS.value],
                                   queue[_TaskQueueDataOri.GLOBAL_TID.value],
                                   ConnectionIdManager().get_id_from_connection_ids([queue[_TaskQueueDataOri.CORRELATION_ID.value] + self._max_cann_connection_id]),
                                   Str2IdManager().get_id_from_str(queue[_TaskQueueDataOri.NAME.value]),
                                   None, None, None, None, None])
        python_trace_apis = fwk_api_data.get("python_trace", [])
        for python_trace_api in python_trace_apis:
            self._fwk_apis.append([python_trace_api[_PythonTraceApiDataOri.START_NS.value],
                                   python_trace_api[_PythonTraceApiDataOri.END_NS.value],
                                   python_trace_api[_PythonTraceApiDataOri.GLOBAL_TID.value],
                                   None,
                                   Str2IdManager().get_id_from_str(python_trace_api[_PythonTraceApiDataOri.NAME.value]),
                                   None, None, None, None, None])
        torch_op_apis = fwk_api_data.get("torch_op", [])
        if not torch_op_apis:
            return
        # update torch op api connection id if there is cann api
        if self._max_cann_connection_id != 0:
            for torch_op_api in torch_op_apis:
                # update torch op api inner connection id, include fwd_bwd id
                if torch_op_api[_TorchOpDataOri.CONNECTION_ID.value]:
                    torch_op_api[_TorchOpDataOri.CONNECTION_ID.value] = [conn_id + self._max_cann_connection_id for conn_id in torch_op_api[_TorchOpDataOri.CONNECTION_ID.value]]
            self.get_torch_op_connection_ids_with_cann_api(task_queues, torch_op_apis)
        for torch_op_api in torch_op_apis:
            self._fwk_apis.append([torch_op_api[_TorchOpDataOri.START_NS.value],
                                   torch_op_api[_TorchOpDataOri.END_NS.value],
                                   torch_op_api[_TorchOpDataOri.GLOBAL_TID.value],
                                   None if not torch_op_api[_TorchOpDataOri.CONNECTION_ID.value] else ConnectionIdManager().get_id_from_connection_ids(torch_op_api[_TorchOpDataOri.CONNECTION_ID.value]),
                                   Str2IdManager().get_id_from_str(torch_op_api[_TorchOpDataOri.NAME.value]),
                                   torch_op_api[_TorchOpDataOri.SEQUENCE_NUM.value],
                                   torch_op_api[_TorchOpDataOri.FWD_THREAD_ID.value],
                                   None if not torch_op_api[_TorchOpDataOri.INPUT_DIMS.value] else Str2IdManager().get_id_from_str(torch_op_api[_TorchOpDataOri.INPUT_DIMS.value]),
                                   None if not torch_op_api[_TorchOpDataOri.INPUT_SHAPES.value] else Str2IdManager().get_id_from_str(torch_op_api[_TorchOpDataOri.INPUT_SHAPES.value]),
                                   None if not torch_op_api[_TorchOpDataOri.CALL_STACK.value] else CallChainIdManager().get_callchain_id_from_callstack(torch_op_api[_TorchOpDataOri.CALL_STACK.value])])
        mstx_mark_apis = fwk_api_data.get("mstx_op", [])
        if not mstx_mark_apis:
            return
        self.get_mstx_mark_op_connection_ids_with_cann_api(task_queues, mstx_mark_apis)
        for mstx_mark_api in mstx_mark_apis:
            self._fwk_apis.append([mstx_mark_api[_TorchOpDataOri.START_NS.value], mstx_mark_api[_TorchOpDataOri.END_NS.value], mstx_mark_api[_TorchOpDataOri.GLOBAL_TID.value],
                                   None if not mstx_mark_api[_TorchOpDataOri.CONNECTION_ID.value] else ConnectionIdManager().get_id_from_connection_ids(mstx_mark_api[_TorchOpDataOri.CONNECTION_ID.value]),
                                   Str2IdManager().get_id_from_str(mstx_mark_api[_TorchOpDataOri.NAME.value]),
                                   None, mstx_mark_api[_TorchOpDataOri.FWD_THREAD_ID.value], None, None, None])

    def get_mstx_mark_op_connection_ids_with_cann_api(self, task_queues: list, mstx_mark_apis: list):
        sql = "select startNs, endNs, globalTid, connectionId from {} order by startNs".format(DbConstant.TABLE_MSTX_EVENTS)
        cann_tx_apis = DbManager.fetch_all_data(self._cur, sql)
        if not cann_tx_apis:
            raise RuntimeWarning("Failed to get msprof_tx apis")
        mstx_mark_apis.sort(key=lambda x: x[_TorchOpDataOri.START_NS.value])
        mstx_op_len = len(mstx_mark_apis)
        if task_queues:
            self.get_torch_op_connection_ids_with_task_queue(task_queues, mstx_mark_apis, mstx_op_len, cann_tx_apis)

    def get_torch_op_connection_ids_with_cann_api(self, task_queues: list, torch_op_apis: list):
        sql = "select id from {} where value = 'launch'".format(DbConstant.TABLE_STRING_IDS)
        node_launch_str_ids = DbManager.fetch_one_data(self._cur, sql)
        node_launch_str_id = 0
        if node_launch_str_ids and node_launch_str_ids[0]:
            node_launch_str_id = node_launch_str_ids[0]
        else:
            raise RuntimeWarning("Failed to find node launch str id")
        sql = "select startNs, endNs, globalTid, connectionId from {} where name = {} and type = 10000 order by startNs".format(DbConstant.TABLE_CANN_API, node_launch_str_id) # 10000 : node level
        node_lauch_apis = DbManager.fetch_all_data(self._cur, sql)
        if not node_lauch_apis:
            raise RuntimeWarning("Failed to get node launch apis")
        torch_op_apis.sort(key=lambda x: x[_TorchOpDataOri.START_NS.value])
        torch_op_len = len(torch_op_apis)
        if task_queues:
            self.get_torch_op_connection_ids_with_task_queue(task_queues, torch_op_apis, torch_op_len, node_lauch_apis)
        else:
            self.get_torch_op_connection_ids_without_task_queue(torch_op_apis, torch_op_len, node_lauch_apis)

    def get_torch_op_connection_ids_with_task_queue(self, task_queues: list, torch_op_apis: list, torch_op_len: int, node_lauch_apis: list):
        mid_queue_idx = len(task_queues) // 2
        enqueue_list = task_queues[:mid_queue_idx]
        dequeue_list = task_queues[mid_queue_idx:]
        last_dequeue_index = 0
        last_torch_op_index = 0
        dequeue_len = len(dequeue_list)
        for node_launch_api in node_lauch_apis:
            for idx in range(last_dequeue_index, dequeue_len):
                if node_launch_api[_CannNodeLaunchApiOri.START_NS.value] > dequeue_list[idx][_TaskQueueDataOri.START_NS.value] and \
                    node_launch_api[_CannNodeLaunchApiOri.END_NS.value] < dequeue_list[idx][_TaskQueueDataOri.END_NS.value]:
                    last_dequeue_index = idx
                    enqeue = enqueue_list[idx]
                    last_torch_op_index = self.get_torch_op_connection_ids_with_enqueue(torch_op_apis,
                                                                                        torch_op_len,
                                                                                        enqeue,
                                                                                        last_torch_op_index,
                                                                                        node_launch_api[_CannNodeLaunchApiOri.CORRELATION_ID.value])
                    break

    def get_torch_op_connection_ids_with_enqueue(self, torch_op_apis: list, torch_op_len: int, enqeue: list, last_torch_op_index: int, connection_id: int) -> int:
        last_op_api = None
        for idx in range(last_torch_op_index, torch_op_len):
            if enqeue[_TaskQueueDataOri.START_NS.value] > torch_op_apis[idx][_TorchOpDataOri.END_NS.value]:
                continue
            if enqeue[_TaskQueueDataOri.START_NS.value] > torch_op_apis[idx][_TorchOpDataOri.START_NS.value] and enqeue[_TaskQueueDataOri.END_NS.value] < torch_op_apis[idx][_TorchOpDataOri.END_NS.value]:
                last_op_api = torch_op_apis[idx]
                last_torch_op_index = idx
            elif last_op_api:
                break
        if last_op_api:
            torch_op_apis[last_torch_op_index][_TorchOpDataOri.CONNECTION_ID.value].append(connection_id)
        return last_torch_op_index

    def get_torch_op_connection_ids_without_task_queue(self, torch_op_apis: list, torch_op_len: int, node_lauch_apis: list):
        last_op_api = None
        last_op_index = 0
        for node_launch_api in node_lauch_apis:
            for idx in range(last_op_index, torch_op_len):
                if torch_op_apis[idx][_TorchOpDataOri.GLOBAL_TID.value] != node_launch_api[_CannNodeLaunchApiOri.GLOBAL_TID.value]:
                    continue
                if node_launch_api[_CannNodeLaunchApiOri.START_NS.value] > torch_op_apis[idx][_TorchOpDataOri.END_NS.value]:
                    continue
                if node_launch_api[_CannNodeLaunchApiOri.START_NS.value] > torch_op_apis[idx][_TorchOpDataOri.START_NS.value] and \
                    node_launch_api[_CannNodeLaunchApiOri.END_NS.value] < torch_op_apis[idx][_TorchOpDataOri.END_NS.value]:
                    last_op_api = torch_op_apis[idx]
                    last_op_index = idx
                elif last_op_api:
                    torch_op_apis[last_op_index][_TorchOpDataOri.CONNECTION_ID.value].append(node_launch_api[_CannNodeLaunchApiOri.CORRELATION_ID.value])
                    last_op_api = None
                    break

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