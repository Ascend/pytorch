from ...prof_common_func._db_manager import TorchDb
from .._base_parser import BaseParser
from ...prof_common_func._log import ProfilerLogger
from ...prof_common_func._id_manager import Str2IdManager, ConnectionIdManager, CallChainIdManager
from ...prof_common_func._constant import (
    Constant,
    DbConstant,
    TableColumnsManager,
    ApiType,
    TorchOpDataOri,
    TaskQueueDataOri,
    CannNodeLaunchApiOri
)

__all__ = []


class FwkApiDbParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._fwk_apis = []
        ProfilerLogger.init(self._profiler_path, "FwkApiDbParser")
        self.logger = ProfilerLogger.get_instance()

    def run(self, deps_data: dict):
        self.logger.info("FwkApiDbParser start.")
        try:
            self.init_db_connect()
            self.set_start_string_id()
            fwk_api_data = deps_data.get(Constant.DB_PRE_PARSER, {})
            self.logger.info("FwkApiDbParser get fwk api data finish.")
            self.get_api_data_for_db(fwk_api_data)
            self.logger.info("FwkApiDbParser get api data for db finish.")
            self.save_api_data_to_db()
            self.logger.info("FwkApiDbParser save api data to db finish.")
        except Exception as error:
            self.logger.error("Failed to generate framework api table, error: %s", str(error), exc_info=True)
            return Constant.FAIL, None
        self.logger.info("FwkApiDbParser finish.")
        return Constant.SUCCESS, None

    def get_api_data_for_db(self, fwk_api_data: dict):
        if not fwk_api_data:
            return
        task_enqueues = fwk_api_data.get(Constant.ENQUEUE_DATA, [])
        task_dequeues = fwk_api_data.get(Constant.DEQUEUE_DATA, [])
        torch_op_apis = fwk_api_data.get(Constant.TORCH_OP_DATA, [])
        python_trace_apis = fwk_api_data.get(Constant.PYTHON_TRACE_DATA, [])
        mstx_mark_apis = fwk_api_data.get(Constant.MSTX_OP_DATA, [])
        self._fwk_apis = python_trace_apis + task_enqueues + task_dequeues

        if TorchDb().judge_table_exist(DbConstant.TABLE_CANN_API):
            self.get_torch_op_connection_ids_with_cann_api(task_enqueues, task_dequeues, torch_op_apis)

        # update connection id for torch op
        connectionId_manager = ConnectionIdManager()
        for torch_op_api in torch_op_apis:
            torch_op_api[TorchOpDataOri.CONNECTION_ID] = connectionId_manager.get_id_from_connection_ids(torch_op_api[TorchOpDataOri.CONNECTION_ID]) if torch_op_api[TorchOpDataOri.CONNECTION_ID] else None
        self._fwk_apis.extend(torch_op_apis)

        if TorchDb().judge_table_exist(DbConstant.TABLE_MSTX_EVENTS):
            self.get_mstx_mark_op_connection_ids_with_cann_api(task_enqueues, task_dequeues, mstx_mark_apis)

        # update connection id for mstx mark op
        for mstx_mark_api in mstx_mark_apis:
            if mstx_mark_api[TorchOpDataOri.CONNECTION_ID]:
                mstx_mark_api[TorchOpDataOri.CONNECTION_ID] = connectionId_manager.get_id_from_connection_ids(mstx_mark_api[TorchOpDataOri.CONNECTION_ID])
        self._fwk_apis.extend(mstx_mark_apis)

    def get_mstx_mark_op_connection_ids_with_cann_api(self, task_enqueues: list, task_dequeues: list, mstx_mark_apis: list):
        if not mstx_mark_apis:
            return
        sql = "select startNs, endNs, globalTid, connectionId from {} order by startNs".format(
            DbConstant.TABLE_MSTX_EVENTS)
        cann_tx_apis = TorchDb().fetch_all_data(sql)
        if not cann_tx_apis:
            raise RuntimeWarning("Failed to get msprof_tx apis")
        mstx_mark_apis.sort(key=lambda x: x[TorchOpDataOri.START_NS])
        mstx_op_len = len(mstx_mark_apis)
        if task_enqueues and task_dequeues:
            self.get_torch_op_connection_ids_with_task_queue(task_enqueues, task_dequeues, mstx_mark_apis, mstx_op_len,
                                                             cann_tx_apis)

    def get_torch_op_connection_ids_with_cann_api(self, task_enqueues: list, task_dequeues: list, torch_op_apis: list):
        if not torch_op_apis:
            return
        sql = "select id from {} where value = 'launch'".format(DbConstant.TABLE_STRING_IDS)
        node_launch_str_ids = TorchDb().fetch_one_data(sql)
        node_launch_str_id = 0
        if node_launch_str_ids and node_launch_str_ids[0]:
            node_launch_str_id = node_launch_str_ids[0]
        else:
            self.logger.error("Failed to find node launch str id")
            return
        sql = "select startNs, endNs, globalTid, connectionId from {} " \
              "where name = {} and type = 10000 order by startNs" \
            .format(DbConstant.TABLE_CANN_API, node_launch_str_id)  # 10000 : node level
        node_launch_apis = TorchDb().fetch_all_data(sql)
        if not node_launch_apis:
            self.logger.error("Failed to get node launch apis")
            return
        torch_op_apis.sort(key=lambda x: x[TorchOpDataOri.START_NS])
        torch_op_len = len(torch_op_apis)
        if task_enqueues and task_dequeues:
            self.get_torch_op_connection_ids_with_task_queue(task_enqueues, task_dequeues, torch_op_apis, torch_op_len,
                                                             node_launch_apis)
        else:
            self.get_torch_op_connection_ids_without_task_queue(torch_op_apis, torch_op_len, node_launch_apis)

    def get_torch_op_connection_ids_with_task_queue(self, task_enqueues: list, task_dequeues: list, torch_op_apis: list, torch_op_len: int, node_lauch_apis: list):
        connection_id_manager = ConnectionIdManager()
        enqueue_corr_ids = {connection_id_manager.get_connection_ids_from_id(task_enqueue[TaskQueueDataOri.CORRELATION_ID])[0] for task_enqueue in task_enqueues}
        dequeue_corr_ids = {connection_id_manager.get_connection_ids_from_id(task_dequeue[TaskQueueDataOri.CORRELATION_ID])[0] for task_dequeue in task_dequeues}
        matched_corr_ids = enqueue_corr_ids & dequeue_corr_ids
        enqueue_list = [enqueue for enqueue in task_enqueues if connection_id_manager.get_connection_ids_from_id(enqueue[TaskQueueDataOri.CORRELATION_ID])[0] in matched_corr_ids]
        dequeue_list = [dequeue for dequeue in task_dequeues if connection_id_manager.get_connection_ids_from_id(dequeue[TaskQueueDataOri.CORRELATION_ID])[0] in matched_corr_ids]

        last_dequeue_index = 0
        last_torch_op_index = 0
        dequeue_len = len(dequeue_list)
        for node_launch_api in node_lauch_apis:
            for idx in range(last_dequeue_index, dequeue_len):
                if node_launch_api[CannNodeLaunchApiOri.START_NS] > dequeue_list[idx][TaskQueueDataOri.START_NS] and \
                    node_launch_api[CannNodeLaunchApiOri.END_NS] < dequeue_list[idx][TaskQueueDataOri.END_NS]:
                    last_dequeue_index = idx
                    enqeue = enqueue_list[idx]
                    last_torch_op_index = self.get_torch_op_connection_ids_with_enqueue(torch_op_apis,
                                                                                        torch_op_len,
                                                                                        enqeue,
                                                                                        last_torch_op_index,
                                                                                        node_launch_api[CannNodeLaunchApiOri.CORRELATION_ID])
                    break
                if dequeue_list[idx][TaskQueueDataOri.START_NS] > node_launch_api[CannNodeLaunchApiOri.END_NS]:
                    last_dequeue_index = idx
                    break

    def get_torch_op_connection_ids_with_enqueue(self, torch_op_apis: list, torch_op_len: int, enqeue: list, last_torch_op_index: int, connection_id: int) -> int:
        last_op_api = None
        for idx in range(last_torch_op_index, torch_op_len):
            if enqeue[TaskQueueDataOri.START_NS] > torch_op_apis[idx][TorchOpDataOri.END_NS]:
                continue
            if enqeue[TaskQueueDataOri.START_NS] > torch_op_apis[idx][TorchOpDataOri.START_NS] and enqeue[TaskQueueDataOri.END_NS] < torch_op_apis[idx][TorchOpDataOri.END_NS]:
                last_op_api = torch_op_apis[idx]
                last_torch_op_index = idx
            elif last_op_api:
                break
            if enqeue[TaskQueueDataOri.END_NS] < torch_op_apis[idx][TorchOpDataOri.START_NS]:
                last_torch_op_index = idx
                break
        if last_op_api:
            torch_op_apis[last_torch_op_index][TorchOpDataOri.CONNECTION_ID].append(connection_id)
        return last_torch_op_index

    def get_torch_op_connection_ids_without_task_queue(self, torch_op_apis: list, torch_op_len: int, node_lauch_apis: list):
        last_op_api = None
        last_op_index = 0
        for node_launch_api in node_lauch_apis:
            for idx in range(last_op_index, torch_op_len):
                if torch_op_apis[idx][TorchOpDataOri.GLOBAL_TID] != node_launch_api[CannNodeLaunchApiOri.GLOBAL_TID]:
                    continue
                if node_launch_api[CannNodeLaunchApiOri.START_NS] > torch_op_apis[idx][TorchOpDataOri.END_NS]:
                    continue
                if node_launch_api[CannNodeLaunchApiOri.START_NS] > torch_op_apis[idx][TorchOpDataOri.START_NS] and \
                    node_launch_api[CannNodeLaunchApiOri.END_NS] < torch_op_apis[idx][TorchOpDataOri.END_NS]:
                    last_op_api = torch_op_apis[idx]
                    last_op_index = idx
                elif last_op_api:
                    torch_op_apis[last_op_index][TorchOpDataOri.CONNECTION_ID].append(node_launch_api[CannNodeLaunchApiOri.CORRELATION_ID])
                    last_op_api = None
                    break
                if node_launch_api[CannNodeLaunchApiOri.END_NS] < torch_op_apis[idx][TorchOpDataOri.START_NS]:
                    last_op_index = idx
                    break

    def set_start_string_id(self):
        Str2IdManager().set_start_id(DbConstant.START_STRING_ID_FWK_API)

    def init_db_connect(self) -> None:
        if not TorchDb().create_connect_db():
            raise RuntimeError(f"Failed to connect to db file: {TorchDb().get_db_path()}")

    def save_fwk_api(self):
        if not self._fwk_apis:
            return
        TorchDb().create_table_with_headers(DbConstant.TABLE_PYTORCH_API,
                                            TableColumnsManager.TableColumns.get(DbConstant.TABLE_PYTORCH_API))
        TorchDb().insert_data_into_table(DbConstant.TABLE_PYTORCH_API, self._fwk_apis)

    def save_string_ids(self):
        TorchDb().create_table_with_headers(DbConstant.TABLE_STRING_IDS,
                                            TableColumnsManager.TableColumns.get(DbConstant.TABLE_STRING_IDS))
        TorchDb().insert_data_into_table(DbConstant.TABLE_STRING_IDS,
                                         Str2IdManager().get_all_string_2_id_data())

    def sava_connection_ids(self):
        connection_ids = ConnectionIdManager().get_all_connection_ids()
        if not connection_ids:
            return
        save_connection_ids = []
        for index, conn_ids in connection_ids.items():
            for conn_id in conn_ids:
                save_connection_ids.append([index, conn_id])
        TorchDb().create_table_with_headers(DbConstant.TABLE_CONNECTION_IDS,
                                            TableColumnsManager.TableColumns.get(DbConstant.TABLE_CONNECTION_IDS))
        TorchDb().insert_data_into_table(DbConstant.TABLE_CONNECTION_IDS, save_connection_ids)

    def save_callchain_ids(self):
        callchain_ids = CallChainIdManager().get_all_callchain_id()
        if not callchain_ids:
            return
        save_callchain_ids = []
        for index, callstack_ids in callchain_ids.items():
            for callstack_id in callstack_ids:
                save_callchain_ids.append([index] + callstack_id)
        TorchDb().create_table_with_headers(DbConstant.TABLE_PYTORCH_CALLCHAINS,
                                            TableColumnsManager.TableColumns.get(DbConstant.TABLE_PYTORCH_CALLCHAINS))
        TorchDb().insert_data_into_table(DbConstant.TABLE_PYTORCH_CALLCHAINS, save_callchain_ids)

    def save_enum_api_types_to_db(self):
        if not TorchDb().judge_table_exist(DbConstant.TABLE_ENUM_API_TYPE):
            TorchDb().create_table_with_headers(DbConstant.TABLE_ENUM_API_TYPE,
                                                TableColumnsManager.TableColumns.get(DbConstant.TABLE_ENUM_API_TYPE))
        api_types = [
            (ApiType.TORCH_OP, 'op'),
            (ApiType.TASK_QUEUE, 'queue'),
            (ApiType.PYTHON_TRACE, 'trace'),
            (ApiType.MSTX_OP, 'mstx')
        ]
        TorchDb().insert_data_into_table(DbConstant.TABLE_ENUM_API_TYPE, api_types)

    def save_api_data_to_db(self):
        self.save_fwk_api()
        self.save_string_ids()
        self.sava_connection_ids()
        self.save_callchain_ids()
        self.save_enum_api_types_to_db()
