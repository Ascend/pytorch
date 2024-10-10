# Copyright (c) 2024, Huawei Technologies.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .._base_parser import BaseParser
from ...prof_bean._torch_op_node import TorchOpNode
from ...prof_common_func._constant import DbConstant, Constant, TableColumnsManager, print_error_msg, print_warn_msg
from ...prof_common_func._db_manager import DbManager

__all__ = []


class StepInfoDbParser(BaseParser):

    NODE_LEVEL = 10000

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self.db_conn = None
        self.db_curs = None
        self._db_path = ""

    def run(self, deps_data: dict):
        try:
            self._db_path = deps_data.get(Constant.DB_PARSER, "")
            torch_op_node = deps_data.get(Constant.TREE_BUILD_PARSER, [])
            step_range = self.get_step_range(torch_op_node[0] if torch_op_node else None)
        except Exception:
            print_error_msg("Failed to get step info from db.")
            DbManager.destroy_db_connect(self.db_conn, self.db_curs)
            return Constant.FAIL, []
        return Constant.SUCCESS, step_range

    def get_api_data_in_time_range(self, begin_ts, end_ts, db_cur) -> list:
        if not DbManager.judge_table_exist(db_cur, DbConstant.TABLE_CANN_API):
            print_warn_msg("Failed to get api data from db.")
            return []
        sql = f"select connectionId from {DbConstant.TABLE_CANN_API} " \
              f"where type={self.NODE_LEVEL} and {begin_ts} <= startNs and endNs <= {end_ts}"
        return DbManager.fetch_all_data(db_cur, sql)

    def get_all_api_data(self, db_cur) -> list:
        if not DbManager.judge_table_exist(db_cur, DbConstant.TABLE_CANN_API):
            print_warn_msg("Failed to get api data from db.")
            return []
        sql = f"select connectionId from {DbConstant.TABLE_CANN_API} where type={self.NODE_LEVEL}"
        return DbManager.fetch_all_data(db_cur, sql)

    def get_task_info_from_api(self, api_data, db_cur) -> dict:
        if not DbManager.judge_table_exist(db_cur, DbConstant.TABLE_TASK):
            print_warn_msg("Failed to get task data from db.")
            return {}
        sql = f"select startNs, endNs, connectionId, globalTaskId from {DbConstant.TABLE_TASK}"
        task_data = DbManager.fetch_all_data(db_cur, sql)
        api_connection_ids = {info[0] for info in api_data}
        api_task_info = {}
        for task_info in task_data:
            if task_info[2] in api_connection_ids:
                api_task_info[task_info[3]] = {'startNs': task_info[0], 'endNs': task_info[1]}
        return api_task_info

    def get_step_range(self, root_node: TorchOpNode) -> list:
        step_node_list = []
        if root_node is not None:
            step_node_list = [node for node in root_node.child_node_list if node.is_profiler_step()]
        conn, curs = DbManager.create_connect_db(self._db_path)
        if not (conn and curs):
            print_warn_msg(f"Failed to connect to db file: {self._db_path}")
            return []
        self.db_conn = conn
        self.db_curs = curs
        step_range = []
        if not step_node_list:
            start_time = 0
            end_time = float('inf')
            step_id = None
            api_data = self.get_all_api_data(curs)
            task_info = self.get_task_info_from_api(api_data, curs)
            device_start_ts = min(info['startNs'] for info in task_info.values()) if task_info else start_time
            device_end_ts = max(info['endNs'] for info in task_info.values()) if task_info else Constant.INVALID_VALUE
            step_range.append(
                {
                    Constant.STEP_ID: step_id,
                    Constant.START_TS: device_start_ts,
                    Constant.END_TS: max(device_end_ts, end_time),
                    Constant.TASK_INFO: task_info,
                    Constant.FWK_START_TS: start_time
                }
            )
        else:
            for step_node in step_node_list:
                step_id = step_node.event.name.split("#")[-1]
                api_data = self.get_api_data_in_time_range(step_node.start_time, step_node.end_time, curs)
                task_info = self.get_task_info_from_api(api_data, curs)
                device_start_ts = \
                    min(info['startNs'] for info in task_info.values()) if task_info else step_node.start_time
                device_end_ts = \
                    max(info['endNs'] for info in task_info.values()) if task_info else Constant.INVALID_VALUE
                step_range.append(
                    {
                        Constant.STEP_ID: step_id,
                        Constant.START_TS: device_start_ts,
                        Constant.END_TS: max(device_end_ts, step_node.end_time),
                        Constant.TASK_INFO: task_info,
                        Constant.FWK_START_TS: step_node.start_time
                    }
                )
        self.save_step_time(step_node_list)
        DbManager.destroy_db_connect(conn, curs)
        return step_range

    def save_step_time(self, step_node_list: list) -> None:
        if not step_node_list:
            return
        step_time_list = []
        for step_node in step_node_list:
            step_time_list.append([step_node.event.name.split("#")[-1], step_node.start_time, step_node.end_time])
        DbManager.create_table_with_headers(self.db_conn, self.db_curs, DbConstant.TABLE_STEP_TIME, TableColumnsManager.TableColumns.get(DbConstant.TABLE_STEP_TIME))
        DbManager.insert_data_into_table(self.db_conn, DbConstant.TABLE_STEP_TIME, step_time_list)
