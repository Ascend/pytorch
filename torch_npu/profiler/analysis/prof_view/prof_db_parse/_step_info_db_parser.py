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
from ...prof_common_func._constant import DbConstant, Constant, TableColumnsManager, print_warn_msg
from ...prof_common_func._db_manager import TorchDb
from ...prof_common_func._log import ProfilerLogger

__all__ = []


class StepInfoDbParser(BaseParser):

    NODE_LEVEL = 10000

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        ProfilerLogger.init(self._profiler_path, "StepInfoDbParser")
        self.logger = ProfilerLogger.get_instance()

    def run(self, deps_data: dict):
        try:
            torch_op_node = deps_data.get(Constant.TREE_BUILD_PARSER, [])
            step_range = self.get_step_range(torch_op_node[0] if torch_op_node else None)
        except Exception as error:
            self.logger.error("Failed to get step info from db, error: %s", str(error), exc_info=True)
            return Constant.FAIL, []
        return Constant.SUCCESS, step_range

    def get_api_data_in_time_range(self, begin_ts, end_ts) -> list:
        if not TorchDb().judge_table_exist(DbConstant.TABLE_CANN_API):
            print_warn_msg("Failed to get api data from db.")
            return []
        sql = f"select connectionId from {DbConstant.TABLE_CANN_API} " \
              f"where type={self.NODE_LEVEL} and {begin_ts} <= startNs and endNs <= {end_ts}"
        return TorchDb().fetch_all_data(sql)

    def get_all_api_data(self) -> list:
        if not TorchDb().judge_table_exist(DbConstant.TABLE_CANN_API):
            print_warn_msg("Failed to get api data from db.")
            return []
        sql = f"select connectionId from {DbConstant.TABLE_CANN_API} where type={self.NODE_LEVEL}"
        return TorchDb().fetch_all_data(sql)

    def get_task_info_from_api(self, api_data) -> dict:
        if not TorchDb().judge_table_exist(DbConstant.TABLE_TASK):
            print_warn_msg("Failed to get task data from db.")
            return {}
        sql = f"select startNs, endNs, connectionId, globalTaskId from {DbConstant.TABLE_TASK}"
        task_data = TorchDb().fetch_all_data(sql)
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
        if not TorchDb().create_connect_db():
            print_warn_msg(f"Failed to connect to db file: {TorchDb().get_db_path()}")
            return []
        step_range = []
        if not step_node_list:
            start_time = 0
            end_time = float('inf')
            step_id = None
            api_data = self.get_all_api_data()
            task_info = self.get_task_info_from_api(api_data)
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
                api_data = self.get_api_data_in_time_range(step_node.start_time, step_node.end_time)
                task_info = self.get_task_info_from_api(api_data)
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
        return step_range

    def save_step_time(self, step_node_list: list) -> None:
        if not step_node_list:
            return
        step_time_list = []
        for step_node in step_node_list:
            step_time_list.append([step_node.event.name.split("#")[-1], step_node.start_time, step_node.end_time])
        TorchDb().create_table_with_headers(DbConstant.TABLE_STEP_TIME, TableColumnsManager.TableColumns.get(DbConstant.TABLE_STEP_TIME))
        TorchDb().insert_data_into_table(DbConstant.TABLE_STEP_TIME, step_time_list)
