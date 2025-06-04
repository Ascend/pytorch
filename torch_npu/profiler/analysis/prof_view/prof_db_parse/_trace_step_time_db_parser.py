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
from collections import defaultdict
from enum import Enum
from .._base_parser import BaseParser
from ...prof_common_func._constant import Constant, print_warn_msg
from ...prof_common_func._constant import DbConstant, TableColumnsManager
from ...prof_common_func._db_manager import AnalysisDb, TorchDb
from ...prof_common_func._constant import convert_ns2us_float
from ...prof_common_func._log import ProfilerLogger
from ...prof_common_func._time_range_calculator import CommunicationTimeRange, RangeCaculator
from ...prof_parse._fwk_file_parser import FwkFileParser

__all__ = []


class OpIndex(Enum):
    OP_NAME = 0
    START_NS = 1
    END_NS = 2
    DEVICE_ID = 3


class TraceStepTimeDbParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self.step_range = []
        self.compute_task_info = defaultdict(list)
        self.communication_op_info = defaultdict(list)
        ProfilerLogger.init(self._profiler_path, "TraceStepTimeDbParser")
        self.logger = ProfilerLogger.get_instance()

    @staticmethod
    def get_e2e_time(task_time_list):
        start_time = -1
        end_time = -1
        for time_range in task_time_list:
            if start_time == -1 or time_range.start_ts < start_time:
                start_time = time_range.start_ts
            if end_time == -1 or time_range.end_ts > end_time:
                end_time = time_range.end_ts
        return end_time - start_time

    def get_prepare_time(self, first_task_start_ts, step_info):
        if not first_task_start_ts:
            return 0
        if step_info.get(Constant.STEP_ID) is None:
            first_fwk_op = FwkFileParser(self._profiler_path).get_first_fwk_op()
            return (first_task_start_ts - first_fwk_op.ts) if first_fwk_op else 0
        return first_task_start_ts - step_info.get(Constant.FWK_START_TS, 0)

    def save_step_trace_db_data(self, step_trace_data):
        if not AnalysisDb().create_connect_db():
            print_warn_msg(f"Failed to connect to db file: {AnalysisDb().get_db_path()}")
            return
        AnalysisDb().create_table_with_headers(DbConstant.TABLE_STEP_TRACE_TIME,
                                               TableColumnsManager.TableColumns.get(DbConstant.TABLE_STEP_TRACE_TIME))
        AnalysisDb().insert_data_into_table(DbConstant.TABLE_STEP_TRACE_TIME, step_trace_data)

    def run(self, deps_data: dict):
        try:
            self._init_step_range(deps_data)
            self._init_task_info_from_db()
            self.generate_view()
        except Exception as error:
            self.logger.error("Failed to generate step_trace_time table, error: %s", str(error), exc_info=True)
            return Constant.FAIL, None
        return Constant.SUCCESS, None

    def generate_view(self) -> None:
        save_time = []
        if not self.step_range:
            save_time.append(
                {'step': None, 'compute': 0, 'comunNotOverlp': 0, 'Overlp': 0, 'comun': 0, 'free': 0,
                 'stage': 0, 'bubble': 0, 'comunNotOverlpRec': 0, 'prepare': 0})
        else:
            device_ids = list(set(self.compute_task_info.keys()) | set(self.communication_op_info.keys()))
            device_ids.sort()
            for device_id in device_ids:
                # get step time
                for cur_step in self.step_range:
                    save_info = {
                        'step': cur_step.get(Constant.STEP_ID), 'compute': 0, 'comunNotOverlp': 0, 'Overlp': 0,
                        'comun': 0, 'free': 0, 'stage': 0, 'bubble': 0, 'comunNotOverlpRec': 0, 'prepare': 0,
                        'deviceId': device_id
                    }
                    origin_compute_data = self._get_compute_data_in_step(cur_step, device_id)
                    origin_communication_data, bubble_data = self._get_communication_data_in_step(cur_step, device_id)
                    compute_data = RangeCaculator.merge_continuous_intervals(origin_compute_data)
                    save_info['compute'] = sum(data.end_ts - data.start_ts for data in compute_data)
                    communication_data = RangeCaculator.merge_continuous_intervals(origin_communication_data)
                    save_info['comun'] = sum(data.end_ts - data.start_ts for data in communication_data)
                    pure_communication_data, free_data = \
                        RangeCaculator.compute_pipeline_overlap(communication_data, compute_data)
                    save_info['comunNotOverlp'] = \
                        sum(data.end_ts - data.start_ts for data in pure_communication_data)
                    save_info['free'] = sum(data.end_ts - data.start_ts for data in free_data)
                    save_info['bubble'] = sum(data.end_ts - data.start_ts for data in bubble_data)
                    save_info['stage'] = self.get_e2e_time(compute_data + communication_data) - save_info['bubble']
                    first_task_start_ts = self._get_first_device_task_ts(compute_data, communication_data)
                    save_info['prepare'] = self.get_prepare_time(first_task_start_ts, cur_step)
                    save_time.append(save_info)

        for calc_time in save_time:
            calc_time['comunNotOverlpRec'] = calc_time['comunNotOverlp'] - calc_time['bubble']
            calc_time['Overlp'] = calc_time['comun'] - calc_time['comunNotOverlp']
        reformat_time = []
        for step in save_time:
            step_time_data = [step['compute'], step['comunNotOverlp'], step['Overlp'], step['comun'], step['free'],
                              step['stage'], step['bubble'], step['comunNotOverlpRec'], step['prepare']]
            reformat_time.append([step['deviceId'], step['step']] + \
                                 [convert_ns2us_float(data) for data in step_time_data])
        self.save_step_trace_db_data(reformat_time)

    def _init_step_range(self, deps_data: dict):
        self.step_range = deps_data.get(Constant.STEP_INFO_DB_PARSER, [])

    def _init_task_info_from_db(self):
        if not TorchDb().create_connect_db():
            print_warn_msg(f"Failed to connect to db file: {TorchDb().get_db_path()}")
            return
        if not TorchDb().judge_table_exist(DbConstant.TABLE_STRING_IDS):
            self.logger.error(f"{DbConstant.TABLE_STRING_IDS} does not exist.")
            return
        if TorchDb().judge_table_exist(DbConstant.TABLE_COMPUTE_TASK_INFO):
            sql = """
            SELECT 
                STRING_IDS.value,
                task.startNs,
                task.endNs,
                task.deviceId
            FROM COMPUTE_TASK_INFO AS comp
            JOIN TASK AS task
                ON comp.globalTaskId = task.globalTaskId
            JOIN STRING_IDS
                ON comp.name = STRING_IDS.id
            """
            compute_task_data = TorchDb().fetch_all_data(sql)
            for item in compute_task_data:
                self.compute_task_info[item[OpIndex.DEVICE_ID.value]].append(item)
        if TorchDb().judge_table_exist(DbConstant.TABLE_COMMUNICATION_OP):
            sql = """
            WITH comm_info AS (
                SELECT (SELECT value FROM STRING_IDS WHERE id = c.opName) AS opName,
                    startNs,
                    endNs,
                    connectionId
                FROM COMMUNICATION_OP c
            )
            SELECT 
                comm.opName,
                comm.startNs,
                comm.endNs,
                t.deviceId
            FROM comm_info comm
            JOIN (
                SELECT 
                    connectionId,
                    deviceId
                FROM TASK
                GROUP BY connectionId
                HAVING COUNT(DISTINCT deviceId) = 1
            ) t
            ON comm.connectionId = t.connectionId
            """
            communication_op_data = TorchDb().fetch_all_data(sql)
            for item in communication_op_data:
                self.communication_op_info[item[OpIndex.DEVICE_ID.value]].append(item)

    def _get_compute_data_in_step(self, step_info, device_id):
        compute_data = []
        for op_info in self.compute_task_info[device_id]:
            op_start_time = op_info[OpIndex.START_NS.value]
            if not (step_info.get(Constant.START_TS) <= op_start_time < step_info.get(Constant.END_TS)):
                continue
            time_range = RangeCaculator.generate_time_range(op_start_time, op_info[OpIndex.END_NS.value])
            compute_data.append(time_range)
        return compute_data

    def _get_communication_data_in_step(self, step_info, device_id):
        communication_data = []
        bubble_data = []
        for op_info in self.communication_op_info[device_id]:
            op_start_time = op_info[OpIndex.START_NS.value]
            if not (step_info.get(Constant.START_TS) <= op_start_time < step_info.get(Constant.END_TS)):
                continue
            time_range = RangeCaculator.generate_time_range(
                op_start_time, op_info[OpIndex.END_NS.value], class_range=CommunicationTimeRange)
            communication_data.append(time_range)
            op_name = op_info[OpIndex.OP_NAME.value]
            if op_name.startswith('hcom_receive'):
                bubble_data.append(time_range)
        return communication_data, bubble_data

    def _get_first_device_task_ts(self, compute_task, communication_task):
        first_compute_task = compute_task[0] if compute_task else None
        first_communication_task = communication_task[0] if communication_task else None
        if not first_compute_task and not first_communication_task:
            return 0
        first_task_start_ts = 0
        if first_compute_task:
            first_task_start_ts = first_compute_task.start_ts
        if first_communication_task and first_communication_task.start_ts < first_task_start_ts:
            first_task_start_ts = first_communication_task.start_ts
        return first_task_start_ts
