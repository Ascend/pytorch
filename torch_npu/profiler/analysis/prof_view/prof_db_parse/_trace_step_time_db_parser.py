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

import os
from enum import Enum
from .._base_parser import BaseParser
from ...prof_common_func._constant import Constant, print_error_msg, print_warn_msg
from ...prof_common_func._constant import DbConstant, TableColumnsManager
from ...prof_common_func._db_manager import DbManager
from ...prof_common_func._constant import convert_ns2us_float
from ...prof_common_func._time_range_calculator import CommunicationTimeRange, RangeCaculator
from ...prof_parse._fwk_file_parser import FwkFileParser

__all__ = []


class CommunicationOpIndex(Enum):
    OP_NAME = 0
    START_NS = 1
    END_NS = 2


class TraceStepTimeDbParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self.step_range = []
        self.string_id_map = {}
        self.compute_task_info = {}
        self.communication_op_info = []
        self.task_db_con = None
        self.task_db_curs = None
        self.analysis_db_con = None
        self.analysis_db_curs = None
        self.db_path = ""

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

    def save_step_trace_db_data(self, output_path, step_trace_data):
        db_path = os.path.join(output_path, DbConstant.DB_ANALYSIS)
        conn, curs = DbManager.create_connect_db(db_path)
        if not (conn and curs):
            print_warn_msg(f"Failed to connect to db file: {db_path}")
            return
        self.analysis_db_con = conn
        self.analysis_db_curs = curs
        DbManager.create_table_with_headers(conn, curs, DbConstant.TABLE_STEP_TRACE_TIME,
                                            TableColumnsManager.TableColumns.get(DbConstant.TABLE_STEP_TRACE_TIME))
        DbManager.insert_data_into_table(conn, DbConstant.TABLE_STEP_TRACE_TIME, step_trace_data)
        DbManager.destroy_db_connect(conn, curs)

    def run(self, deps_data: dict):
        try:
            self.db_path = deps_data.get(Constant.DB_PARSER, "")
            self._init_step_range(deps_data)
            self._init_task_info_from_db()
            self.generate_view()
        except Exception:
            print_error_msg("Failed to generate step_trace_time table.")
            DbManager.destroy_db_connect(self.task_db_con, self.task_db_curs)
            DbManager.destroy_db_connect(self.analysis_db_con, self.analysis_db_curs)
            return Constant.FAIL, None
        return Constant.SUCCESS, None

    def generate_view(self) -> None:
        save_time = []
        if not self.step_range:
            save_time.append(
                {'step': None, 'compute': 0, 'comunNotOverlp': 0, 'Overlp': 0, 'comun': 0, 'free': 0,
                 'stage': 0, 'bubble': 0, 'comunNotOverlpRec': 0, 'prepare': 0})
        else:
            # get step time
            for cur_step in self.step_range:
                save_info = {
                    'step': cur_step.get(Constant.STEP_ID), 'compute': 0, 'comunNotOverlp': 0, 'Overlp': 0, 
                    'comun': 0, 'free': 0, 'stage': 0, 'bubble': 0, 'comunNotOverlpRec': 0, 'prepare': 0
                }
                origin_compute_data = self._get_compute_data_in_step(cur_step)
                origin_communication_data, bubble_data = self._get_communication_data_in_step(cur_step)
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
            reformat_time.append([step['step'], ] + [convert_ns2us_float(data) for data in step_time_data])
        self.save_step_trace_db_data(self._output_path, reformat_time)

    def _init_step_range(self, deps_data: dict):
        self.step_range = deps_data.get(Constant.STEP_INFO_DB_PARSER, [])

    def _init_task_info_from_db(self):
        conn, curs = DbManager.create_connect_db(self.db_path)
        if not (conn and curs):
            print_warn_msg(f"Failed to connect to db file: {self.db_path}")
            return
        self.task_db_con = conn
        self.task_db_curs = curs
        if DbManager.judge_table_exist(curs, DbConstant.TABLE_STRING_IDS):
            sql = "select id, value from {}".format(DbConstant.TABLE_STRING_IDS)
            string_id_data = DbManager.fetch_all_data(curs, sql)
            self.string_id_map = {data[0]: data[1] for data in string_id_data}
        if DbManager.judge_table_exist(curs, DbConstant.TABLE_COMPUTE_TASK_INFO):
            sql = "select name, globalTaskId from {}".format(DbConstant.TABLE_COMPUTE_TASK_INFO)
            compute_task_data = DbManager.fetch_all_data(curs, sql)
            self.compute_task_info = {data[1]: data[0] for data in compute_task_data}
        if DbManager.judge_table_exist(curs, DbConstant.TABLE_COMMUNICATION_OP):
            sql = "select opName, startNs, endNs from {}".format(DbConstant.TABLE_COMMUNICATION_OP)
            self.communication_op_info = DbManager.fetch_all_data(curs, sql)
        DbManager.destroy_db_connect(conn, curs)

    def _get_compute_data_in_step(self, step_info):
        compute_data = []
        for task_id, task_info in step_info.get(Constant.TASK_INFO, {}).items():
            if task_id in self.compute_task_info:
                compute_data.append(
                    RangeCaculator.generate_time_range(task_info.get("startNs"), task_info.get("endNs")))
        return compute_data

    def _get_communication_data_in_step(self, step_info):
        communication_data = []
        bubble_data = []
        for op_info in self.communication_op_info:
            op_start_time = op_info[CommunicationOpIndex.START_NS.value]
            if not (step_info.get(Constant.START_TS) <= op_start_time < step_info.get(Constant.END_TS)):
                continue
            time_range = RangeCaculator.generate_time_range(
                op_start_time, op_info[CommunicationOpIndex.END_NS.value], class_range=CommunicationTimeRange)
            communication_data.append(time_range)
            op_name = self.string_id_map.get(op_info[CommunicationOpIndex.OP_NAME.value], '')
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
