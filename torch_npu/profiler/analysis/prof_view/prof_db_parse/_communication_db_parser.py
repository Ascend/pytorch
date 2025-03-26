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

from enum import Enum

from ...prof_parse._cann_file_parser import CANNDataEnum, CANNFileParser
from ...prof_common_func._constant import Constant, DbConstant, TableColumnsManager
from ...prof_common_func._db_manager import AnalysisDb
from ...prof_common_func._constant import convert_us2ns
from ...prof_common_func._db_manager import DbManager
from ...prof_common_func._log import ProfilerLogger
from .._communication_parser import CommunicationParser

__all__ = []


class CommBandwidthTableRow(Enum):
    HCCL_OP_NAME = 0
    GROUP_NAME = 1
    TRANSPORT_TYPE = 2
    TRANSIT_SIZE = 3
    TRANSIT_TIME = 4
    BANDWIDTH = 5
    LARGE_PACKET_RATIO = 6
    PACKAGE_SIZE = 7
    COUNT = 8
    TOTALL_DURATION = 9
    STEP = 10
    OP_TYPE = 11


class CommMatarixTableRow(Enum):
    HCCL_OP_NAME = 0
    GROUP_NAME = 1
    SRC_RANK = 2
    DST_RANK = 3
    TRANSPORT_TYPE = 4
    TRANSIT_SIZE = 5
    TRANSIT_TIME = 6
    BANDWIDTH = 7
    STEP = 8
    OP_TYPE = 9


class CommTimeTableRow(Enum):
    HCCL_OP_NAME = 0
    GROUP_NAME = 1
    START_TIMESTAMP = 2
    ELAPSE_TIME = 3
    TRANSIT_TIME = 4
    WAIT_TIME = 5
    SYNCHRONIZATION_TIME = 6
    IDLE_TIME = 7
    STEP = 8
    OP_TYPE = 9


class CommunicationDbParser(CommunicationParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self.cann_comm_db_conn = None
        self.cann_comm_db_curs = None
        ProfilerLogger.init(self._profiler_path, "CommunicationDbParser")
        self.logger = ProfilerLogger.get_instance()

    def run(self, deps_data: dict):
        try:
            self._init_step_list(deps_data)
            self.generate_view()
        except Exception as error:
            self.logger.error("Failed to generate communication table, error: %s", str(error), exc_info=True)
            DbManager.destroy_db_connect(self.cann_comm_db_conn, self.cann_comm_db_curs)
            return Constant.FAIL, None
        return Constant.SUCCESS, None

    def _init_step_list(self, deps_data: dict):
        self.step_list = deps_data.get(Constant.STEP_INFO_DB_PARSER, [])
        if not self.step_list:
            self.step_list = [{
                Constant.STEP_ID: None, Constant.START_TS: 0, Constant.END_TS: float('inf'), Constant.TASK_INFO: []
            }]

    def generate_view(self) -> None:
        self.generate_communication_db()
    
    def generate_communication_db(self):
        db_files = CANNFileParser(self._profiler_path).get_file_list_by_type(CANNDataEnum.ANALYSIS_DB)
        if not db_files:
            return
        band_width_data, matrix_data, time_data = self.get_communication_db_data(list(db_files)[0])
        band_width_data, matrix_data, time_data = \
            self.set_step_and_type_info_for_db_data(band_width_data, matrix_data, time_data)
        matrix_data = self.reformat_matrix_db_data(matrix_data)
        self.save_communication_db_data(band_width_data, matrix_data, time_data)

    def get_communication_db_data(self, db_path: str):
        # 在处理原analysis.db里的数据
        band_width_data, matrix_data, time_data = [], [], []
        conn, curs = DbManager.create_connect_db(db_path)
        if not (conn and curs):
            self.logger.warning("Failed to connect to db file: %s", db_path)
            return band_width_data, matrix_data, time_data
        self.cann_comm_db_conn = conn
        self.cann_comm_db_curs = curs
        if DbManager.judge_table_exist(curs, DbConstant.TABLE_ANALYZER_BANDWIDTH):
            sql = "select hccl_op_name, group_name, transport_type, transit_size, transit_time, " \
                  "bandwidth, large_packet_ratio, package_size, count, total_duration " \
                  "from {};".format(DbConstant.TABLE_ANALYZER_BANDWIDTH)
            band_width_data = DbManager.fetch_all_data(curs, sql)
        if DbManager.judge_table_exist(curs, DbConstant.TABLE_ANALYZER_MATRIX):
            sql = "select hccl_op_name, group_name, src_rank, dst_rank, "\
                  "transport_type, transit_size, transit_time, bandwidth " \
                  "from {};".format(DbConstant.TABLE_ANALYZER_MATRIX)
            matrix_data = DbManager.fetch_all_data(curs, sql)
        if DbManager.judge_table_exist(curs, DbConstant.TABLE_ANALYZER_TIME):
            sql = "select hccl_op_name, group_name, start_timestamp, elapse_time, "\
                  "transit_time, wait_time, synchronization_time, idle_time " \
                  "from {};".format(DbConstant.TABLE_ANALYZER_TIME)
            time_data = DbManager.fetch_all_data(curs, sql)
        DbManager.destroy_db_connect(conn, curs)
        return band_width_data, matrix_data, time_data

    def set_step_and_type_info_for_db_data(self, band_width_data, matrix_data, time_data):
        step_band_width_data = []
        step_matrix_data = []
        step_time_data = []
        time_op_name_idx_map = {
            "%s@%s" % (data[0], data[1]): {"band_idx": [], "matrix_idx": []} for data in time_data
        }
        for idx, data in enumerate(band_width_data):
            op_name = "%s@%s" % (
                data[CommBandwidthTableRow.HCCL_OP_NAME.value], data[CommBandwidthTableRow.GROUP_NAME.value])
            if op_name in time_op_name_idx_map:
                time_op_name_idx_map[op_name]["band_idx"].append(idx)
        for idx, data in enumerate(matrix_data):
            op_name = "%s@%s" % (
                data[CommMatarixTableRow.HCCL_OP_NAME.value], data[CommMatarixTableRow.GROUP_NAME.value])
            if op_name in time_op_name_idx_map:
                time_op_name_idx_map[op_name]["matrix_idx"].append(idx)
        for idx, data in enumerate(time_data):
            op_name = "%s@%s" % (data[CommTimeTableRow.HCCL_OP_NAME.value], data[CommTimeTableRow.GROUP_NAME.value])
            lower_op_name = op_name.lower()
            if lower_op_name.startswith(self.HCOM_SEND) or lower_op_name.startswith(self.HCOM_RECEIVE) \
                    or lower_op_name.startswith(self.HCOM_BATCHSENDRECV):
                op_type = self.P2P
            elif lower_op_name.startswith(self.TOTAL):
                op_type = ''
            else:
                op_type = self.COLLECTIVE
            start_time = convert_us2ns(data[CommTimeTableRow.START_TIMESTAMP.value])
            for step_info in self.step_list:
                if step_info.get("start_ts", -1) <= start_time <= step_info.get("end_ts", -1):
                    step = "step" + step_info.get("step_id") if step_info.get("step_id") else "step"
                    step_time_data.append(list(data) + [step, op_type])
                    for matrix_idx in time_op_name_idx_map.get(op_name, {}).get("matrix_idx", []):
                        step_matrix_data.append(list(matrix_data[matrix_idx]) + [step, op_type])
                    for band_idx in time_op_name_idx_map.get(op_name, {}).get("band_idx", []):
                        step_band_width_data.append(list(band_width_data[band_idx]) + [step, op_type])
                    break
        return step_band_width_data, step_matrix_data, step_time_data

    def reformat_matrix_db_data(self, matrix_data) -> list:
        def extract_data_from_dict(step, op_type, op_dict: dict):
            res_data = []
            for op_name, val in op_dict.items():
                matrix_op_name, group_name = op_name.split("@")
                for link, op_info in val.items():
                    src_rank, dst_rank = link.split("-")
                    hccl_op_name = op_info.get("Op Name", "")
                    res_data.append([
                        matrix_op_name, group_name, src_rank, dst_rank, op_info.get(self.TRANSPORT_TYPE),
                        op_info.get(self.TRANSIT_SIZE_MB), op_info.get(self.TRANSIT_TIME_MS),
                        op_info.get(self.BANDWIDTH_GB_S), step, op_type, hccl_op_name
                    ])
            return res_data
    
        step_op_dict = {}
        for data in matrix_data:
            op_name = \
                "%s@%s" % (data[CommMatarixTableRow.HCCL_OP_NAME.value], data[CommMatarixTableRow.GROUP_NAME.value])
            link = "%s-%s" % (data[CommMatarixTableRow.SRC_RANK.value], data[CommMatarixTableRow.DST_RANK.value])
            info = {
                self.TRANSPORT_TYPE: data[CommMatarixTableRow.TRANSPORT_TYPE.value],
                self.TRANSIT_SIZE_MB: data[CommMatarixTableRow.TRANSIT_SIZE.value],
                self.TRANSIT_TIME_MS: data[CommMatarixTableRow.TRANSIT_TIME.value],
                self.BANDWIDTH_GB_S: data[CommMatarixTableRow.BANDWIDTH.value]
            }
            step = data[CommMatarixTableRow.STEP.value]
            op_type = data[CommMatarixTableRow.OP_TYPE.value]
            if op_type == self.P2P:
                p2p_dict = step_op_dict.setdefault(step, {}).setdefault(self.P2P, {})
                p2p_dict.setdefault(op_name, {}).update({link: info})
            elif op_type == self.COLLECTIVE:
                collective_dict = step_op_dict.setdefault(step, {}).setdefault(self.COLLECTIVE, {})
                collective_dict.setdefault(op_name, {}).update({link: info})
        reformat_data = []
        for step, op_dict in step_op_dict.items():
            p2p_dict = op_dict.get(self.P2P, {})
            p2p_dict = self.integrate_matrix_data(self.get_comm_type(p2p_dict))
            reformat_data += extract_data_from_dict(step, self.P2P, p2p_dict)
            collective_dict = op_dict.get(self.COLLECTIVE, {})
            collective_dict = self.integrate_matrix_data(self.get_comm_type(collective_dict))
            reformat_data += extract_data_from_dict(step, self.COLLECTIVE, collective_dict)
        return reformat_data

    def save_communication_db_data(self, band_width_data, matrix_data, time_data):
        if not AnalysisDb().create_connect_db():
            self.logger.warning("Failed to connect to db file: %s", AnalysisDb().get_db_path())
            return
        AnalysisDb().create_table_with_headers(DbConstant.TABLE_ANALYZER_BANDWIDTH,
                                               TableColumnsManager.TableColumns.get(DbConstant.TABLE_ANALYZER_BANDWIDTH))
        AnalysisDb().insert_data_into_table(DbConstant.TABLE_ANALYZER_BANDWIDTH, band_width_data)
        AnalysisDb().create_table_with_headers(DbConstant.TABLE_ANALYZER_MATRIX,
                                               TableColumnsManager.TableColumns.get(DbConstant.TABLE_ANALYZER_MATRIX))
        AnalysisDb().insert_data_into_table(DbConstant.TABLE_ANALYZER_MATRIX, matrix_data)
        AnalysisDb().create_table_with_headers(DbConstant.TABLE_ANALYZER_TIME,
                                               TableColumnsManager.TableColumns.get(DbConstant.TABLE_ANALYZER_TIME))
        AnalysisDb().insert_data_into_table(DbConstant.TABLE_ANALYZER_TIME, time_data)
