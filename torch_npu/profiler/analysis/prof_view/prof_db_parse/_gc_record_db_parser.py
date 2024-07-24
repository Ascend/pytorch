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

from ...prof_common_func._db_manager import DbManager
from ...prof_common_func._constant import Constant, DbConstant, TableColumnsManager, print_error_msg
from ...prof_parse._fwk_file_parser import FwkFileParser
from .._base_parser import BaseParser

__all__ = []


class GCRecordDbParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._conn = None
        self._cur = None
        self._db_path = ""
        self._gc_record_data = []

    def run(self, deps_data: dict):
        try:
            self._db_path = deps_data.get(Constant.DB_PARSER, "")
            self.init_db_connect()
            self._gc_record_data = FwkFileParser(self._profiler_path).get_gc_record_db_data()
            self.save_gc_record_data_to_db()
        except Exception:
            print_error_msg("Failed to generate gc record table.")
            DbManager.destroy_db_connect(self._conn, self._cur)
            return Constant.FAIL, None
        return Constant.SUCCESS, None

    def init_db_connect(self) -> None:
        self._conn, self._cur = DbManager.create_connect_db(self._db_path)
        if not (self._conn and self._cur):
            raise RuntimeError(f"Failed to connect to db file: {self._db_path}")

    def save_gc_record_data_to_db(self):
        if self._gc_record_data:
            DbManager.create_table_with_headers(self._conn, self._cur, DbConstant.TABLE_GC_RECORD, TableColumnsManager.TableColumns.get(DbConstant.TABLE_GC_RECORD))
            DbManager.insert_data_into_table(self._conn, DbConstant.TABLE_GC_RECORD, self._gc_record_data)
        DbManager.destroy_db_connect(self._conn, self._cur)
