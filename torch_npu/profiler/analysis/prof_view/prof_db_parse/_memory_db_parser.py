import os

from enum import Enum
from collections import namedtuple
from ...prof_parse._fwk_file_parser import FwkFileParser
from ...prof_view._memory_prepare_parser import MemoryPrepareParser
from ...prof_common_func._db_manager import DbManager
from ...prof_common_func._id_manager import Str2IdManager
from ...prof_common_func._path_manager import ProfilerPathManager
from ...prof_parse._cann_file_parser import CANNFileParser, CANNDataEnum
from ...prof_common_func._constant import Constant, DbConstant, TableColumnsManager, print_error_msg
from .._base_parser import BaseParser

__all__ = []


class MemoryRecordTableRow(Enum):
    COMPONENT = 0
    TIME_STAMP = 1
    TOTAL_ALLOCATED = 2
    TOTAL_RESERVED = 3
    TOTAL_ACTIVATE = 4
    STREAM_PTR = 5
    DEVICE_ID = 6


class OpMemoryTableRow(Enum):
    NAME = 0
    SIZE = 1
    ALLOCATION_TIME = 2
    RELEASE_TIME = 3
    ACTIVE_RELEASE_TIME = 4
    ACTIVE_DURATION = 5
    DURATION = 6
    ALLOCATION_TOTAL_ALLOCATED = 7
    ALLOCATION_TOTAL_RESERVED = 8
    ALLOCATION_TOTAL_ACTIVE = 9
    RELEASE_TOTAL_ALLOCATED = 10
    RELEASE_TOTAL_RESERVED = 11
    RELEASE_TOTAL_ACTIVE = 12
    STREAM_PTR = 13
    DEVICE_ID = 14


class GeOpMemRecordsOri(Enum):
    NAME = 0
    ADDR = 1
    TYPE = 2
    SIZE = 3
    TIME_STAMP = 4
    TOTAL_ALLOCATED = 5
    TOTAL_RESERVED = 6
    DEVICE_ID = 7


class MemoryDbParser(BaseParser):
    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._conn = None
        self._cur = None
        self._db_path = ""
        self._pta_op_memory_data = []
        self._ge_op_memory_data = []
        self._pta_memory_bean_list = []
        self._pta_record_list = []
        self._ge_record_list = []
        self._record_list = []

    @staticmethod
    def _combine_record(last_record, cur_record):
        pta_ge_record_list = cur_record[:]
        pta_ge_record_list[MemoryRecordTableRow.COMPONENT.value] = Str2IdManager().get_id_from_str(Constant.PTA_GE)
        if last_record:
            pta_ge_record_list[MemoryRecordTableRow.TOTAL_ALLOCATED.value] = last_record[MemoryRecordTableRow.TOTAL_ALLOCATED.value] + cur_record[MemoryRecordTableRow.TOTAL_ALLOCATED.value]
            pta_ge_record_list[MemoryRecordTableRow.TOTAL_RESERVED.value] = last_record[MemoryRecordTableRow.TOTAL_RESERVED.value] + cur_record[MemoryRecordTableRow.TOTAL_RESERVED.value]
            pta_ge_record_list[MemoryRecordTableRow.TOTAL_ACTIVATE.value] = last_record[MemoryRecordTableRow.TOTAL_ACTIVATE.value] + cur_record[MemoryRecordTableRow.TOTAL_ACTIVATE.value]
            pta_ge_record_list[MemoryRecordTableRow.STREAM_PTR.value] = cur_record[MemoryRecordTableRow.STREAM_PTR.value] if cur_record[MemoryRecordTableRow.STREAM_PTR.value] else last_record[MemoryRecordTableRow.STREAM_PTR.value]
        return [cur_record, pta_ge_record_list]
    
    def run(self, deps_data: dict):
        try:
            self._db_path = deps_data.get(Constant.DB_PARSER, "")
            self.init_db_connect()
            self.set_start_string_id()
            self._pta_op_memory_data = deps_data.get(Constant.MEMORY_PREPARE, {}).get("memory_data", [])
            self._pta_memory_bean_list = deps_data.get(Constant.MEMORY_PREPARE, {}).get("pta_record_list", [])
            self.init_pta_memory_data()
            self.save_memory_data_to_db()
        except Exception:
            print_error_msg("Failed to generate memory_record table or op_memory table.")
            DbManager.destroy_db_connect(self._conn, self._cur)
            return Constant.FAIL, None
        return Constant.SUCCESS, None
    
    def init_db_connect(self):
        self._conn, self._cur = DbManager.create_connect_db(self._db_path)
        if not (self._conn and self._cur):
            raise RuntimeError(f"Failed to connect to db file: {self._db_path}")

    def set_start_string_id(self):
        Str2IdManager().set_start_id(DbConstant.START_STRING_ID_MEMORY)

    def get_ge_memory_data(self):
        if not DbManager.judge_table_exist(self._cur, DbConstant.TABLE_NPU_OP_MEM):
            return
        sql = "select operatorName, addr, type, size, timestampNs, totalAllocate, totalReserve, deviceId from {}".format(DbConstant.TABLE_NPU_OP_MEM)
        ge_mem_records = DbManager.fetch_all_data(self._cur, sql)
        record_type_dict = {}
        for index, mem_record in enumerate(ge_mem_records):
            if ge_mem_records[index][GeOpMemRecordsOri.TYPE.value] in record_type_dict:
                if record_type_dict[ge_mem_records[index][GeOpMemRecordsOri.TYPE.value]] == "release":
                    record = list(ge_mem_records[index])
                    record[GeOpMemRecordsOri.SIZE.value] *= -1
                    ge_mem_records[index] = tuple(record)
                continue
            sql = "select value from {} where id = {}".format(DbConstant.TABLE_STRING_IDS, ge_mem_records[index][GeOpMemRecordsOri.TYPE.value])
            record_type = DbManager.fetch_one_data(self._cur, sql)
            if record_type and record_type[0]:
                if record_type[0] == "release":
                    record = list(ge_mem_records[index])
                    record[GeOpMemRecordsOri.SIZE.value] *= -1
                    ge_mem_records[index] = tuple(record)
                record_type_dict[ge_mem_records[index][GeOpMemRecordsOri.TYPE.value]] = record_type[0]
            else:
                raise RuntimeError("Failed to find ge memory record type")
        self.get_ge_record_list(ge_mem_records)
        self.get_ge_op_mem_data(ge_mem_records)

    def get_ge_record_list(self, ge_mem_records: list):
        if not ge_mem_records:
            return
        component_id = Str2IdManager().get_id_from_str(Constant.GE)
        for record in ge_mem_records:
            ge_record = [component_id, record[GeOpMemRecordsOri.TIME_STAMP.value],
                         record[GeOpMemRecordsOri.TOTAL_ALLOCATED.value], record[GeOpMemRecordsOri.TOTAL_RESERVED.value],
                         0, None, record[GeOpMemRecordsOri.DEVICE_ID.value]]
            self._ge_record_list.append(ge_record)

    def get_ge_op_mem_data(self, ge_mem_records: list):
        if not ge_mem_records:
            return
        allocated_datas = {}
        operator_key = namedtuple('operator_key', ['operator', 'addr', 'device_id'])
        operator_value = namedtuple('operator_value', ['size', 'timestamp', 'total_allocate', 'total_reserve'])
        for mem_record in ge_mem_records:
            if mem_record[GeOpMemRecordsOri.SIZE.value] > 0:
                record_key = operator_key(operator=mem_record[GeOpMemRecordsOri.NAME.value], addr=mem_record[GeOpMemRecordsOri.ADDR.value], device_id=mem_record[GeOpMemRecordsOri.DEVICE_ID.value])
                record_value = operator_value(size=mem_record[GeOpMemRecordsOri.SIZE.value],
                                              timestamp=mem_record[GeOpMemRecordsOri.TIME_STAMP.value],
                                              total_allocate=mem_record[GeOpMemRecordsOri.TOTAL_ALLOCATED.value],
                                              total_reserve=mem_record[GeOpMemRecordsOri.TOTAL_RESERVED.value])
                allocated_datas[record_key] = record_value
            elif mem_record[GeOpMemRecordsOri.SIZE.value] < 0:
                record_key = operator_key(operator=mem_record[GeOpMemRecordsOri.NAME.value], addr=mem_record[GeOpMemRecordsOri.ADDR.value], device_id=mem_record[GeOpMemRecordsOri.DEVICE_ID.value])
                record_value = operator_value(size=mem_record[GeOpMemRecordsOri.SIZE.value],
                                              timestamp=mem_record[GeOpMemRecordsOri.TIME_STAMP.value],
                                              total_allocate=mem_record[GeOpMemRecordsOri.TOTAL_ALLOCATED.value],
                                              total_reserve=mem_record[GeOpMemRecordsOri.TOTAL_RESERVED.value])
                if record_key in allocated_datas:
                    allocated_data = allocated_datas[record_key]
                    ge_op_mem = [record_key.operator, allocated_data.size, allocated_data.timestamp,
                                 record_value.timestamp, None, int(record_value.timestamp) - int(allocated_data.timestamp),
                                 None, allocated_data.total_allocate, allocated_data.total_reserve,
                                 None, record_value.total_allocate, record_value.total_reserve,
                                 None, None, record_key.device_id]
                    self._ge_op_memory_data.append(ge_op_mem)
                    allocated_datas.pop(record_key)
        if len(allocated_datas) > 0:
            for key, value in allocated_datas.items():
                self._ge_op_memory_data.append([key.operator, value.size, value.timestamp,
                                                None, None, None,
                                                None, value.total_allocate, value.total_reserve,
                                                None, None, None,
                                                None, None, key.device_id])

    def save_op_memory_data_to_db(self):
        if not self._pta_op_memory_data and not self._ge_op_memory_data:
            return
        for memory in self._pta_op_memory_data:
            memory[OpMemoryTableRow.NAME.value] = Str2IdManager().get_id_from_str(memory[OpMemoryTableRow.NAME.value])
        DbManager.create_table_with_headers(self._conn, self._cur, DbConstant.TABLE_OPERATOR_MEMORY, TableColumnsManager.TableColumns.get(DbConstant.TABLE_OPERATOR_MEMORY))
        DbManager.insert_data_into_table(self._conn, DbConstant.TABLE_OPERATOR_MEMORY, self._pta_op_memory_data + self._ge_op_memory_data)

    def get_pta_memort_record_list(self):
        if not self._pta_memory_bean_list:
            return
        for memory_bean in self._pta_memory_bean_list:
            self._pta_record_list.append([Str2IdManager().get_id_from_str(Constant.PTA), memory_bean.time_ns,
                                          memory_bean.total_allocated_for_db, memory_bean.total_reserved_for_db,
                                          memory_bean.total_active_for_db, memory_bean.stream_ptr, memory_bean.device_index])
    
    def get_pta_ge_record_list(self):
        """
        ge records are to be sorted firstly and pta records are already sorted,
        then generate ge+pta records
        """
        try:
            if self._ge_record_list:
                self._ge_record_list.sort(key=lambda x: x[1])
        except Exception as e:
            raise RuntimeError(f"Can't sort records for cann memory record") from e
        ge_ptr = 0
        pta_ptr = 0
        last_ge_record = None
        last_pta_record = None
        while ge_ptr < len(self._ge_record_list) and pta_ptr < len(self._pta_record_list):
            ge_record = self._ge_record_list[ge_ptr]
            pta_record = self._pta_record_list[pta_ptr]
            if ge_record[1] >= pta_record[1]:
                self._record_list.extend(self._combine_record(last_ge_record, pta_record))
                pta_ptr += 1
                last_pta_record = pta_record
            else:
                self._record_list.extend(self._combine_record(last_pta_record, ge_record))
                ge_ptr += 1
                last_ge_record = ge_record
        while ge_ptr < len(self._ge_record_list):
            ge_record = self._ge_record_list[ge_ptr]
            self._record_list.extend(self._combine_record(last_pta_record, ge_record))
            ge_ptr += 1
        while pta_ptr < len(self._pta_record_list):
            pta_record = self._pta_record_list[pta_ptr]
            self._record_list.extend(self._combine_record(last_ge_record, pta_record))
            pta_ptr += 1

    def save_memory_record_data_to_db(self):
        self.get_pta_memort_record_list()
        self.get_pta_ge_record_list()
        if not self._record_list:
            return
        DbManager.create_table_with_headers(self._conn, self._cur, DbConstant.TABLE_MEMORY_RECORD, TableColumnsManager.TableColumns.get(DbConstant.TABLE_MEMORY_RECORD))
        DbManager.insert_data_into_table(self._conn, DbConstant.TABLE_MEMORY_RECORD, self._record_list)

    def init_pta_memory_data(self):
        if not ProfilerPathManager.get_cann_path(self._profiler_path):
            torch_nop_node = FwkFileParser(self._profiler_path).get_torch_op_tree_node(only_fwk=True)
            deps_data = {Constant.TREE_BUILD_PARSER: torch_nop_node}
            _, pta_data = MemoryPrepareParser(Constant.MEMORY_PREPARE, self._param_dict).run(deps_data)
            self._pta_op_memory_data = pta_data.get("memory_data", [])
            self._pta_memory_bean_list = pta_data.get("pta_record_list", [])

    def save_strings_id(self):
        DbManager.create_table_with_headers(self._conn, self._cur, DbConstant.TABLE_STRING_IDS, TableColumnsManager.TableColumns.get(DbConstant.TABLE_STRING_IDS))
        DbManager.insert_data_into_table(self._conn, DbConstant.TABLE_STRING_IDS, Str2IdManager().get_all_string_2_id_data())
    
    def save_memory_data_to_db(self):
        self.get_ge_memory_data()
        self.save_memory_record_data_to_db()
        self.save_op_memory_data_to_db()
        self.save_strings_id()
        DbManager.destroy_db_connect(self._conn, self._cur)