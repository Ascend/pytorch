import os
from datetime import datetime
from typing import Union

from torch_npu.utils.error_code import ErrCode, prof_error

__all__ = []


class Constant(object):
    INVALID_VALUE = -1
    NULL_VALUE = 0
    DEFAULT_PROCESS_NUMBER = os.cpu_count() // 2

    # dir name
    FRAMEWORK_DIR = "FRAMEWORK"
    OUTPUT_DIR = "ASCEND_PROFILER_OUTPUT"
    ASCEND_WORK_PATH = "ASCEND_WORK_PATH"
    PROFILING_WORK_PATH = "profiling_data"

    # file authority
    FILE_AUTHORITY = 0o640
    DIR_AUTHORITY = 0o750
    MAX_FILE_SIZE = 1024 * 1024 * 1024 * 10
    MAX_CSV_SIZE = 1024 * 1024 * 1024 * 5
    MAX_PATH_LENGTH = 4096
    MAX_WORKER_NAME_LENGTH = 226
    MAX_FILE_NAME_LENGTH = 255
    PROF_WARN_SIZE = 1024 * 1024 * 1024

    # tlv constant struct
    CONSTANT_BYTES = "constant_bytes"
    NS_TO_US = 1000

    # field name
    SEQUENCE_UNMBER = "Sequence number"
    FORWORD_THREAD_ID = "Fwd thread id"
    OP_NAME = "op_name"
    INPUT_SHAPES = "Input Dims"
    INPUT_DTYPES = "Input type"
    CALL_STACK = "Call stack"
    MODULE_HIERARCHY = "Module Hierarchy"
    FLOPS = "flops"
    NAME = "name"
    MODULE_UID = "module_uid"
    MODULE_NAME = "module_name"

    # trace constant
    PROCESS_NAME = "process_name"
    PROCESS_LABEL = "process_labels"
    PROCESS_SORT = "process_sort_index"
    THREAD_NAME = "thread_name"
    THREAD_SORT = "thread_sort_index"
    FLOW_START_PH = "s"
    FLOW_END_PH = "f"

    # framework
    TENSORBOARD_TRACE_HANDLER = "tensorboard_trace_handler"
    EXPORT_CHROME_TRACE = "export_chrome_trace"
    EXPORT_STACK = "export_stack"

    ACL_OP_EXE_NAME = ("AscendCL@aclopCompileAndExecute".lower(), "AscendCL@aclopCompileAndExecuteV2".lower())
    AI_CORE = "AI_CORE"

    # memory
    PTA = "PTA"
    GE = "GE"
    APP = "APP"
    PTA_GE = "PTA+GE"
    B_TO_KB = 1024.0
    KB_TO_MB = 1024.0
    B_TO_MB = 1024.0 ** 2
    CANN_OP_MEM_LEN = 10
    PTA_OP_MEM_LEN = 14
    CANN_MEM_RECORD_LEN = 4
    PTA_MEM_RECORD_LEN = 5
    PTA_RECORD_TYPE_NUM = 3
    MEMORY_MALLOC = 0
    MEMORY_FREE = 1
    MEMORY_BLOCK_FREE = 2

    # profiler config
    CONFIG = "config"
    RANK_ID = "rank_id"
    COMMON_CONFIG = "common_config"
    ACTIVITIES = "activities"
    EXPERIMENTAL_CONFIG = "experimental_config"
    PROFILER_LEVEL = '_profiler_level'
    AI_CORE_METRICS = '_aic_metrics'
    L2_CACHE = '_l2_cache'
    DATA_SIMPLIFICATION = '_data_simplification'
    EXPORT_TYPE = '_export_type'
    LEVEL0 = "Level0"
    LEVEL1 = "Level1"
    LEVEL2 = "Level2"
    AicPipeUtilization = "ACL_AICORE_PIPE_UTILIZATION"
    AicArithmeticUtilization = "ACL_AICORE_ARITHMETIC_UTILIZATION"
    AicMemory = "ACL_AICORE_MEMORY_BANDWIDTH"
    AicMemoryL0 = "ACL_AICORE_L0B_AND_WIDTH"
    AicMemoryUB = "ACL_AICORE_MEMORY_UB"
    AicResourceConflictRatio = "ACL_AICORE_RESOURCE_CONFLICT_RATIO"
    AicL2Cache = "ACL_AICORE_L2_CACHE"
    AicMetricsNone = "ACL_AICORE_NONE"
    Db = "db"
    Text = "text"

    # profiler end info
    END_INFO = "end_info"
    CANN_BEGIN_TIME = "collectionTimeBegin"
    CANN_BEGIN_MONOTONIC = "clockMonotonicRaw"
    FWK_END_TIME = "collectionTimeEnd"
    FWK_END_MONOTONIC = "MonotonicTimeEnd"

    # profiler start info
    START_INFO = "start_info"
    SysCntFreq = "freq"
    StartCnt = "start_cnt"
    StartMonotonic = "start_monotonic"
    SyscntEable = "syscnt_enable"

    # metric
    METRIC_CPU_TIME = "self_cpu_time_total"
    METRIC_NPU_TIME = "self_npu_time_total"

    # prepare data
    TREE_NODE = "tree_node"
    STEP_NODE = "step_range"

    # step_range
    STEP_ID = "step_id"
    START_TS = "start_ts"
    END_TS = "end_ts"
    COMM_OPS = "comm_ops"

    # multiprocess
    MODE = "mode"
    DEPS = "deps_parser"
    SUCCESS = 0
    FAIL = 1

    # parser name
    TRACE_PRE_PARSER = "trace_prepare"
    TREE_BUILD_PARSER = "build_tree"
    CANN_EXPORT_PARSER = "export"
    CANN_TIMELINE_PARSER = "timeline"
    CANN_ANALYZE_PARSER = "analyze"
    OPERATOR_VIEW_PARSER = "operator"
    TRACE_VIEW_PARSER = "trace"
    KERNEL_VIEW_PARSER = "kernel"
    TRACE_STEP_TIME_PARSER = "step_time"
    MEMORY_VIEW_PARSER = "memory"
    INTEGRATE_PARSER = "integrate"
    COMMUNICATION_PARSER = "communication"
    RELATION_PARSER = "relation"
    STACK_VIEW_PARSER = "export_stack"
    MEMORY_PREPARE = "memory_prepare"

    DB_PARSER = "cann_db"
    FWK_API_DB_PARSER = "fwk_api_db"
    MEMORY_DB_PARSER = "memory_db"

    TRACE_VIEW_TEMP = "trace_view_temp.json"

    # db data type
    SQL_TEXT_TYPE = "TEXT"
    SQL_INTEGER_TYPE = "INTEGER"
    SQL_NUMERIC_TYPE = "NUMERIC"
    SQL_REAL_TYPE = "REAL"


def print_info_msg(message: str):
    time_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{time_str} [INFO] [{os.getpid()}] profiler.py: {message}")


def print_warn_msg(message: str):
    time_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{time_str} [WARNING] [{os.getpid()}] profiler.py: {message}")


def print_error_msg(message: str):
    time_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{time_str} [ERROR] [{os.getpid()}] profiler.py: {message}")


def convert_ns2us_float(ns) -> float:
    # convert ns to us
    if abs(ns) == float("inf"):
        return ns
    if not isinstance(ns, int):
        raise RuntimeError("Input must be integer." + prof_error(ErrCode.TYPE))
    us = float(ns / Constant.NS_TO_US)
    return us


def convert_ns2us_str(ns, tail="") -> str:
    # convert ns to us
    if abs(ns) == float("inf"):
        return str(ns)
    if not isinstance(ns, int):
        raise RuntimeError("Input must be integer." + prof_error(ErrCode.TYPE))
    ns = str(ns)
    if len(ns) <= 3:
        result = "0." + (3 - len(ns)) * "0" + ns
    else:
        result = ns[:-3] + "." + ns[-3:]
    return result + tail


def convert_us2ns(us: Union[str, float, int], tail="\t") -> int:
    # convert us to ns
    us = str(us)
    # remove \t
    us = us.strip(tail)
    int_dcm = us.split(".")
    if len(int_dcm) == 2:
        result = int(int_dcm[0] + int_dcm[1][:3] + (3 - len(int_dcm[1])) * "0")
    elif len(int_dcm) == 1:
        result = int(int_dcm[0] + 3 * "0")
    else:
        raise RuntimeError("Invalid input us!" + prof_error(ErrCode.PARAM))
    return result


def contact_2num(high_num: int, low_num: int) -> int:
    MOVE_BIT = 32
    return high_num << MOVE_BIT | low_num


class DbConstant():
    # db invalid value
    DB_INVALID_VALUE = 4294967295

    # db name
    DB_ASCEND_PYTORCH = "ascend_pytorch.db"
    DB_ANALYSIS = "analysis.db"

    TABLE_STRING_IDS = "STRING_IDS"

    # api table name
    TABLE_API = "API"
    # api info table name
    TABLE_API_INFO = "PYTORCH_API_INFO"
    # api type table name
    TABLE_ENUM_API_TYPE = "ENUM_API_TYPE"

    # api type table name
    TABLE_API_TYPE = "ENUM_API_TYPE"
    # pytorch api type
    PYTORCH_API_TYPE = "pytorch"
    # pytorch api level id
    PYTORCH_API_TYPE_ID = 30000

    # profiler data table name
    TABLE_MEMORY_RECORD = "MEMORY_RECORD"
    TABLE_OPERATOR_MEMORY = "OP_MEMORY"
    TABLE_NPU_OP_MEM = "NPU_OP_MEM"

    # analyzer table name
    TABLE_ANALYZER_BANDWIDTH = "CommAnalyzerBandwidth"
    TABLE_ANALYZER_MATRIX = "CommAnalyzerMatrix"
    TABLE_ANALYZER_TIME = "CommAnalyzerTime"

    # session info table name
    TABLE_SESSION_INFO = "SESSION_INFO"

    # analyzer table name
    TABLE_ANALYZER_BANDWIDTH = "CommAnalyzerBandwidth"
    TABLE_ANALYZER_MATRIX = "CommAnalyzerMatrix"
    TABLE_ANALYZER_TIME = "CommAnalyzerTime"
    TABLE_STEP_TRACE_TIME = "StepTraceTime"

    # pytorch start string_id
    START_STRING_ID_FWK_API = 1 << 28
    START_STRING_ID_MEMORY = 2 << 28


class TableColumnsManager():
    TableColumns = {
        DbConstant.TABLE_STRING_IDS : [
            ("id", Constant.SQL_INTEGER_TYPE),
            ("value", Constant.SQL_TEXT_TYPE)
        ],
        DbConstant.TABLE_API : [
            ("startNs", Constant.SQL_TEXT_TYPE),
            ("endNs", Constant.SQL_TEXT_TYPE),
            ("type", Constant.SQL_INTEGER_TYPE),
            ("globalTid", Constant.SQL_INTEGER_TYPE),
            ("connectionId", Constant.SQL_INTEGER_TYPE),
            ("name", Constant.SQL_INTEGER_TYPE),
            ("apiId", Constant.SQL_INTEGER_TYPE)
        ],
        DbConstant.TABLE_API_TYPE : [
            ("id", Constant.SQL_INTEGER_TYPE),
            ("name", Constant.SQL_TEXT_TYPE)
        ],
        DbConstant.TABLE_API_INFO : [
            ("apiId", Constant.SQL_INTEGER_TYPE),
            ("sequenceNumber", Constant.SQL_INTEGER_TYPE),
            ("fwdThreadId", Constant.SQL_INTEGER_TYPE),
            ("inputDtypes", Constant.SQL_INTEGER_TYPE),
            ("inputShapes", Constant.SQL_INTEGER_TYPE),
            ("stack", Constant.SQL_TEXT_TYPE)
        ],
        DbConstant.TABLE_MEMORY_RECORD : [
            ("component", Constant.SQL_TEXT_TYPE),
            ("time_stamp", Constant.SQL_TEXT_TYPE),
            ("total_allocated", Constant.SQL_NUMERIC_TYPE),
            ("total_reserved", Constant.SQL_NUMERIC_TYPE),
            ("total_active", Constant.SQL_NUMERIC_TYPE),
            ("stream_ptr", Constant.SQL_INTEGER_TYPE),
            ("device_id", Constant.SQL_INTEGER_TYPE)
        ],
        DbConstant.TABLE_OPERATOR_MEMORY : [
            ("name", Constant.SQL_TEXT_TYPE),
            ("size", Constant.SQL_INTEGER_TYPE),
            ("allocation_time", Constant.SQL_NUMERIC_TYPE),
            ("release_time", Constant.SQL_NUMERIC_TYPE),
            ("active_release_time", Constant.SQL_NUMERIC_TYPE),
            ("active_duration", Constant.SQL_NUMERIC_TYPE),
            ("duration", Constant.SQL_NUMERIC_TYPE),
            ("allocation_total_allocated", Constant.SQL_NUMERIC_TYPE),
            ("allocation_total_reserved", Constant.SQL_NUMERIC_TYPE),
            ("allocation_total_active", Constant.SQL_NUMERIC_TYPE),
            ("release_total_allocated", Constant.SQL_NUMERIC_TYPE),
            ("release_total_reserved", Constant.SQL_NUMERIC_TYPE),
            ("release_total_active", Constant.SQL_NUMERIC_TYPE),
            ("stream_ptr", Constant.SQL_INTEGER_TYPE),
            ("device_id", Constant.SQL_INTEGER_TYPE),
        ],
        DbConstant.TABLE_SESSION_INFO : [
            ("rankId", Constant.SQL_INTEGER_TYPE)
        ],
        DbConstant.TABLE_ENUM_API_TYPE : [
            ("id", Constant.SQL_INTEGER_TYPE),
            ("name", Constant.SQL_TEXT_TYPE)
        ],
        DbConstant.TABLE_ANALYZER_BANDWIDTH : [
            ("hccl_op_name", Constant.SQL_TEXT_TYPE),
            ("group_name", Constant.SQL_TEXT_TYPE),
            ("transport_type", Constant.SQL_TEXT_TYPE),
            ("transit_size", Constant.SQL_NUMERIC_TYPE),
            ("transit_time", Constant.SQL_NUMERIC_TYPE),
            ("bandwidth", Constant.SQL_NUMERIC_TYPE),
            ("large_packet_ratio", Constant.SQL_NUMERIC_TYPE),
            ("package_size", Constant.SQL_NUMERIC_TYPE),
            ("count", Constant.SQL_NUMERIC_TYPE),
            ("total_duration", Constant.SQL_NUMERIC_TYPE),
            ("step", Constant.SQL_TEXT_TYPE),
            ("type", Constant.SQL_TEXT_TYPE)
        ],
        DbConstant.TABLE_ANALYZER_MATRIX : [
            ("hccl_op_name", Constant.SQL_TEXT_TYPE),
            ("group_name", Constant.SQL_TEXT_TYPE),
            ("src_rank", Constant.SQL_TEXT_TYPE),
            ("dst_rank", Constant.SQL_TEXT_TYPE),
            ("transport_type", Constant.SQL_TEXT_TYPE),
            ("transit_size", Constant.SQL_NUMERIC_TYPE),
            ("transit_time", Constant.SQL_NUMERIC_TYPE),
            ("bandwidth", Constant.SQL_NUMERIC_TYPE),
            ("step", Constant.SQL_TEXT_TYPE),
            ("type", Constant.SQL_TEXT_TYPE),
            ("op_name", Constant.SQL_TEXT_TYPE)
        ],
        DbConstant.TABLE_ANALYZER_TIME : [
            ("hccl_op_name", Constant.SQL_TEXT_TYPE),
            ("group_name", Constant.SQL_TEXT_TYPE),
            ("start_timestamp", Constant.SQL_NUMERIC_TYPE),
            ("elapse_time", Constant.SQL_NUMERIC_TYPE),
            ("transit_time", Constant.SQL_NUMERIC_TYPE),
            ("wait_time", Constant.SQL_NUMERIC_TYPE),
            ("synchronization_time", Constant.SQL_NUMERIC_TYPE),
            ("idle_time", Constant.SQL_NUMERIC_TYPE),
            ("step", Constant.SQL_TEXT_TYPE),
            ("type", Constant.SQL_TEXT_TYPE)
        ],
        DbConstant.TABLE_STEP_TRACE_TIME : [
            ("step", Constant.SQL_TEXT_TYPE),
            ("computing", Constant.SQL_NUMERIC_TYPE),
            ("communication_not_overlapped", Constant.SQL_NUMERIC_TYPE),
            ("overlapped", Constant.SQL_NUMERIC_TYPE),
            ("communication", Constant.SQL_NUMERIC_TYPE),
            ("free", Constant.SQL_NUMERIC_TYPE),
            ("stage", Constant.SQL_NUMERIC_TYPE),
            ("bubble", Constant.SQL_NUMERIC_TYPE),
            ("communication_not_overlapped_and_exclude_receive", Constant.SQL_NUMERIC_TYPE)
        ]
    }