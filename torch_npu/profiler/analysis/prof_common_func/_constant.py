import os
from datetime import datetime
from typing import Union

from torch_npu.utils._error_code import ErrCode, prof_error
from torch_npu.utils import should_print_warning

__all__ = []


class Constant(object):
    INVALID_VALUE = -1
    NULL_VALUE = 0
    DEFAULT_PROCESS_NUMBER = os.cpu_count() // 2
    SLEEP_TIME = 0.1

    # dir name
    FRAMEWORK_DIR = "FRAMEWORK"
    OUTPUT_DIR = "ASCEND_PROFILER_OUTPUT"
    ASCEND_WORK_PATH = "ASCEND_WORK_PATH"
    PROFILING_WORK_PATH = "profiling_data"
    PROFILER_META_DATA = "profiler_metadata.json"

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
    NS_TO_MS = 1000 * 1000

    # gc record struct format
    GC_RECORD_FORMAT = "<3Q"

    # field name
    SEQUENCE_NUMBER = "Sequence number"
    FORWARD_THREAD_ID = "Fwd thread id"
    OP_NAME = "op_name"
    INPUT_DTYPES = "Input type"
    INPUT_SHAPES = "Input Dims"
    INPUT_TENSORS = "Input Tensors"
    INPUT_SCALARS = "Input Scalars"
    SCOPE = "Scope"
    CALL_STACK = "Call stack"
    MODULE_HIERARCHY = "Module Hierarchy"
    FLOPS = "flops"
    NAME = "name"
    VALUE = "value"
    MODULE_PARAM = "module parameter"
    OPTIMIZER_PARAM = "optimizer parameter"

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
    MSPROF_TX = '_msprof_tx'
    OP_ATTR = "_op_attr"
    DATA_SIMPLIFICATION = '_data_simplification'
    EXPORT_TYPE = '_export_type'
    LEVEL0 = "Level0"
    LEVEL1 = "Level1"
    LEVEL2 = "Level2"
    LEVEL_NONE = "Level_none"
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

    # incompatible_features.json
    AffectedComponent = "affectedComponent"
    AffectedComponentVersion = "affectedComponentVersion"
    Compatibility = "compatibility"
    FeatureVersion = "featureVersion"
    InfoLog = "infoLog"

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
    TASK_INFO = "task_info"
    FWK_START_TS = "fwk_step_start_ts"

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
    STEP_INFO_DB_PARSER = "step_info_db"
    COMMUNICATION_DB_PARSER = "communication_db"
    TRACE_STEP_TIME_DB_PARSER = "trace_step_time_db"
    GC_RECORD_DB_PARSER = "gc_record_db"

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
    if not should_print_warning():
        return
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
    DB_ASCEND_PYTORCH_PROFILER = "ascend_pytorch_profiler.db"
    DB_ANALYSIS = "analysis.db"

    TABLE_STRING_IDS = "STRING_IDS"
    TABLE_ENUM_API_TYPE = "ENUM_API_TYPE"

    TABLE_MSTX_EVENTS = "MSTX_EVENTS"
    # python gc record table name
    TABLE_GC_RECORD = "GC_RECORD"
    # pytorch api table name
    TABLE_PYTORCH_API = "PYTORCH_API"
    # api connection ids table name
    TABLE_CONNECTION_IDS = "CONNECTION_IDS"
    # call chain table name
    TABLE_PYTORCH_CALLCHAINS = "PYTORCH_CALLCHAINS"
    # cann api table name
    TABLE_CANN_API = "CANN_API"
    # task table name
    TABLE_TASK = "TASK"
    # communication op table name
    TABLE_COMMUNICATION_OP = "COMMUNICATION_OP"
    # compute task table name
    TABLE_COMPUTE_TASK_INFO = "COMPUTE_TASK_INFO"
    # communication task table name
    TABLE_COMMUNICATION_TASK_INFO = "COMMUNICATION_TASK_INFO"

    # profiler data table name
    TABLE_MEMORY_RECORD = "MEMORY_RECORD"
    TABLE_OPERATOR_MEMORY = "OP_MEMORY"
    TABLE_NPU_OP_MEM = "NPU_OP_MEM"
    TABLE_META_DATA = "META_DATA"
    
    # rank device map table name
    TABLE_RANK_DEVICE_MAP = "RANK_DEVICE_MAP"
    # host info
    TABLE_HOST_INFO = "HOST_INFO"

    # step time
    TABLE_STEP_TIME = "STEP_TIME"

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
        DbConstant.TABLE_CANN_API : [
            ("startNs", Constant.SQL_INTEGER_TYPE),
            ("endNs", Constant.SQL_INTEGER_TYPE),
            ("type", Constant.SQL_INTEGER_TYPE),
            ("globalTid", Constant.SQL_INTEGER_TYPE),
            ("connectionId", Constant.SQL_INTEGER_TYPE),
            ("name", Constant.SQL_INTEGER_TYPE)
        ],
        DbConstant.TABLE_STRING_IDS : [
            ("id", Constant.SQL_INTEGER_TYPE),
            ("value", Constant.SQL_TEXT_TYPE)
        ],
        DbConstant.TABLE_CONNECTION_IDS : [
            ("id", Constant.SQL_INTEGER_TYPE),
            ("connectionId", Constant.SQL_INTEGER_TYPE)
        ],
        DbConstant.TABLE_PYTORCH_CALLCHAINS : [
            ("id", Constant.SQL_INTEGER_TYPE),
            ("stack", Constant.SQL_INTEGER_TYPE),
            ("stackDepth", Constant.SQL_INTEGER_TYPE)
        ],
        DbConstant.TABLE_RANK_DEVICE_MAP : [
            ("rankId", Constant.SQL_INTEGER_TYPE),
            ("deviceId", Constant.SQL_INTEGER_TYPE)
        ],
        DbConstant.TABLE_ENUM_API_TYPE : [
            ("id", Constant.SQL_INTEGER_TYPE),
            ("name", Constant.SQL_INTEGER_TYPE)
        ],
        DbConstant.TABLE_PYTORCH_API : [
            ("startNs", Constant.SQL_TEXT_TYPE),
            ("endNs", Constant.SQL_TEXT_TYPE),
            ("globalTid", Constant.SQL_INTEGER_TYPE),
            ("connectionId", Constant.SQL_INTEGER_TYPE),
            ("name", Constant.SQL_INTEGER_TYPE),
            ("sequenceNumber", Constant.SQL_INTEGER_TYPE),
            ("fwdThreadId", Constant.SQL_INTEGER_TYPE),
            ("inputDtypes", Constant.SQL_INTEGER_TYPE),
            ("inputShapes", Constant.SQL_INTEGER_TYPE),
            ("callchainId", Constant.SQL_INTEGER_TYPE),
            ("type", Constant.SQL_INTEGER_TYPE)
        ],
        DbConstant.TABLE_MEMORY_RECORD : [
            ("component", Constant.SQL_INTEGER_TYPE),
            ("timestamp", Constant.SQL_INTEGER_TYPE),
            ("totalAllocated", Constant.SQL_INTEGER_TYPE),
            ("totalReserved", Constant.SQL_INTEGER_TYPE),
            ("totalActive", Constant.SQL_INTEGER_TYPE),
            ("streamPtr", Constant.SQL_INTEGER_TYPE),
            ("deviceId", Constant.SQL_INTEGER_TYPE)
        ],
        DbConstant.TABLE_OPERATOR_MEMORY : [
            ("name", Constant.SQL_INTEGER_TYPE),
            ("size", Constant.SQL_INTEGER_TYPE),
            ("allocationTime", Constant.SQL_INTEGER_TYPE),
            ("releaseTime", Constant.SQL_INTEGER_TYPE),
            ("activeReleaseTime", Constant.SQL_INTEGER_TYPE),
            ("duration", Constant.SQL_INTEGER_TYPE),
            ("activeDuration", Constant.SQL_INTEGER_TYPE),
            ("allocationTotalAllocated", Constant.SQL_INTEGER_TYPE),
            ("allocationTotalReserved", Constant.SQL_INTEGER_TYPE),
            ("allocationTotalActive", Constant.SQL_INTEGER_TYPE),
            ("releaseTotalAllocated", Constant.SQL_INTEGER_TYPE),
            ("releaseTotalReserved", Constant.SQL_INTEGER_TYPE),
            ("releaseTotalActive", Constant.SQL_INTEGER_TYPE),
            ("streamPtr", Constant.SQL_INTEGER_TYPE),
            ("deviceId", Constant.SQL_INTEGER_TYPE),
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
            ("communication_not_overlapped_and_exclude_receive", Constant.SQL_NUMERIC_TYPE),
            ("preparing", Constant.SQL_NUMERIC_TYPE)
        ],
        DbConstant.TABLE_HOST_INFO : [
            ('hostUid', Constant.SQL_TEXT_TYPE),
            ('hostName', Constant.SQL_TEXT_TYPE)
        ],
        DbConstant.TABLE_META_DATA : [
            ('name', Constant.SQL_TEXT_TYPE),
            ('value', Constant.SQL_TEXT_TYPE)
        ],
        DbConstant.TABLE_STEP_TIME : [
            ("id", Constant.SQL_INTEGER_TYPE),
            ("startNs", Constant.SQL_INTEGER_TYPE),
            ("endNs", Constant.SQL_INTEGER_TYPE)
        ],
        DbConstant.TABLE_GC_RECORD : [
            ("startNs", Constant.SQL_INTEGER_TYPE),
            ("endNs", Constant.SQL_INTEGER_TYPE),
            ("globalTid", Constant.SQL_INTEGER_TYPE)
        ]
    }
