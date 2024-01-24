import os
from datetime import datetime
from typing import Union


class Constant(object):
    INVALID_VALUE = -1
    NULL_VALUE = 0

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

    # profiler config
    CONFIG = "config"
    COMMON_CONFIG = "common_config"
    EXPERIMENTAL_CONFIG = "experimental_config"
    PROFILER_LEVEL = '_profiler_level'
    AI_CORE_METRICS = '_aic_metrics'
    L2_CACHE = '_l2_cache'
    DATA_SIMPLIFICATION = '_data_simplification'
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

    TRACE_VIEW_TEMP = "trace_view_temp.json"


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
        raise RuntimeError("Input must be integer.")
    us = float(ns / Constant.NS_TO_US)
    return us


def convert_ns2us_str(ns, tail="") -> str:
    # convert ns to us
    if abs(ns) == float("inf"):
        return str(ns)
    if not isinstance(ns, int):
        raise RuntimeError("Input must be integer.")
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
        raise RuntimeError("Invalid input us!")
    return result
