class Constant(object):
    INVALID_VALUE = -1

    # dir name
    FRAMEWORK_DIR = "FRAMEWORK"
    OUTPUT_DIR = "ASCEND_PROFILER_OUTPUT"

    # file authority
    FILE_AUTHORITY = 0o640
    DIR_AUTHORITY = 0o750
    MAX_FILE_SIZE = 1024 * 1024 * 1024 * 10
    MAX_CSV_SIZE = 1024 * 1024 * 1024 * 5
    MAX_PATH_LENGTH = 4096
    PROF_WARN_SIZE = 1024 * 1024 * 400

    # tlv constant struct
    CONSTANT_BYTES = "constant_bytes"

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

    #profiler config
    CONFIG = "config"
    EXPERIMENTAL_CONFIG = "experimental_config"
    PROFILER_LEVEL = '_profiler_level'
    AI_CORE_METRICS = '_aic_metrics'
    L2_CACHE = '_l2_cache'
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
