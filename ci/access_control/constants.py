import os
from pathlib import Path

BASE_DIR = Path(__file__).absolute().parent.parent.parent
TEST_DIR = BASE_DIR / 'test'

# Add slow test cases here (first element must be test_ops)
SLOW_TEST_BLOCKLIST = [
    'test_ops',
    'test_modules',
    'test_binary_ufuncs',
    'test_ops_fwd_gradients',
    'test_ops_gradients',
    'test_reductions',
    'test_unary_ufuncs',
    'test_ops_jit',
    'onnx/test_op_consistency',
    'onnx/test_fx_op_consistency',
    'test_foreach',
    'dynamo/test_dynamic_shapes'
]

# exclude some not run directly test files
NOT_RUN_DIRECTLY = {
    "jit": "test_jit.py",
}

# include some files
INCLUDE_FILES = [
    'jit/test_complexity.py',
]

# default ut cmd execution timeout is 2000s
EXEC_TIMEOUT = os.getenv("PTA_UT_EXEC_TIMEOUT", "2000")
EXEC_TIMEOUT = int(EXEC_TIMEOUT) if EXEC_TIMEOUT.isdigit() else 2000
