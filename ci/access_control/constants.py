import os
from pathlib import Path

BASE_DIR = Path(__file__).absolute().parent.parent.parent
TEST_DIR = BASE_DIR / 'test'
NETWORK_OPS_DIR = BASE_DIR / 'third_party/op-plugin/test'

SLOW_TEST_BLOCKLIST = [
    'test_ops',
    'test_modules',
    'test_binary_ufuncs',
    'test_ops_fwd_gradients',
    'test_ops_gradients',
    'test_reductions',
    'test_unary_ufuncs',
    'test_ops_jit',
    'test_jit_fuser_te',
    "onnx/test_op_consistency",
    'onnx/test_pytorch_onnx_onnxruntime'
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
EXEC_TIMEOUT = os.getenv("PTA_UT_EXEC_TIMEOUT", 2000)
try:
    EXEC_TIMEOUT = int(EXEC_TIMEOUT)
except ValueError:
    EXEC_TIMEOUT = 2000
