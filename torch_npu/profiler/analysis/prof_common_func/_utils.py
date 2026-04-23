import os
import functools
from ._constant import print_error_msg, print_warn_msg
from ._constant import Constant

def collect_env_vars():
    collected_env_vars = {
        "ASCEND_GLOBAL_LOG_LEVEL": os.environ.get("ASCEND_GLOBAL_LOG_LEVEL", ""),
        "HCCL_RDMA_TC": os.environ.get("HCCL_RDMA_TC", ""),
        "HCCL_RDMA_SL": os.environ.get("HCCL_RDMA_SL", ""),
        "ACLNN_CACHE_LIMIT": os.environ.get("ACLNN_CACHE_LIMIT", ""),
        "HOST_CACHE_CAPACITY": os.environ.get("HOST_CACHE_CAPACITY", ""),
        "ASCEND_ENHANCE_ENABLE": os.environ.get("ASCEND_ENHANCE_ENABLE", ""),
        "PYTORCH_NPU_ALLOC_CONF": os.environ.get("PYTORCH_NPU_ALLOC_CONF", ""),
        "ASCEND_LAUNCH_BLOCKING": os.environ.get("ASCEND_LAUNCH_BLOCKING", ""),
        "HCCL_ALGO": os.environ.get("HCCL_ALGO", ""),
    }

    return {"ENV_VARIABLES": collected_env_vars}


def check_msprof_env() -> bool:
    """Check if msprof environment variables are set."""
    if hasattr(check_msprof_env, '_called'):
        return check_msprof_env._cached_result

    check_msprof_env._called = True

    static_env = os.getenv(Constant.MSPROF_STATIC_ENV)
    dynamic_env = os.getenv(Constant.MSPROF_DYNAMIC_ENV)

    if static_env:
        print_warn_msg("If performance data is collected using the 'msprof' tool. "
                       "torch_npu profiler api is not effective.")

    if dynamic_env:
        print_warn_msg(
            f"Environment variable '{Constant.MSPROF_DYNAMIC_ENV}' is set. "
            f"torch_npu profiler api is not effective. "
            f"Please execute 'unset {Constant.MSPROF_DYNAMIC_ENV}'.")

    check_msprof_env._cached_result = not (static_env or dynamic_env)
    return check_msprof_env._cached_result

def no_exception_func(default_ret=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
            except Exception as ex:
                print_error_msg(f"Call {func.__name__} failed. Exception: {str(ex)}")
                return default_ret
            return result
        return wrapper
    return decorator
