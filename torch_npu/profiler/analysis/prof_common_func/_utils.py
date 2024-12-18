import os
import functools
from ._constant import print_error_msg


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
