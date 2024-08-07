import os


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
    }

    return {"ENV_VARIABLES": collected_env_vars}