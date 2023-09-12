import os
import re
import yaml
from pathlib import Path

from functools import wraps

import torch
import torch_npu
from .unsupport_api import unsupport_Tensor_api, unsupport_nn_api, unsupport_nested_api


cann_pytorch_version_map = {
    "6.3.RC2": ["1.8.1.post2", "1.11.0.post1", "2.0.0.rc1"],
    "6.3.RC1": ["1.8.1.post1", "1.11.0"],
    "6.1.RC1": ["1.8.1.post1", "1.11.0"],
    "6.0.1": ["1.8.1", "1.11.0.rc2"],
    "6.0.RC1": ["1.8.1", "1.11.0.rc1"]
}


def get_cann_version(ascend_home_path):
    cann_version = ""
    for dirpath, _, filenames in os.walk(os.path.realpath(ascend_home_path)):
        if cann_version:
            break
        install_files = [file for file in filenames if re.match(r"ascend_.*_install\.info", file)]
        if install_files:
            filepath = os.path.join(dirpath, install_files[0])
            with open(filepath, "r") as f:
                for line in f:
                    if line.find("version") != -1:
                        cann_version = line.strip().split("=")[-1]
                        break
    return cann_version


def cann_package_check():
    if "ASCEND_HOME_PATH" in os.environ:
        ascend_home_path = os.environ["ASCEND_HOME_PATH"]
        if not os.path.exists(ascend_home_path):
            raise Exception(f"ASCEND_HOME_PATH : {ascend_home_path} does not exist. "
                            "Please run 'source set_env.sh' in the CANN installation path.")

        # check whether environment variables are correctly configured
        if "ASCEND_OPP_PATH" not in os.environ:
            raise Exception(f"ASCEND_OPP_PATH environment variable is not set. "
                            "Please check whether the opp package has been installed. If exist, please run "
                            "'source set_env.sh' in the CANN installation path.")

        ascend_opp_path = os.environ["ASCEND_OPP_PATH"]
        if not os.path.exists(ascend_opp_path):
            raise Exception(f"ASCEND_OPP_PATH : {ascend_opp_path} does not exist. "
                            "Please check whether the opp package has been installed. If exist, please run "
                            "'source set_env.sh' in the CANN installation path.")

        ascend_runtime_path = os.path.join(ascend_home_path, "runtime")
        if not os.path.exists(ascend_runtime_path):
            raise Exception(f"ASCEND_RUNTIME_PATH : {ascend_runtime_path} does not exist. "
                            "Please check whether the runtime package has been installed. If exist, please run "
                            "'source set_env.sh' in the CANN installation path.")

        ascend_compiler_path = os.path.join(ascend_home_path, "compiler")
        if not os.path.exists(ascend_compiler_path):
            raise Exception(f"ASCEND_COMPILER_PATH : {ascend_compiler_path} does not exist. "
                            "Please check whether the compiler package has been installed. If exist, please run "
                            "'source set_env.sh' in the CANN installation path.")

        # get the cann version
        cann_version = get_cann_version(ascend_home_path)

        # check whether the CANN package version matches the pytorch version
        if cann_version in cann_pytorch_version_map and \
                torch_npu.__version__ not in cann_pytorch_version_map[cann_version]:
            print(f"Warning : CANN package version {cann_version} and PyTorch version {torch_npu.__version__} "
                  "is not matched, please check the README in repo of https://gitee.com/ascend/pytorch")
    else:
        print(f"Warning : ASCEND_HOME_PATH environment variable is not set.")


def create_wrap_func(check_func):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if check_func(*args, **kwargs):
                raise RuntimeError(f"{str(func)} is not supported in npu.")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Specific check functions
def is_tensor_npu_supported(*args, **kwargs):
    return torch.is_tensor(args[0]) and args[0].is_npu


def is_module_parameters_supported(*args, **kwargs):
    module_args = [m for m in args if isinstance(m, torch.nn.Module) and hasattr(m, "_modules")]
    module_parameters = [p for _, p in module_args[0].named_parameters()]
    return any(p.device is not None and p.device.type == "npu" for p in module_parameters)


def is_nested_tensor_npu_supported(*args, **kwargs):
    return any(torch.is_tensor(t) and t.is_npu for t in args[0])


def apply_wrap_func_to_modules(wrap_func, unsupported_modules):
    for module in unsupported_modules:
        module_name, attr_name = module.rsplit(".", 1)
        setattr(eval(module_name), attr_name, wrap_func(eval(module)))


# Apply wrap functions to specific modules
def add_intercept_methods():
    apply_wrap_func_to_modules(create_wrap_func(is_tensor_npu_supported), unsupport_Tensor_api)
    apply_wrap_func_to_modules(create_wrap_func(is_module_parameters_supported), unsupport_nn_api)
    apply_wrap_func_to_modules(create_wrap_func(is_nested_tensor_npu_supported), unsupport_nested_api)
