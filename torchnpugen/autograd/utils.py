import os
from pathlib import Path
from typing import List, Dict
import yaml

from torchgen.model import NativeFunction, FunctionSchema
from torchgen.api.autograd import (
    match_differentiability_info, NativeFunctionWithDifferentiabilityInfo,
    DifferentiabilityInfo
)
from torchgen.packaged.autograd.load_derivatives import load_derivatives

from torchnpugen.utils import get_torchgen_dir, CUSTOM_YAML_NAME, PathManager
from torchnpugen.gen_backend_stubs import parse_native_and_custom_yaml


AUTOGRAD_BLACK_LIST = {'npu_format_cast.Tensor', 'npu_format_cast_', 'npu_format_cast_.acl_format'}

torch_npu_root = Path(__file__).parent.parent.parent
PathManager.check_directory_path_readable(torch_npu_root / "version.txt")
with open(torch_npu_root / "version.txt") as version_f:
    version = version_f.read().strip()
VERSION_PART = version.split('.')


def parse_derivatives(
    native_functions_path: str,
    tags_path: str,
    autograd_dir: str,
    npu_native_functions_path: str
):
    
    ## aclnn extension for customers:
    env_aclnn_extension_switch = os.getenv('ACLNN_EXTENSION_SWITCH')
    env_derivatives_path = os.getenv('PYTORCH_CUSTOM_DERIVATIVES_PATH')
    if env_aclnn_extension_switch and os.path.exists(env_derivatives_path):
        # if apply aclnn extension
        derivatives_path = env_derivatives_path
    elif env_aclnn_extension_switch and not os.path.exists(env_derivatives_path):
        # if apply aclnn extension but path not exists
        error_msg = f"ERROR: 环境变量PYTORCH_CUSTOM_DERIVATIVES_PATH指定的路径不存在\n指定路径：{env_derivatives_path}"
        print(error_msg, file=sys.stderr)
        raise FileNotFoundError(error_msg)
    else:
        # original code logic
        derivatives_path = str(Path(autograd_dir).parents[1].joinpath(
            f'third_party/op-plugin/op_plugin/config/v{VERSION_PART[0]}r{VERSION_PART[1]}/derivatives.yaml'
            ))    

    differentiability_infos, _ = load_derivatives(
        derivatives_path, native_functions_path, tags_path)
    native_funcs = parse_native_and_custom_yaml(native_functions_path,
                                                tags_path, npu_native_functions_path).native_functions
    funcs = filte_out_native_autograd_function(native_funcs, differentiability_infos)
    funcs_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo] = []
    funcs_with_diff_infos = match_differentiability_info(funcs, differentiability_infos)

    filt_funcs_with_diff_infos = [f for f in funcs_with_diff_infos if str(f.func.func.name) not in AUTOGRAD_BLACK_LIST]

    return (differentiability_infos, native_funcs, filt_funcs_with_diff_infos)


def filt_npu_autograd_functions(
    native_functions_path: str,
    funcs_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo]
):
    npu_funcs_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo] = []
    torch_functions = set()
    PathManager.check_directory_path_readable(native_functions_path)
    with open(native_functions_path, 'r') as f:
        es = yaml.safe_load(f)
    for e in es:
        torch_functions.add(e.get('func').split('(')[0])

    npu_autograd_functions = set()
    torch_derivatives_functions = set()
    for f in funcs_with_diff_infos:
        name = str(f.func.func.name)
        # f.info is differentiabilityinfo. Existence of variants ops with a differentiabilityinfo of none.
        if f.info and name not in torch_functions:
            npu_funcs_with_diff_infos.append(f)
            npu_autograd_functions.add(name)
        if f.info and name in torch_functions:
            torch_derivatives_functions.add(name)
    return npu_funcs_with_diff_infos, npu_autograd_functions, torch_derivatives_functions


def filte_out_native_autograd_function(
    native_funcs: List[NativeFunction],
    differentiability_infos: Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]],
):
    result: List[NativeFunction] = []
    derivatives_name_list: List[str] = []

    for diffinfo_dict in differentiability_infos.values():
        for info in diffinfo_dict.values():
            derivatives_name_list.append(str(info.func.func.name))
    for funcs in native_funcs:
        func_name = str(funcs.func.name)
        func_base_name = str(funcs.func.name.name.base)
        if (func_name in derivatives_name_list) or (func_base_name in derivatives_name_list):
            result.append(funcs)
    return result


_, NPU_AUTOGRAD_FUNCTION, TORCH_AUTOGRAD_FUNCTION = filt_npu_autograd_functions(
    str(Path(get_torchgen_dir()).joinpath('packaged/ATen/native/native_functions.yaml')),
    parse_derivatives(
        str(Path(get_torchgen_dir()).joinpath('packaged/ATen/native/native_functions.yaml')),
        str(Path(get_torchgen_dir()).joinpath('packaged/ATen/native/tags.yaml')),
        str(Path(__file__).parent),
        str(Path(__file__).parents[2].joinpath(f'torch_npu/csrc/aten/{CUSTOM_YAML_NAME}')))[-1]
)
