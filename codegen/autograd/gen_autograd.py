"""
To run this file by hand from the root of the PyTorch
repository, run:

python -m codegen.autograd.gen_autograd \
       --npu_native_function_dir="./torch_npu/csrc/aten/npu_native_functions.yaml" \
       --out_dir=$OUTPUT_DIR \
       --autograd_dir="./codegen/autograd/"

Where $OUTPUT_DIR is where you would like the files to be
generated.  In the full build system, OUTPUT_DIR is
torch_npu/csrc/aten/
"""

# gen_autograd.py generates C++ autograd functions and Python bindings.
#
# It delegates to the following scripts:
#
#  gen_autograd_functions.py: generates subclasses of torch::autograd::Node
#  gen_variable_type.py: generates VariableType.h which contains all tensor methods
#  gen_python_functions.py: generates Python bindings to THPVariable
#

import argparse
import os
from typing import List, Dict

from torchgen.model import NativeFunction, FunctionSchema
from torchgen.api.autograd import (
    match_differentiability_info, NativeFunctionWithDifferentiabilityInfo,
    DifferentiabilityInfo
)

from codegen.torch_autograd.gen_inplace_or_view_type import gen_inplace_or_view_type
from codegen.torch_autograd.load_derivatives import load_derivatives
from codegen.torch_autograd.gen_autograd_functions import gen_autograd_functions_lib
from codegen.gen_backend_stubs import parse_native_and_custom_yaml
from codegen.utils import get_torchgen_dir


from .gen_variable_type import (
    gen_variable_type, gen_npu_variable_type, 
    NPU_AUTOGRAD_FUNCTION, gen_variable_type_head
)
from .gen_variable_factories import gen_variable_factories

def gen_autograd(
    native_functions_path: str,
    tags_path: str,
    out: str,
    autograd_dir: str,
    npu_native_functions_path: str
) -> None:
    differentiability_infos, _ = load_derivatives(
        os.path.join(autograd_dir, 'derivatives.yaml'), native_functions_path, tags_path)
    template_path = os.path.join(autograd_dir, 'templates')
    torch_templace_path = os.path.join(os.path.dirname(autograd_dir), 'torch_autograd/templates')

    native_funcs = parse_native_and_custom_yaml(native_functions_path,
                                                tags_path, npu_native_functions_path).native_functions
    funcs = filte_out_native_autograd_function(native_funcs, differentiability_infos)
    funcs_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo] = []
    funcs_with_diff_infos = match_differentiability_info(funcs, differentiability_infos)

    torch_funcs_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo] = []
    npu_funcs_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo] = []
    for func in funcs_with_diff_infos:
        f = func.func
        name = str(f.func.name)
        if name in NPU_AUTOGRAD_FUNCTION:
            npu_funcs_with_diff_infos.append(func)
        else:
            torch_funcs_with_diff_infos.append(func)

    # Generate VariableType.cpp
    gen_variable_type(out, torch_funcs_with_diff_infos, template_path)
    
    # Generate VariableTypeNPU.cpp
    gen_npu_variable_type(out, npu_funcs_with_diff_infos, template_path)
    
    # Generate VariableType.h
    gen_variable_type_head(out, funcs_with_diff_infos, template_path)

    # Generate ADInplaceOrViewType.cpp
    gen_inplace_or_view_type(out, native_functions_path, tags_path, npu_funcs_with_diff_infos, torch_templace_path)
    
    # Generate Functions.h/cpp
    gen_autograd_functions_lib(out, differentiability_infos, template_path)

    # Generate variable_factories.h
    gen_variable_factories(out,torch_templace_path, native_funcs)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate autograd C++ files script')
    parser.add_argument('--out_dir', metavar='OUT',
                        help='path to output directory')
    parser.add_argument('--autograd_dir', metavar='AUTOGRAD',
                        help='path to autograd directory')
    parser.add_argument('--npu_native_function_dir', 
                        help='path to npu_native_functions.yaml')
    args = parser.parse_args()

    torchgen_path = get_torchgen_dir()

    tags_yaml_path = os.path.join(torchgen_path, 'packaged/ATen/native/tags.yaml')
    native_yaml_path = os.path.join(torchgen_path, 'packaged/ATen/native/native_functions.yaml')

    gen_autograd(native_yaml_path,
                 tags_yaml_path,
                 args.out_dir,
                 args.autograd_dir, 
                 args.npu_native_function_dir)


if __name__ == '__main__':
    main()
