# Copyright (c) 2023 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
To run this file by hand from the root of the PyTorch
repository, run:

python -m codegen.autograd.gen_autograd \
       --native_functions_dir="./codegen/native_functions.yaml" \
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
from pathlib import Path
from typing import List

from codegen.api.autograd import (
    match_differentiability_info, NativeFunctionWithDifferentiabilityInfo,
    DifferentiabilityInfo)
from codegen.model import NativeFunction
from .gen_autograd_functions import gen_autograd_functions_lib
from .gen_variable_type import (
    gen_variable_type, gen_npu_variable_type,
    gen_aclnn_variable_type, gen_variable_type_head
)
from .gen_inplace_or_view_type import gen_inplace_or_view_type
from .gen_variable_factories import gen_variable_factories
from .utils import parse_derivatives, filt_npu_autograd_functions
from .load_derivatives import load_derivatives

from codegen.utils import gen_custom_yaml_path, enable_opplugin

def gen_autograd(
    native_functions_path: str,
    out: str,
    autograd_dir: str,
    npu_native_functions_path: str
) -> None:
    npu_native_functions_path = gen_custom_yaml_path(npu_native_functions_path)
    differentiability_infos, native_funcs , funcs_with_diff_infos =\
    parse_derivatives(native_functions_path, autograd_dir, npu_native_functions_path)
    torch_funcs_with_diff_infos, npu_funcs_with_diff_infos, _ = \
    filt_npu_autograd_functions(native_functions_path, funcs_with_diff_infos)
    template_path = os.path.join(autograd_dir, 'templates')
    
    # The purpose of the following code is to handle this situation:
    # Is aclnn kernel, and only have backward function in aclnn kernel.
    aclnn_derivatives_path =  ('third_party/op-plugin/op_plugin/config/v1r11/aclnn_derivatives.yaml'
        if enable_opplugin()
        else "codegen/autograd/aclnn_derivatives.yaml")
    aclnn_differentiability_infos = load_derivatives(
        str(Path(autograd_dir).parents[1].joinpath(aclnn_derivatives_path)), 
            native_functions_path, 
            npu_native_functions_path)
    
    if aclnn_differentiability_infos:
        aclnn_funcs: List[NativeFunction] = []
        derivatives_name_list: List[str] = []
        for info in aclnn_differentiability_infos:
            derivatives_name_list.append(str(info.func.func.name))
        for funcs in native_funcs:
            func_name = str(funcs.func.name)
            func_base_name = str(funcs.func.name.name.base)
            if (func_name in derivatives_name_list) or (func_base_name in derivatives_name_list):
                aclnn_funcs.append(funcs)
        
        aclnn_funcs_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo] = []
        aclnn_funcs_with_diff_infos = match_differentiability_info(aclnn_funcs, aclnn_differentiability_infos)
        #Merge diff infos to generate header in one file.
        differentiability_infos = differentiability_infos + aclnn_differentiability_infos
        funcs_with_diff_infos.extend(aclnn_funcs_with_diff_infos)
        
        gen_aclnn_variable_type(out, aclnn_funcs_with_diff_infos, template_path)

    # Generate VariableType.h/cpp
    gen_variable_type(out, torch_funcs_with_diff_infos, template_path)

    gen_npu_variable_type(out, npu_funcs_with_diff_infos, template_path)

    gen_variable_type_head(out, funcs_with_diff_infos, template_path)

    gen_inplace_or_view_type(out, npu_funcs_with_diff_infos, template_path)
    
    # Generate Functions.h/cpp
    gen_autograd_functions_lib(out, differentiability_infos, template_path)

    # Generate variable_factories.h
    gen_variable_factories(out, native_functions_path, npu_native_functions_path, template_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate autograd C++ files')
    parser.add_argument('--native_functions_dir', metavar='NATIVE',
                        help='path to native_functions.yaml')
    parser.add_argument('--out_dir', metavar='OUT',
                        help='path to output directory')
    parser.add_argument('--autograd_dir', metavar='AUTOGRAD',
                        help='path to autograd directory')
    parser.add_argument('--npu_native_function_dir',
                        help='path to npu_native_functions.yaml')
    args = parser.parse_args()
    gen_autograd(args.native_functions_dir,
                 args.out_dir,
                 args.autograd_dir,
                 args.npu_native_function_dir)


if __name__ == '__main__':
    main()
