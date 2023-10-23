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

from codegen.utils import gen_custom_yaml_path

from .gen_autograd_functions import gen_autograd_functions_lib
from .gen_variable_type import (
    gen_variable_type, gen_npu_variable_type,
    gen_variable_type_head
)
from .gen_inplace_or_view_type import gen_inplace_or_view_type
from .gen_variable_factories import gen_variable_factories
from .utils import parse_derivatives, filt_npu_autograd_functions


def gen_autograd(
    native_functions_path: str,
    out: str,
    autograd_dir: str,
    npu_native_functions_path: str
) -> None:
    npu_native_functions_path = gen_custom_yaml_path(npu_native_functions_path)
    differentiability_infos, _, funcs_with_diff_infos =\
    parse_derivatives(native_functions_path, autograd_dir, npu_native_functions_path)
    torch_funcs_with_diff_infos, npu_funcs_with_diff_infos, _ = \
    filt_npu_autograd_functions(native_functions_path, funcs_with_diff_infos)
    template_path = os.path.join(autograd_dir, 'templates')

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
        description='Generate autograd C++ files script')
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
