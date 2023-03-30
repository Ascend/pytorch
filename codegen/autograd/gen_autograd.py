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

from .gen_variable_factories import gen_variable_factories


def gen_autograd(
    native_functions_path: str,
    out: str,
    autograd_dir: str,
    npu_native_functions_path: str
) -> None:
    template_path = os.path.join(autograd_dir, 'templates')

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
