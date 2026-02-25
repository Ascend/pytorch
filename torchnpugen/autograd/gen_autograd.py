"""
To run this file by hand from the root of the PyTorch
repository, run:

python -m torchnpugen.autograd.gen_autograd \
       --npu_native_function_dir="./torch_npu/csrc/aten/npu_native_functions.yaml" \
       --out_dir=$OUTPUT_DIR \
       --autograd_dir="./torchnpugen/autograd/"

Where $OUTPUT_DIR is where you would like the files to be
generated.  In the full build system, OUTPUT_DIR is
torch_npu/csrc/aten/
"""

# gen_autograd.py generates C++ autograd functions and Python bindings.
#
# It delegates to the following scripts:
#  gen_variable_type.py: generates VariableType.h which contains all tensor methods


import argparse
import os

from torchgen.packaged.autograd.gen_inplace_or_view_type import gen_inplace_or_view_type
from torchgen.packaged.autograd.gen_autograd_functions import gen_autograd_functions_lib

import torchnpugen
from torchnpugen.utils import get_torchgen_dir, gen_custom_yaml_path
from .gen_variable_type import (
    gen_variable_type, gen_variable_type_head
)
from .gen_variable_factories import gen_variable_factories
from .gen_autograd_functions import gen_autograd_functions_python
from .utils import parse_derivatives, filt_npu_autograd_functions


def gen_autograd(
    native_functions_path: str,
    tags_path: str,
    out: str,
    autograd_dir: str,
    npu_native_functions_path: str
) -> None:
    npu_native_functions_path = gen_custom_yaml_path(npu_native_functions_path)
    differentiability_infos, native_funcs, funcs_with_diff_infos =\
        parse_derivatives(native_functions_path, tags_path, autograd_dir, npu_native_functions_path)
    npu_funcs_with_diff_infos, _, _ = filt_npu_autograd_functions(native_functions_path, funcs_with_diff_infos)

    env_aclnn_extension_switch = os.getenv('ACLNN_EXTENSION_SWITCH')
    if env_aclnn_extension_switch:
        # if apply aclnn extension
        torchnpugen_root = os.path.dirname(torchnpugen.__file__)
        template_path = os.path.join(torchnpugen_root, "autograd", "templates")
    else:
        # original code logic
        template_path = os.path.join(autograd_dir, 'templates')

    torch_template_path = os.path.join(get_torchgen_dir(), 'packaged/autograd/templates')

    # Generate VariableType.cpp
    gen_variable_type(out, funcs_with_diff_infos, template_path)

    # Generate VariableType.h
    gen_variable_type_head(out, funcs_with_diff_infos, template_path)

    # Generate ADInplaceOrViewType.cpp
    gen_inplace_or_view_type(out, native_functions_path, tags_path, npu_funcs_with_diff_infos, template_path)

    # Generate Functions.h/cpp
    gen_autograd_functions_lib(out, differentiability_infos, template_path)

    # Generate variable_factories.h
    gen_variable_factories(out, torch_template_path, native_funcs)

    # Generate python_functions.h/cpp
    gen_autograd_functions_python(out, differentiability_infos, template_path)


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
