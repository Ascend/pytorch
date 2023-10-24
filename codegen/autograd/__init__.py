import re
from typing import List, Dict

import torchgen.gen
from torchgen.code_template import CodeTemplate
from torchgen.api.autograd import NativeFunctionWithDifferentiabilityInfo, DifferentiabilityInfo

from torchgen.packaged.autograd.gen_inplace_or_view_type import gen_inplace_or_view_type_env
from torchgen.packaged.autograd.gen_autograd_functions import process_function
from codegen.gen_backend_stubs import parse_native_and_custom_yaml
from codegen.utils import CUSTOM_YAML_NAME


def parse_native_and_custom_yaml_(*args, **kwargs):
    return parse_native_and_custom_yaml(*args, **kwargs, custom_path=f'./torch_npu/csrc/aten/{CUSTOM_YAML_NAME}')


def gen_inplace_or_view_type_env_for_npu(
    fn: NativeFunctionWithDifferentiabilityInfo,
) -> Dict[str, List[str]]:
    gen_code = gen_inplace_or_view_type_env(fn)

    if len(gen_code['inplace_or_view_method_definitions']):
        gen_code['ops_headers'] = []
        method_definitions = re.sub(pattern=r"at::_ops::(\w+)::redispatch",
                                    repl=r'at_npu::redispatch::\1',
                                    string=gen_code['inplace_or_view_method_definitions'][0])
        gen_code['inplace_or_view_method_definitions'] = [method_definitions]
    return gen_code


# A temporary solution, due to op coupling, temporarily removing symint
def process_function_(info: DifferentiabilityInfo, template: CodeTemplate) -> str:
    function_codegen = process_function(info, template)
    if '_symint' in function_codegen:
        function_codegen = function_codegen.replace('_symint', '')
    return function_codegen


def apply_autograd_patches():
    torchgen.gen.parse_native_yaml = parse_native_and_custom_yaml_
    torchgen.packaged.autograd.gen_inplace_or_view_type.gen_inplace_or_view_type_env = \
        gen_inplace_or_view_type_env_for_npu
    torchgen.packaged.autograd.gen_autograd_functions.process_function = process_function_


apply_autograd_patches()
