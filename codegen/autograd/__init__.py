import re
from typing import List, Dict

import torchgen.gen
from torchgen.api.autograd import NativeFunctionWithDifferentiabilityInfo

import codegen
from codegen.torch_autograd.gen_inplace_or_view_type import gen_inplace_or_view_type_env
from codegen.gen_backend_stubs import parse_native_and_custom_yaml


def parse_native_and_custom_yaml_(*args, **kwargs):
    return parse_native_and_custom_yaml(*args, **kwargs, custom_path='./torch_npu/csrc/aten/npu_native_functions.yaml')


def gen_inplace_or_view_type_env_for_npu(
    fn: NativeFunctionWithDifferentiabilityInfo,
) -> Dict[str, List[str]]:
    gen_code = gen_inplace_or_view_type_env(fn)
    
    if len(gen_code['inplace_or_view_method_definitions']):
        gen_code['ops_headers'] = []      
        method_definitions = re.sub(pattern=r"at::_ops::(\w+)::redispatch",
                                    repl=r'op_plugin::\1',
                                    string=gen_code['inplace_or_view_method_definitions'][0])
        method_definitions = method_definitions.replace('ks & c10::after_ADInplaceOrView_keyset, ', '')
        gen_code['inplace_or_view_method_definitions'] = [method_definitions]
    return gen_code


def apply_autograd_patches():
    torchgen.gen.parse_native_yaml = parse_native_and_custom_yaml_
    codegen.torch_autograd.gen_inplace_or_view_type.gen_inplace_or_view_type_env = gen_inplace_or_view_type_env_for_npu


apply_autograd_patches()
