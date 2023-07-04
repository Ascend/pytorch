import torchgen.gen
from codegen.gen_backend_stubs import parse_native_and_custom_yaml


def parse_native_and_custom_yaml_(*args, **kwargs):
    return parse_native_and_custom_yaml(*args, **kwargs, custom_path='./torch_npu/csrc/aten/npu_native_functions.yaml')


def apply_autograd_patches():
    torchgen.gen.parse_native_yaml = parse_native_and_custom_yaml_


apply_autograd_patches()
