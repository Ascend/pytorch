# Generates C++ functions that wrap ATen tensor factory methods to turn them into Variables.
#
# This writes one file: variable_factories.h

from torchgen.gen import FileManager
from torchgen.utils import mapMaybe
from torchgen.packaged.autograd.gen_variable_factories import (
    process_function,
    is_factory_function,
)

from .utils import NPU_AUTOGRAD_FUNCTION


def gen_variable_factories(
    out: str,
    template_path: str,
    native_functions: list
) -> None:
    factory_functions = [fn for fn in native_functions if
                         (is_factory_function(fn) and fn.func.name.name.base in NPU_AUTOGRAD_FUNCTION)]
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_with_template('variable_factories.h', 'variable_factories.h', lambda: {
        'generated_comment': '@' + f'generated from {fm.template_dir}/variable_factories.h',
        'ops_headers': [f'#include <ATen/ops/{fn.root_name}.h>' for fn in factory_functions],
        'function_definitions': list(mapMaybe(process_function, factory_functions)),
    })
