# Copyright (c) 2020 Huawei Technologies Co., Ltd
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

from typing import List, Union, Optional

from codegen.context import with_native_function_and_index
from codegen.utils import map_maybe
from codegen.model import NativeFunction, NativeFunctionsGroup, BackendIndex
from codegen.api.signature import kernel_signature
import codegen.api.meta as meta
import codegen.api.structured as structured


@with_native_function_and_index
def gen_unstructured(f: NativeFunction, backend_index: BackendIndex) -> Optional[str]:
    sig = kernel_signature(f, backend_index)
    metadata = backend_index.get_kernel(f)
    if metadata is None:
        return None
    if "legacy::" in metadata.kernel:
        return None
    else:
        prefix = 'static' if backend_index.external else 'TORCH_API'
        return f"{prefix} {sig.decl(name=metadata.kernel)};"


@with_native_function_and_index
def gen_structured(g: NativeFunctionsGroup, backend_index: BackendIndex) -> List[str]:
    meta_name = meta.name(g)
    out_args = structured.impl_arguments(g)
    metadata = backend_index.get_kernel(g)
    if metadata is None:
        return []
    prefix = '' if backend_index.external else 'TORCH_API '
    return [f"""\
struct {prefix}structured_{metadata.kernel} : public at::meta::structured_{meta_name} {{
void impl({', '.join(a.decl() for a in out_args)});
}};
"""]


# Generates NativeFunctions.h, a list of forward declarations of all
# actual kernel definitions we keep in aten/src/ATen/native/
@with_native_function_and_index
def compute_native_function_declaration(
        g: Union[NativeFunctionsGroup, NativeFunction],
        backend_index: BackendIndex
) -> List[str]:
    metadata = backend_index.get_kernel(g)
    if isinstance(g, NativeFunctionsGroup):
        if metadata is not None and metadata.structured:
            if backend_index.external:
                # Structured hasn't been tested with external backends yet.
                raise AssertionError("Structured external backend functions are not implemented yet.")
            else:
                return gen_structured(g, backend_index)
        else:
            return list(map_maybe(lambda f: gen_unstructured(f, backend_index), g.functions()))
    else:
        x = gen_unstructured(g, backend_index)
        return [] if x is None else [x]
