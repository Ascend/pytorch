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

import itertools

from dataclasses import dataclass
from typing import Optional

from torchgen.context import native_function_manager
from torchgen.utils import Target
from torchgen.model import (NativeFunction, NativeFunctionsGroup,
                            TensorOptionsArguments, Argument, assert_never,
                            is_cuda_dispatch_key,)
from torchgen.api.types import (BaseCType, Binding, ConstRefCType, TupleCType,
                                CppSignature, CppSignatureGroup,)
import torchgen.api.meta as meta
import torchgen.api.cpp as cpp
from torchgen.api.translate import translate
from torchgen.dest.register_dispatch_key import RegisterDispatchKey
from .utils import transfer_args_of_wrapper_func_to_cpu, transfer_ret_of_wrapper_func_to_xla

# Generates Register{dispatch}.cpp (e.g., RegisterCPU.cpp).
#
#   - The primary function of this file is to register all of the
#     implementations for the given dispatch key to the dispatcher,
#     so they are available for use in PyTorch.  If dispatch is
#     None, we generate schema (def) registrations and catchall
#     registrations.
#   - The secondary function of this file is to generate a wrapper
#     around functions.  In CPUType these wrappers do nothing
#     (and should be removed), but in other cases they handle
#     DeviceGuard. A small extra benefit of wrappers is they
#     are not overloaded, so they can be used in the registration
#     API without having to disambiguate which overload you want
#     (as would be the case if you directly registered native::
#     functions).
#   - The tertiary function of this file is to generate *static*
#     cpp API bindings which can be used to bypass dispatcher
#     directly to kernels, but with user-friendly cpp-style API
@dataclass(frozen=True)
class RegisterDispatchKeyCPU(RegisterDispatchKey):
    def gen_unstructured(self, f: NativeFunction, g: Optional[NativeFunctionsGroup] = None) -> Optional[str]:
        with native_function_manager(f):
            inplace_meta = False
            gets_out_inplace_wrapper = False

            if not self.backend_index.has_kernel(f):
                return None
            if f.manual_kernel_registration:
                return None

            if self.target is Target.REGISTRATION and not self.selector.is_native_function_selected(f):
                return None

            sig = self.wrapper_kernel_sig(f)

            name = sig.name()
            returns_type = sig.returns_type().cpp_type()
            args = sig.arguments()
            if True not in ['Tensor' in str(arg.type) for arg in args]:
                return None
            if 'quant' in self.backend_index.get_kernel(f).kernel:
                return None
            if '_th_' in self.backend_index.get_kernel(f).kernel:
                return None

            args_str = ', '.join(a.defn() for a in args)

            # See Note [Direct dispatch bindings]
            cpp_sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)

            if self.target is Target.NAMESPACED_DECLARATION:
                result = f"TORCH_API {cpp_sig_group.signature.decl()};\n"
                if cpp_sig_group.faithful_signature is not None:
                    result += f"TORCH_API {cpp_sig_group.faithful_signature.decl()};\n"
                return result
            elif self.target is Target.NAMESPACED_DEFINITION:
                def generate_defn(cpp_sig: CppSignature) -> str:
                    return f"""
{cpp_sig.defn()} {{
return {sig.name()}({', '.join(e.expr for e in translate(cpp_sig.arguments(), sig.arguments()))});
}}
"""
                result = generate_defn(cpp_sig_group.signature)
                if cpp_sig_group.faithful_signature is not None:
                    result += generate_defn(cpp_sig_group.faithful_signature)
                return result
            elif self.target is Target.ANONYMOUS_DEFINITION:
                # short circuit for inplace_meta
                if inplace_meta:
                    assert f.func.arguments.self_arg is not None
                    self_arg_name = f.func.arguments.self_arg.argument.name
                    # TODO: handle in place on tensor list
                    return f"""
{returns_type} {name}({args_str}) {{
  TORCH_CHECK_NOT_IMPLEMENTED({self_arg_name}.is_meta(),
    "Cannot inplace into non-meta tensor with meta tensor argument");
  return {self_arg_name};
}}
"""

                # short circuit for generated inplace/out wrappers
                if gets_out_inplace_wrapper:
                    return self.gen_out_inplace_wrapper(f, g)

                metadata = self.backend_index.get_kernel(f)
                if metadata is None:
                    return None
                if self.class_method_name is None:
                    impl_name = f"at::{metadata.kernel}"
                else:
                    impl_name = f"at::native::{metadata.kernel}"

                trans_to_cpu_code, args_name =  transfer_args_of_wrapper_func_to_cpu(sig, f)
                args_exprs_str = ', '.join(_ for _ in args_name)
                func_call_code = f"{impl_name}({args_exprs_str})"
                return_part = transfer_ret_of_wrapper_func_to_xla(sig, func_call_code)

                device_check = '  // No device check\n'

                device_guard = "// DeviceGuard omitted"  # default
                if f.device_guard and is_cuda_dispatch_key(self.backend_index.dispatch_key):
                    has_tensor_options = any(isinstance(a.argument, TensorOptionsArguments) for a in args)
                    if has_tensor_options:
                        # kernel is creating a tensor
                        device_guard = """globalContext().lazyInitCUDA();
  const DeviceGuard device_guard(device_or_default(device));"""
                    else:
                        # kernel is operating on existing tensors

                        # There is precedence for which argument we use to do
                        # device guard.  This describes the precedence order.
                        self_arg = [f.func.arguments.self_arg.argument] if f.func.arguments.self_arg is not None else []
                        candidate_args = itertools.chain(
                            self_arg,
                            f.func.arguments.out,
                            f.func.arguments.flat_positional
                        )

                        # Only tensor like arguments are eligible
                        device_of = next((f'{a.name}' for a in candidate_args if a.type.is_tensor_like()), None)
                        if device_of is not None:
                            device_guard = f"const OptionalDeviceGuard device_guard(device_of({device_of}));"

                return f"""\
namespace {{

{returns_type} {name}({args_str}) {{
  {device_check}

  {device_guard}
  {trans_to_cpu_code}
  {return_part}
}}

}} // anonymous namespace
"""

            elif self.target is Target.REGISTRATION:
                if f.manual_kernel_registration:
                    return None
                else:
                    payload = f"TORCH_FN({name})"
                    return f'm.impl("{f.func.name}",\n{payload});\n'
            else:
                assert_never(self.target)
        return ""
