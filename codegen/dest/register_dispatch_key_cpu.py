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
import textwrap

from dataclasses import dataclass
from typing import List, Optional, Union
from typing_extensions import Literal

from codegen.context import method_with_native_function, native_function_manager
from codegen.utils import Target, map_maybe
from codegen.model import (DispatchKey, NativeFunction,
                           NativeFunctionsGroup, SchemaKind,
                           TensorOptionsArguments,
                           Argument, TensorOptionsArguments,
                           DeviceCheckType, Argument, assert_never,
                           is_cuda_dispatch_key, BackendIndex)
from codegen.api.types import (BaseCType, Binding, ConstRefCType,
                               Expr, MutRefCType, tensorT, NamedCType)
from codegen.api.signature import (CppSignature, CppSignatureGroup, kernel_signature,
                                   NativeSignature, DispatcherSignature)
import codegen.api.meta as meta
import codegen.api.cpp as cpp
import codegen.api.structured as structured
from codegen.api.translate import translate
from codegen.selective_build.selector import SelectiveBuilder
from codegen.api.types import BaseCType
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
class RegisterDispatchKeyCPU:
    backend_index: BackendIndex

    target: Union[
        Literal[Target.ANONYMOUS_DEFINITION],
        Literal[Target.NAMESPACED_DEFINITION],
        Literal[Target.NAMESPACED_DECLARATION],
        Literal[Target.REGISTRATION]
    ]

    # Selector object to determine which operators to generate
    # registration code for.
    selector: SelectiveBuilder

    # Whether or not we are actually code-genning for ROCm
    rocm: bool

    # The namespace that the kernels are written in. This is just `at::native` for in-tree kernels.
    cpp_namespace: str

    # The class that all unstructured native functions live under. This is used to improve
    # compiler error messages when a kernel writer adds a native function with the wrong signature.
    # This is only used in unstructured kernels, since structured kernels already live in a class.
    # Finally, this field is currently Optional because it is only used by external backends.
    # It would be nice if we can add the same logic to in-tree kernels too, but that requires updating
    # all of the existing kernel signatures scattered across aten/src/ATen/native.
    class_method_name: Optional[str]

    @staticmethod
    def gen_device_check(check_type: DeviceCheckType, args: List[Argument], method_name: str) -> str:
        if check_type == DeviceCheckType.NoCheck:
            return '  // No device check\n'

        device_check = 'c10::optional<Device> common_device = nullopt;\n'
        device_check += '(void)common_device; // Suppress unused variable warning\n'
        for arg in args:
            # Only tensor like arguments are eligible
            if arg.type.is_tensor_like():
                device_check += f"""
  c10::impl::check_and_update_common_device(common_device, {arg.name}, "{method_name}", "{arg.name}");"""
        return device_check

    @method_with_native_function
    def __call__(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        if isinstance(f, NativeFunctionsGroup):
            g: NativeFunctionsGroup = f
            # Note: We call gen_structured() if the operator is marked structured, regardless of the backend.
            # gen_structured() has special logic to handle auto-generated kernels.
            if g.structured:
                return self.gen_structured(g)
            else:
                return list(map_maybe(lambda f: self.gen_unstructured(f, g), g.functions()))
        elif isinstance(f, NativeFunction):
            r = self.gen_unstructured(f)
            return [] if r is None else [r]
        else:
            assert_never(f)
            return []

    def wrapper_kernel_sig(self, f: NativeFunction) -> Union[NativeSignature, DispatcherSignature]:
        # The prefix is just to ensure uniqueness. The Dispatcher API doesn't guarantee unique kernel names.
        return kernel_signature(f, self.backend_index, prefix=f'wrapper_{f.func.name.overload_name}_')

    def gen_out_inplace_wrapper(self, f: NativeFunction, g: Optional[NativeFunctionsGroup]) -> Optional[str]:
        if g is None:
            return None
        k = f.func.kind()
        if k is SchemaKind.inplace:
            copy_op = 'at::_copy_from'
        elif k is SchemaKind.out:
            copy_op = 'at::_copy_from_and_resize'
        else:
            raise AssertionError("gen_out_inplace_wrapper called on a functional op")

        sig = self.wrapper_kernel_sig(f)
        name = sig.name()

        func_res = f'{name}_tmp'
        return_names = cpp.return_names(f)
        if len(return_names) > 1:
            updates = '\n  '.join(
                f'{copy_op}(std::get<{i}>({func_res}), {ret_name});'
                for i, ret_name in enumerate(return_names))
            returns = f'{sig.returns_type().cpp_type()}({", ".join(return_names)})'
        else:
            ret_name = return_names[0]
            updates = f'{copy_op}({func_res}, {ret_name});'
            returns = ret_name

        functional_sig = self.wrapper_kernel_sig(g.functional)
        wrapper_name = sig.name()

        return f"""\
{sig.defn(name=wrapper_name)} {{
  auto {func_res} = {functional_sig.name()}({", ".join(e.expr for e in translate(sig.arguments(), functional_sig.arguments()))});
  {updates}
  return {returns};
}}
"""

    def gen_structured(self, g: NativeFunctionsGroup) -> List[str]:
        metadata = self.backend_index.get_kernel(g)
        if self.backend_index.dispatch_key == DispatchKey.Meta:
            if self.backend_index.has_kernel(g.out):
                raise ValueError("Do not explicitly specify Meta dispatch key on structured " \
                                 "functions, they will be automatically generated for you")
        elif self.backend_index.dispatch_key == DispatchKey.CompositeExplicitAutograd:
            if self.backend_index.has_kernel(g.out):
                raise ValueError("Do not explicitly specify CompositeExplicitAutograd dispatch key on structured " \
                                 "functions, they will be automatically generated for you")
        elif metadata is None or not metadata.structured:
            return list(map_maybe(lambda f: self.gen_unstructured(f, g), g.functions()))

        structured_gen = StructuredRegisterDispatchKey(
            self.backend_index,
            self.target,
            self.selector,
            self.rocm,
            self.cpp_namespace,
            self.class_method_name,
            g
        )
        return list(map_maybe(structured_gen.gen_one, g.functions()))

    def gen_unstructured(self, f: NativeFunction, g: Optional[NativeFunctionsGroup] = None) -> Optional[str]:
        with native_function_manager(f):
            inplace_meta = False
            gets_out_inplace_wrapper = False

            current_backend_index = None;

            if self.backend_index.has_kernel(f):
                current_backend_index = self.backend_index
            else:
                return None

            '''
                if (self.backend_index.dispatch_key == DispatchKey.Meta and
                        f.func.kind() is SchemaKind.inplace and
                        # Defer to composites for meta implementation
                        not f.has_composite_kernel and
                        # Inplace list operations are not supported
                        len(f.func.returns) == 1):
                    inplace_meta = True
                elif (not self.backend_index.use_out_as_primary and
                        g is not None
                        and gets_generated_out_inplace_wrapper(f, g, self.backend_index)):
                    # We want to generate inplace/out wrappers, that don't have a kernel for the backend.
                    gets_out_inplace_wrapper = True
                else:
                    return None
            '''
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
            if 'quant' in current_backend_index.get_kernel(f).kernel:
                return None
            if '_th_' in current_backend_index.get_kernel(f).kernel:
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
                    if f.func.arguments.self_arg is None:
                        raise ValueError("f.func.arguments.self_arg is None")
                    self_arg_name = f.func.arguments.self_arg.argument.name
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

                metadata = current_backend_index.get_kernel(f)
                if metadata is None:
                    return None
                if self.class_method_name is None:
                    impl_name = f"at::{metadata.kernel}"
                else:
                    impl_name = f"at::native::{metadata.kernel}"

                trans_to_cpu_code, args_name = transfer_args_of_wrapper_func_to_cpu(sig, f)
                args_exprs_str = ', '.join(_ for _ in args_name)
                func_call_code = f"{impl_name}({args_exprs_str})"
                return_part = transfer_ret_of_wrapper_func_to_xla(sig, func_call_code)

                device_check = '  // No device check\n'

                device_guard = "// DeviceGuard omitted"  # default
                if f.device_guard and is_cuda_dispatch_key(current_backend_index.dispatch_key):
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           STRUCTURED
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

@dataclass(frozen=True)
class StructuredRegisterDispatchKey(RegisterDispatchKeyCPU):
    g: NativeFunctionsGroup

    def gen_class_set_output_functions(
        self, k: SchemaKind, parent_class: str, generate_super: bool
    ) -> str:
        if generate_super:
            set_output_super = f"{parent_class}::set_output_raw_strided(output_idx, sizes, strides, options, names);"
        else:
            set_output_super = ""

        def gen_set_output_function(name: str, maybe_create_proxy: bool) -> str:
            maybe_star = "*" if k is SchemaKind.functional else ""
            return f"""
void set_output_{name}(
    int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
    TensorOptions options, DimnameList names
) override {{
{textwrap.indent(self.gen_class_set_output_body(k, maybe_create_proxy), "    ")}
    if (!names.empty()) {{
      namedinference::propagate_names({maybe_star}outputs_[output_idx], names);
    }}
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
{textwrap.indent(set_output_super, "    ")}
}}
"""

        return f"""
{gen_set_output_function("strided", maybe_create_proxy=True)}
{gen_set_output_function("raw_strided", maybe_create_proxy=False)}
"""

    def gen_class_set_output_body(self, k: SchemaKind, maybe_create_proxy: bool) -> str:

        maybe_set_guard_line = maybe_set_guard = ""

        if maybe_create_proxy:
            create_proxy = """
auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
if (C10_UNLIKELY(maybe_proxy.has_value())) {
    proxy_outputs_[output_idx] = c10::ExclusivelyOwned<Tensor>(std::move(maybe_proxy).value());
}
"""
        else:
            create_proxy = ""

        if k is SchemaKind.functional:
            if self.backend_index.dispatch_key != DispatchKey.CPU:
                raise KeyError("dispatch_key != DispatchKey.CPU")
            return f"""{maybe_set_guard_line}
outputs_[output_idx] = create_out(sizes, strides, options);"""
        elif k is SchemaKind.inplace:
            return f"""{maybe_set_guard_line}
const auto& out = outputs_[output_idx].get();
check_inplace(out, sizes, options);
{create_proxy}"""
        elif k is SchemaKind.out:
            return f"""{maybe_set_guard_line}
const auto& out = outputs_[output_idx].get();
resize_out(out, sizes, strides, options);
{create_proxy}"""
        elif k is SchemaKind.mutable or k is SchemaKind.scratch:
            raise AssertionError(
                f"{k} structured operators are currently not supported"
            )
        else:
            assert_never(k)

    # returns the definition of a ctor, as well as how to construct
    # this class to a variable named op
    def gen_class_ctor(self, k: SchemaKind, class_name: str, returns: int) -> str:
        if k is SchemaKind.functional:
            return ""
        elif k is SchemaKind.inplace:
            return f"{class_name}(Tensor& self) : outputs_{{std::ref(self)}} {{}}"
        elif k is SchemaKind.out:
            out_args = ", ".join(f"Tensor& out{i}" for i in range(returns))
            out_refs = ", ".join(f"std::ref(out{i})" for i in range(returns))
            return f"{class_name}({out_args}) : outputs_{{ {out_refs} }} {{}}"
        elif k is SchemaKind.mutable or k is SchemaKind.scratch:
            raise AssertionError(
                f"{k} structured operators are currently not supported"
            )
        else:
            assert_never(k)
        return ""

    def gen_class(
        self,
        f: NativeFunction,
        k: SchemaKind,
        *,
        class_name: str,
        parent_class: str,
        generate_super: bool,
    ) -> str:
        if k is SchemaKind.functional:
            output_type = "c10::ExclusivelyOwned<Tensor>"
            output_value = "*outputs_[output_idx]"
            proxy_field = ""
        elif k is SchemaKind.inplace:
            output_type = "std::reference_wrapper<Tensor>"
            output_value = (
                "proxy_outputs_[output_idx].has_value() ? "
                + "**proxy_outputs_[output_idx] : outputs_[output_idx].get()"
            )
            proxy_field = "std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, "
            proxy_field += f"{len(f.func.returns)}> proxy_outputs_;"
        elif k is SchemaKind.out:
            output_type = "std::reference_wrapper<Tensor>"
            output_value = (
                "proxy_outputs_[output_idx].has_value() ? "
                + "**proxy_outputs_[output_idx] : outputs_[output_idx].get()"
            )
            proxy_field = "std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, "
            proxy_field += f"{len(f.func.returns)}> proxy_outputs_;"

        guard_field = ""
        indent = " " * 4
        class_ctor_str = self.gen_class_ctor(k, class_name, len(f.func.returns))
        lines = (
            f"struct {class_name} final : public {parent_class} {{",
            f"{textwrap.indent(class_ctor_str, indent)}",
            f"{textwrap.indent(self.gen_class_set_output_functions(k, parent_class, generate_super), indent)}",
            "    const Tensor& maybe_get_output(int64_t output_idx) override {",
            f"      return {output_value};\n",
            "    }",
            f"    std::array<{output_type}, {len(f.func.returns)}> outputs_;",
            f"{textwrap.indent(proxy_field, indent)}",
            f"{textwrap.indent(guard_field, indent)}",
            "};",
        )
        return "\n".join(line for line in lines if line)

    @method_with_native_function
    def gen_one(self, f: NativeFunction) -> Optional[str]:
        if f.manual_kernel_registration:
            raise ValueError("f.manual_kernel_registration")

        if self.target is Target.REGISTRATION and not self.selector.is_native_function_selected(f):
            return None

        if self.backend_index.dispatch_key == DispatchKey.CompositeExplicitAutograd and f.func.kind() is SchemaKind.out:
            # Never generate a default implementation for out, that's what you
            # have to define as a backend implementor
            return None

        # Note [Direct dispatch bindings]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Signature of the non-dispatched function we'll expose in a header
        # (e.g., at::cpu::add).  We don't generate methods
        cpp_sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)

        # Signature of the wrapper function we'll register to the dispatcher
        sig = NativeSignature(f.func, prefix="wrapper_")

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

            k = f.func.kind()

            # Construct the body of the wrapper function with signature sig
            sig_body = []
            # We'll use context to keep track of any variables we've brought
            # into scope while generating code
            context: List[Union[Binding, Expr]] = list(sig.arguments())

            # Initialize the class corresponding to this structured
            # operator; feeding it the output argument(s) if it is known
            metadata = self.backend_index.get_kernel(self.g)
            if metadata is None:
                raise ValueError("metadata is None")
            class_name = f"structured_{metadata.kernel}_{k.name}"
            parent_class = f"at::native::structured_{metadata.kernel}"

            if is_cuda_dispatch_key(self.backend_index.dispatch_key):
                device_check_args = itertools.chain(
                    f.func.arguments.out,
                    f.func.arguments.flat_positional
                )
                sig_body.append(
                    RegisterDispatchKeyCPU.gen_device_check(f.device_check, list(device_check_args), sig.name()))

            trans_to_cpu_code, args_name = transfer_args_of_wrapper_func_to_cpu(sig, f)
            sig_body.append(trans_to_cpu_code)

            if k is SchemaKind.functional:
                sig_body.append(f"{class_name} op;")
            elif k is SchemaKind.inplace:
                sig_body.append(f"{class_name} op(self_cpu);")
            elif k is SchemaKind.out:
                out_args_arr = []
                for a in f.func.arguments.out:
                    out_args_arr.append(f"{a.name}_cpu")
                for arg in out_args_arr:
                    if arg in args_name:
                        args_name.remove(arg)
                out_args_str = ', '.join(out_args_arr)
                sig_body.append(f"{class_name} op({out_args_str});")

            # Translate the input native arguments into structured
            # arguments for the meta call
            meta_exprs = ', '.join(args_name)

            if self.g.out.precomputed:
                # If this function group has precomputed elements, the meta function
                # returns a struct containing them which must be saved so that it
                # can be unpacked when generating code to call the impl.
                sig_body.append(f"auto precompute = op.meta({meta_exprs});")

                # Put all of the contents of the precompute struct into the context
                # so that translate will be able to return the correct args for the
                # call to the impl.
                precomputed_values = [
                    *self.g.out.precomputed.replace.values(),
                    self.g.out.precomputed.add,
                ]
                for precomputed_elems in precomputed_values:
                    for arg in precomputed_elems:
                        context.append(
                            Expr(
                                expr=f"precompute.{arg.name}",
                                type=structured.argument_type(arg, binds=arg.name),
                            )
                        )

                # Add a use of the precompute struct so FB internal compilers don't
                # complain that there is an unused variable.
                sig_body.append("(void)precompute;")
            else:
                sig_body.append(f"op.meta({meta_exprs});")


            # After running meta, op.outputs_ is guaranteed to be valid;
            # add it to the context
            out_args = structured.out_arguments(self.g)
            for i, out_arg in enumerate(out_args):
                if ConstRefCType(BaseCType(tensorT)) != out_arg.nctype.type:
                    raise ValueError("ConstRefCType(BaseCType(tensorT)) != out_arg.nctype.type")

                if k is SchemaKind.out:
                    expr = f"op.maybe_get_output({i})"
                else:
                    maybe_star = "*" if k is SchemaKind.functional else ""
                    expr = f"{maybe_star}op.outputs_[{i}]"

                context.append(
                    Expr(
                        expr=expr,
                        type=NamedCType(
                            out_arg.nctype.name, MutRefCType(BaseCType(tensorT))
                        ),
                    )
                )

            # With the expanded context, do the impl call (if not a meta
            # function)
            if self.backend_index.dispatch_key == DispatchKey.CompositeExplicitAutograd:
                out_sig_group = CppSignatureGroup.from_native_function(
                    self.g.out, method=False, fallback_binding=f.manual_cpp_binding)
                out_sig = out_sig_group.most_faithful_signature()
                api_name = out_sig.name()
                out_exprs = ', '.join(
                    e.expr for e in translate(
                        context,
                        out_sig.arguments(),
                        method=False
                    )
                )
                sig_body.append(f"at::{api_name}({out_exprs});")
            elif self.backend_index.dispatch_key != DispatchKey.Meta:
                impl_arr = [e.expr for e in translate(context,
                                                        structured.impl_arguments(self.g),
                                                        method=False)]
                tmp_arr = []
                for x in impl_arr:
                    if '.' not in x and x not in args_name:
                        tmp_arr.append(f"{x}_cpu")
                    else:
                        tmp_arr.append(f"{x}")

                sig_body.append(f"op.impl({', '.join(tmp_arr)});")

            # Destructively return the final tensors
            if k is SchemaKind.functional:
                if len(f.func.returns) == 1:
                    ret_expr = "std::move(op.outputs_[0]).take()"  # small optimization
                else:
                    moved = ', '.join(f"std::move(op.outputs_[{i}]).take()" for i in range(len(f.func.returns)))
                    ret_expr = f"std::make_tuple({moved})"
            elif k is SchemaKind.inplace:
                ret_expr = "self"
            elif k is SchemaKind.out:
                if len(f.func.returns) == 1:
                    ret_expr = f.func.arguments.out[0].name
                else:
                    refs = ', '.join(a.name for a in f.func.arguments.out)
                    ret_expr = f"std::forward_as_tuple({refs})"

            ret_code = transfer_ret_of_wrapper_func_to_xla(sig, ret_expr)
            sig_body.append(f"{ret_code}")

            sig_body_str = "\n".join(sig_body)

            # For an overview of what this template code looks like.
            return f"""\
{self.gen_class(
f, k,
class_name=class_name,
parent_class=parent_class,
generate_super=self.g.out.structured_inherits is not None
)}

{sig.defn()} {{
{sig_body_str}
}}
"""

        elif self.target is Target.REGISTRATION:
            return f'm.impl("{f.func.name}", TORCH_FN({sig.name()}));'
        else:
            assert_never(self.target)
            # Silence mypy's "Missing return statement" error
            return None
