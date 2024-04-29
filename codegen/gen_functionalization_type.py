from typing import List, Union

from torchgen.api import dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
    DispatcherSignature,
)
from torchgen.context import (
    with_native_function,
    with_native_function_and,
)
from torchgen.model import (
    BaseTy,
    BaseType,
    NativeFunction,
    NativeFunctionsGroup,
    SchemaKind
)

from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.gen_functionalization_type import (
    wrapper_name,
    modifies_arguments,
    unwrap_tensor_args,
    convert_to_meta_tensors,
    get_mutable_redispatch_return_names,
    return_str,
    maybe_create_output,
    return_from_mutable_noop_redispatch,
)
from .utils import NativeFunctionsGroupOptionalOut


def wrap_propagate_mutations_and_return(
    f: NativeFunction, functional_op: NativeFunction, inner_return_var: str
) -> str:
    mutable_arg_names = f.func.arguments.mutable_arg_names()
    (
        aliased_outer_rets,
        non_aliased_outer_rets,
    ) = get_mutable_redispatch_return_names(f, inner_return_var)
    _, non_aliased_inner_rets = get_mutable_redispatch_return_names(
        functional_op, inner_return_var
    )
    # The outer function may have a mix of aliased and non-aliased outputs,
    # But the inner functional op that we're transforming to should only have non-aliased outputs
    if len(mutable_arg_names) + len(non_aliased_outer_rets) != len(non_aliased_inner_rets):
        raise RuntimeError("The sum of the lengths of mutable_arg_names and non_aliased_outer_rets "
                            "does not equal the length of non_aliased_inner_rets.")

    # First, take all of the newly created outputs from the inner call and wrap them into functional tensors
    updates = []
    non_aliased_wrapped_ret_names = []
    for i, inner_ret in enumerate(
        non_aliased_inner_rets[: len(non_aliased_outer_rets)]
    ):
        ret_name = f"output_{i}"
        updates.append(
            f"""\
  auto output_{i} = at::functionalization::impl::to_functional_tensor({inner_ret});"""
        )
        non_aliased_wrapped_ret_names.append(ret_name)

    # Next, take all of the mutated outputs from the inner call corresponding to mutated inputs,
    # and propagate the mutations
    for outer_arg, inner_ret in zip(
        mutable_arg_names, non_aliased_inner_rets[len(non_aliased_outer_rets) :]
    ):
        updates.append(
            f"""\
at::functionalization::impl::replace_({outer_arg}, {inner_ret});
        at::functionalization::impl::commit_update({outer_arg});
        at::functionalization::impl::sync({outer_arg});"""
        )

    # Finally, we return:
    # - Any mutable arguments that also returns
    # - Any immutable returns that were created wrapping the output from the inner call
    returns_str = return_str(
        f.func.returns, aliased_outer_rets + non_aliased_wrapped_ret_names
    )
    updates_str = "\n".join(updates)
    return f"""\
{updates_str}
        {returns_str}"""


# Generates the Functionalization kernel for:
# - mutation ops (inplace and out= ops)
@with_native_function_and
def emit_inplace_functionalization_body(
    f: NativeFunction, g: NativeFunctionsGroup
) -> str:
    # mutation case
    if not modifies_arguments(f):
        raise RuntimeError(f"The function {f.func.name} does not modify its arguments.")

    dispatcher_sig = DispatcherSignature.from_schema(f.func)
    dispatcher_sig_func = DispatcherSignature.from_schema(g.functional.func)

    unwrap_tensor_args_str, unwrapped_args_ctx = unwrap_tensor_args(
        dispatcher_sig, is_view_op=False
    )

    mutated_names = [
        a.name
        for a in f.func.arguments.flat_all
        if a.type.is_tensor_like() and a.annotation is not None
    ]
    non_mutated_names = [
        a.name
        for a in f.func.arguments.flat_all
        if a.type.is_tensor_like() and a.annotation is None
    ]
    # all mutable inputs must be functional tensors in order to participate in functionalization
    check_all_mutated_args_are_functional = " && ".join(
        ["true"]
        + [
            f"at::functionalization::impl::isFunctionalTensor({a})"
            for a in mutated_names
        ]
    )
    check_any_non_mutated_args_are_functional = " || ".join(
        ["false"]
        + [
            f"at::functionalization::impl::isFunctionalTensor({a})"
            for a in non_mutated_names
        ]
    )

    # These are used in the cases where we don't functionalize and redispatch to the inplace op
    # case 1: we hit an inplace op that doesn't have an out-of-place equivalent
    # case 2: we hit an inplace ops but our inputs are not functional tensors (in which case our kernel just no-ops)
    inplace_exprs = [
        e.expr
        for e in translate(unwrapped_args_ctx, dispatcher_sig.arguments(), method=False)
    ]

    # call the out-of-place variant of the op
    return_type = (
        dispatcher.returns_type(g.functional.func.returns).remove_const_ref().cpp_type()
    )
    functional_sig = DispatcherSignature.from_schema(g.functional.func)
    functional_exprs = [
        e.expr
        for e in translate(unwrapped_args_ctx, functional_sig.arguments(), method=False)
    ]

    meta_conversion_str, meta_call_ctx = convert_to_meta_tensors(dispatcher_sig)
    # We don't want to run the inplace meta func for ops like .set_(), because:
    # (1) they're unnecessary: inplace meta checks are only useful for ops like add_(),
    #     where broadcasting will work for the out-of-place case but should fail on the inplace call
    # (2) They'll also fail without adding extra infra: we'd need to convert the input storage argument
    #     into a meta storage
    any_storage_args = any(
        a.type == BaseType(BaseTy.Storage) for a in f.func.arguments.flat_all
    )

    return f"""
{dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{
    if ({str(not any_storage_args and f.func.kind() == SchemaKind.inplace).lower()}) {{
        // Before converting the mutable op to its functional variant, run meta tensors through the original op.
        // This will help us catch shape errors that apply to inplace ops that wouldn't apply to their functional variants.
        // (We can only do this for inplace ops today though, because they technicaly all support meta tensors).
        {meta_conversion_str}
        at::AutoDispatchSkipFunctionalize func_guard;
        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);
        at_npu::native::custom_ops::{dispatcher_sig.name()}({', '.join(a.name for a in meta_call_ctx)});
    }}
{unwrap_tensor_args_str}
    if (!({check_all_mutated_args_are_functional})) {{
        if ({check_any_non_mutated_args_are_functional}) {{
            // case 1: trying to mutate a non functional tensor with a functional tensor is an error
            TORCH_INTERNAL_ASSERT(false,
            "mutating a non-functional tensor with a functional tensor is not allowed.",
            " Please ensure that all of your inputs are wrapped inside of a functionalize() call.", PTA_ERROR(ErrCode::PARAM));
        }} else {{
            // case 2: arguments are not functional tensors, so we no-op and redispatch.
            at::AutoDispatchSkipFunctionalize guard;
            {maybe_create_output(f, 'tmp_output')}at_npu::native::custom_ops::{dispatcher_sig.name()}({', '.join(inplace_exprs)});
            {return_from_mutable_noop_redispatch(f, 'tmp_output')}
        }}
    }} else {{
        {return_type} tmp_output;
        {{
            at::AutoDispatchSkipFunctionalize guard;
            tmp_output = at_npu::native::custom_ops::{dispatcher_sig_func.name()}({', '.join(functional_exprs)});
        }}
        {wrap_propagate_mutations_and_return(f, g.functional, 'tmp_output')}
    }}
}}"""


def gen_functionalization_registration(
    selector: SelectiveBuilder,
    g: Union[NativeFunction, NativeFunctionsGroupOptionalOut],
) -> List[str]:
    @with_native_function
    def emit_registration_helper(f: NativeFunction) -> str:
        registration_str = f"TORCH_FN(functionalization::{wrapper_name(f.func)})"
        return f'm.impl("{f.func.name}", {registration_str});'

    # Don't generate kernels in mobile build
    if not selector.include_all_operators:
        return []

    if isinstance(g, NativeFunctionsGroupOptionalOut):
        fns = list(g.functions())
    else :
        fns = []

    registrations = []
    for f in fns:
        if len(f.func.arguments.out) > 1:
            return []
        if str(f.func.name) == "lift":
            # See Note [Functionalization <> torch.Tensor constructor]
            return []
        if str(f.func.name) == "resize_":
            # See Note [resize_ in Functionalization]
            return []
        if f.is_view_op:
            raise RuntimeError(f"The {f.func.name} func is a view op.")
        # functionalization needs to generate and register kernals for inplace ops.
        # We *also* need to directly register CompositeImplicitAUtograd kernels
        # so that they decompose properly before functioanlization.
        if modifies_arguments(f):
            registrations.append(emit_registration_helper(f))
    return registrations


def gen_functionalization_definition(
    selector: SelectiveBuilder,
    # Note: Ideally this code should never have to look at NativeFunction
    # (and instead only need to operate on grouped NativeFunctions).
    # The only reason currently is because we need to emit direct dispatch registrations
    # For CompositeImplicitAutograd operators, which are potentially ungrouped.
    g: Union[NativeFunction, NativeFunctionsGroupOptionalOut],
) -> List[str]:
    # Don't generate kernels in mobile build
    if not selector.include_all_operators:
        return []

    if isinstance(g, NativeFunction):
        return []
    else:
        # Case 2: emit inplace -> out-of-place kernels for the functionalization.
        mutation_defs = []
        if g.out is not None:
            if len(g.out.func.arguments.out) > 1:
                return []
            mutation_defs.append(emit_inplace_functionalization_body(g.out, g))
        if g.inplace is not None:
            mutation_defs.append(emit_inplace_functionalization_body(g.inplace, g))
        if g.mutable is not None:
            mutation_defs.append(emit_inplace_functionalization_body(g.mutable, g))
        return mutation_defs
    return []
