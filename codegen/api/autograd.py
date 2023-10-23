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

from dataclasses import dataclass
import re
from typing import Optional, Sequence, Set, List, Tuple, Match

from codegen.api import cpp
from codegen.api.types import Binding, NamedCType
from codegen.model import NativeFunction, Type, SchemaKind
from codegen.utils import IDENT_REGEX

# Represents a saved attribute involved in backward calculation.
# Note that it can be a derived property of an input argument, e.g.:
# we could save `other.scalar_type()` instead of the entire `other` tensor.
@dataclass(frozen=True)
class SavedAttribute:
    # The NamedCType holds the updated name and cpp type of the attribute
    # for the name, Suffix is appended if it's derived property, e.g.: `other_scalar_type`
    nctype: NamedCType

    # The expression to read the derived property at save time, e.g.:
    # `other.scalar_type()`.
    expr: str

# Represents a backward formula that calculates derivatives for one
# or more tensors.
@dataclass(frozen=True)
class Derivative:
    # The formula string (legit C++ expression).
    # Note that expressions against input arguments have been replaced with the
    # corresponding saved attributes.
    # E.g.:
    #  raw formula: `mul_tensor_backward(grad, self, other.scalar_type())`
    #         here: `mul_tensor_backward(grad, self, other_scalar_type)`
    formula: str

    # The formula string before input argument replacement
    original_formula: str

    # Names of the arguments for which this formula calculates derivatives.
    var_names: Tuple[str, ...]

    # Saved inputs that are referenced by the formula.
    saved_inputs: Tuple[SavedAttribute, ...]

    # Saved outputs that are referenced by the formula.
    saved_outputs: Tuple[SavedAttribute, ...]

    # Gradients that are referenced by name in the formula.
    named_gradients: Set[str]

# Represents a forward formula that calculates forward derivatives
# for one tensor.
@dataclass(frozen=True)
class ForwardDerivative:
    # The formula string (legit C++ expression).
    # Note that special keywords such as "linear" or "element_wise" have been
    # replaced by the automatically generated formula.
    formula: str

    # Name of the output argument for which this formula calculates forward
    # derivatives
    var_name: str

    # Type of the output argument for which this formula calculates forward
    # derivatives
    var_type: Type

    # Inputs for which the forward derivatives are required for this formula
    required_inputs_fw_grad: Optional[Tuple[str, ...]]

    # Inputs for which the primal is required for this formula
    required_inputs_primal: Optional[Tuple[str, ...]]

    # Flag to specify if this formula requires the original value of self
    # This is only used by inplace operations
    required_original_self_value: bool

    # If this formula is specified in derivatives.yaml or if we are re-using the
    # out of place formula for inplace
    is_reusing_outplace_formula: bool

# Represents differentiability info for a NativeFunction.
@dataclass(frozen=True)
class DifferentiabilityInfo:
    # The base name read from derivatives.yaml.
    name: str

    # The matching native function.
    #
    # There can be multiple NativeFunction having the same base name:
    #  - different overloads with different types of input arguments;
    #  - in-place/out/functional variants of the same function;
    #
    # We first use the schema string (under the 'name' key) in derivatives.yaml
    # to find the NativeFunction having the same schema string.
    # Then we find the in-place/out/functional variants of the matching function.
    # Among these variants, we choose the one having the same name as the
    # derivatives.yaml entry. If there is no exact match, then we choose the
    # in-place variant.
    func: NativeFunction

    # The name of the generated autograd function.
    # It's set only if we will calculate a derivative, i.e.
    # 'args_with_derivatives' is not empty.
    op: Optional[str]

    # The derivatives formulae for this function.
    # Note that the length of this sequence is the number of differentiable inputs
    derivatives: Sequence[Derivative]

    # The forward derivatives formulae for this function.
    # Note that the length of this sequence is the number of differentiable outputs
    forward_derivatives: Sequence[ForwardDerivative]

    # The union of 'saved_inputs' of all 'derivatives'.
    all_saved_inputs: Sequence[SavedAttribute]

    # The union of 'saved_outputs' of all 'derivatives'.
    all_saved_outputs: Sequence[SavedAttribute]

    # All named gradients that are available for use, in the same
    # order as in the grads vector.
    available_named_gradients: Sequence[str]

    # The named gradients that are used in any of the derivatives.
    # Invariant: all(name in available_named_gradients for name in used_named_gradients)
    used_named_gradients: Set[str]

    # The function's input arguments for which it calculates derivatives.
    # It's the union of 'var_names' of all 'derivatives', sorted by the
    # argument order in the function schema.
    args_with_derivatives: Sequence[Binding]

    # Names of arguments whose derivative formula is 'non_differentiable'.
    non_differentiable_arg_names: Sequence[str]

    # Raw data read from derivatives.yaml.
    output_differentiability: Optional[List[bool]]

    # output_differentiability in derivatives.yaml can be a list of
    # conditions that express if the output is differentiable. In this case,
    # the number of conditions must match the number of outputs
    # (NB: we only support one condition right now).
    # output_differentiability gets populated with True for each condition,
    # while output_differentiability_conditions gets populated with the conditions
    output_differentiability_conditions: Optional[List[str]]

    @property
    def has_derivatives(self) -> bool:
        return len(self.args_with_derivatives) > 0

def uses_ident(info: Optional[DifferentiabilityInfo], ident: str) -> bool:
    if info is None:
        return False
    for derivative in info.derivatives:
        formula = derivative.formula
        if re.search(IDENT_REGEX.format(ident), formula):
            return True
    return False

def uses_retain_variables(info: Optional[DifferentiabilityInfo]) -> bool:
    return uses_ident(info, 'retain_variables')

def uses_single_grad(info: Optional[DifferentiabilityInfo]) -> bool:
    return uses_ident(info, 'grad')

# Represents a differentiable `Argument`.
# How is it different from the `Argument` type?
# - It's processed Arguments which are differentiable and only used in the
#   context of the autograd codegen;
# - It can represent SelfArgument or regular Argument but not TensorOptionsArgument;
@dataclass(frozen=True)
class DifferentiableInput:
    name: str
    type: Type

    cpp_type: str

# Represents a differentiable `Return`.
# How it it different from the `Return` type?
# - The name in `Return` is optional. Here it is always populated using the same
#   `cpp.return_names()` method.
# - It's processed Returns which are differentiable, in compliance with the
#   `output_differentiability` field defined in derivatives.yaml (if specified),
#   and are only used in the context of the autograd codegen;
@dataclass(frozen=True)
class DifferentiableOutput:
    name: str
    type: Type

    cpp_type: str

@dataclass(frozen=True)
class NativeFunctionWithDifferentiabilityInfo:
    func: NativeFunction
    info: Optional[DifferentiabilityInfo]
    fw_derivatives: Sequence[ForwardDerivative]

def dispatch_strategy(fn: NativeFunctionWithDifferentiabilityInfo) -> str:
    """How are we going to call the underlying implementation of a
    declaration?  There are two strategies:
        - use_derived: we want to call the implementation on CPUDoubleType
          (or a similar, derived Type instance).  Because these derived
          instances deal in Tensors, not Variables (it's a completely different
          object, so it doesn't dispatch back to VariableType), code on
          this dispatch path needs to wrap/unwrap tensors.  If the
          derived implementation takes and returns tensors, the
          implementation is usually differentiable (although we also use
          the derived dispatch path for non-differentiable functions
          that we still want to dispatch on the derived Type instance;
          e.g., size())
        - use_type: we want to call the implementation on Type, because
          it is implemented concretely, and the functions it invokes will
          get dispatched back to VariableType (which will ensure that they
          are differentiable.)
    """
    if fn.func.is_abstract or (fn.info is not None and fn.info.has_derivatives):
        # If the function is abstract (not implemented on at::Type), we must
        # call the implementation on the derived type with unpacked tensors.

        # If the function has a derivative specified and is concrete, we could
        # call either implementation. We prefer the calling the derived
        # type's implementation with unpacked tensors because it is more
        # performant in some cases: any internal calls to other ATen functions
        # won't have the history tracked.

        # If the function has a type dispatched argument (i.e. is a factory),
        # we prefer calling the derived type's implementation both because it is
        # more performant and to ensure factory functions return tensors with _version
        # of 0 (probably not strictly necessary, but nice to have to keeps versions simple
        # to understand.

        return 'use_derived'
    else:
        # If the function is concrete (we don't have to override it) and we
        # didn't declare it in derivatives.yaml, we'll assume that it is
        # actually implemented out of differentiable functions. (This
        # assumption might not hold, but then you'll see gradcheck fail.)
        return 'use_type'

def match_differentiability_info(
    native_functions: List[NativeFunction],
    differentiability_infos: Sequence[DifferentiabilityInfo],
) -> List[NativeFunctionWithDifferentiabilityInfo]:
    """Sets the "derivative" key on declarations to matching autograd function
    In-place functions will use the out-of-place derivative definition if there
    is no in-place specific derivative.
    """

    info_by_schema = {info.func.func: info for info in differentiability_infos}
    functional_info_by_signature = {
        info.func.func.signature(strip_default=True): info
        for info in differentiability_infos
        if info.func.func.kind() == SchemaKind.functional}

    def find_info(f: NativeFunction) -> Tuple[Optional[DifferentiabilityInfo], bool]:
        if f.func in info_by_schema:
            return info_by_schema[f.func], True

        # if there is no exact match look for the out-of-place signature.
        # i.e mul() for mul_() or mul_out()
        return functional_info_by_signature.get(f.func.signature(strip_default=True)), False

    def assert_strides_or_error(f, info):
        if f.func.kind() == SchemaKind.inplace and (info is not None):
            for derivative in info.derivatives:
                if 'self' not in derivative.var_names:
                    continue

                for saved_input in derivative.saved_inputs:
                    if 'strides_or_error' in saved_input.expr:
                        raise KeyError("Calling '.strides()' in the 'self' derivative formula of an "
                                       f"in-place function is not supported: {f.func}")

    result: List[NativeFunctionWithDifferentiabilityInfo] = []
    for f in native_functions:
        info, is_exact_match = find_info(f)
        # Currently, the '.strides()' to 'strides_or_error' replacement does not support
        # 'self' derivatives of an inplace function, so we must check for this case.
        assert_strides_or_error(f, info)
        # For functions that have a single def for out-of-place and inplace (like abs())
        if not info or not info.forward_derivatives:
            forward_derivatives = []
            result.append(NativeFunctionWithDifferentiabilityInfo(
                func=f,
                info=info,
                fw_derivatives=forward_derivatives
            ))
            continue

        forward_derivatives = info.forward_derivatives
        if f.func.kind() == SchemaKind.inplace:
            # For inplace functions there is a little bit of work to do:
            #  1) Validate the formula and make sure the input that is modified in not used:
            #    - If there is a formula for the inplace variant of the function (is_exact_match == True) then
            #      we make sure that the original value of the input that is being modified inplace (self_p) is
            #      not used in the formula. Note that the formula can use "original_self_p" here and that would
            #      trigger a clone of the original input.
            #    - If we are re-using the out of place formula (is_exact_match == False) then we replace every
            #      occurrence of self_p and self_t by original_self_p and original_self_t. These will be
            #      populated by cloned version of the original input (either the clone done by the backward AD
            #      logic if self is also used in a backward formula or a special clone that we add).
            #  2) At this point, there cannot be a self_p in the formula.
            #  3) Change "result" into "self_p" as by design, in the inplace function codegen, the result is
            #     simply called self (as it is modified inplace).
            #  4) Update the required primals data in case it used to contain "result" but should now contain
            #     "self"
            #  5) If it is not an exact match, the user formula is not modifying the existing forward grad
            #     inplace as it should. So add some code that makes sure that we do so if the forward grad
            #     already exists.

            # Only single output inplace should exist
            if len(info.forward_derivatives) != 1:
                raise ValueError("len(info.forward_derivatives) != 1")
            fw_info = info.forward_derivatives[0]
            formula = fw_info.formula

            def replace_self_with_original_self(formula: str, postfix: str) -> str:
                def repl(m: Match[str]) -> str:
                    return f'{m.group(1)}original_self{postfix}{m.group(2)}'
                return re.sub(IDENT_REGEX.format(f'self{postfix}'), repl, formula)

            if re.search(IDENT_REGEX.format("self_p"), formula):
                if is_exact_match:
                    raise ValueError(f'The formula for "{f.func.name}" is using the original value of self'
                                     ' that is being modified inplace. This would lead to wrong forward '
                                     'gradients. Please use "result" in the formula only.')
                # When the original formula is out of place, we save a clone of the primal
                # value to be able to access this value if needed
                # replace "self_p"/"self_t" from the formula by "original_self_p"/"original_self_t"
                formula = replace_self_with_original_self(formula, "_p")
                formula = replace_self_with_original_self(formula, "_t")

            # replace "result" from the formula by "self_p"
            def repl(m: Match[str]) -> str:
                return f'{m.group(1)}self_p{m.group(2)}'
            formula = re.sub(IDENT_REGEX.format("result"), repl, formula)

            required_primals = fw_info.required_inputs_primal
            if re.search(IDENT_REGEX.format("self_p"), formula):
                required_primals = required_primals + ("self",) if required_primals else ("self",)

            if not is_exact_match:
                # Make sure that the forward grad is modified inplace when the original formula
                # is out of place
                formula = f"self_t_raw.defined() ? self_t_raw.copy_({formula}) : {formula}"

            required_original_self_value = bool(re.search(IDENT_REGEX.format("original_self_p"), formula))

            forward_derivatives = [ForwardDerivative(
                formula=formula, var_name="self", var_type=fw_info.var_type,
                required_inputs_fw_grad=fw_info.required_inputs_fw_grad,
                required_inputs_primal=required_primals,
                required_original_self_value=required_original_self_value,
                is_reusing_outplace_formula=not is_exact_match), ]

        result.append(NativeFunctionWithDifferentiabilityInfo(func=f, info=info, fw_derivatives=forward_derivatives))
    return result

def is_differentiable(name: str, func_type: Type, info: Optional[DifferentiabilityInfo]) -> bool:
    return func_type.is_tensor_like() and (info is None or name not in info.non_differentiable_arg_names)

def gen_differentiable_outputs(fn: NativeFunctionWithDifferentiabilityInfo) -> List[DifferentiableOutput]:
    f = fn.func
    info = fn.info
    outputs: List[DifferentiableOutput] = [
        DifferentiableOutput(name=name, type=ret.type, cpp_type=cpp.return_type(ret).cpp_type())
        for name, ret in zip(cpp.return_names(f), f.func.returns)]
    output_differentiability = info.output_differentiability if info else None
    if output_differentiability is not None:
        if len(output_differentiability) != len(outputs):
            raise RuntimeError(f"The length of output_differentiability ({len(output_differentiability)}), "
                               f"does not match the number of outputs ({len(outputs)}).")
        differentiable_outputs: List[DifferentiableOutput] = []
        if False in output_differentiability and f.func.kind() == SchemaKind.inplace:
            raise RuntimeError("output_differentiability=False for inplace operation.")
        for differentiable, output in zip(output_differentiability, outputs):
            if differentiable:
                differentiable_outputs.append(output)
        return differentiable_outputs
    candidate_differentiable_outputs = list(filter(lambda r: is_differentiable(r.name, r.type, info), outputs))
    if uses_single_grad(info):
        return candidate_differentiable_outputs[:1]
    else:
        return candidate_differentiable_outputs
