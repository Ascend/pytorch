from collections.abc import Sequence
from functools import partial
import warnings
import sys
import threading
import time
import unittest
import inspect
import itertools
import contextlib
import re
import os
import stat

from typing import Dict
from collections import defaultdict
from importlib import import_module

import torch
import torch_npu
import torch_npu.testing
from torch.utils._pytree import tree_map

from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
    floating_and_complex_types_and,
    all_types_and_complex_and,
)

from torch.testing._internal.common_utils import (
    TestCase,
    is_iterable_of_tensors,
    run_tests,
    IS_SANDCASTLE,
    clone_input_helper,
    set_default_dtype,
    suppress_warnings,
    noncontiguous_like,
    parametrize,
    skipIfTorchInductor,
)
from torch.testing._internal.common_methods_invocations import (
    op_db,
    UnaryUfuncInfo,
    ReductionOpInfo,
    ReductionPythonRefInfo,
    SpectralFuncInfo,
    ops_and_refs,
    python_ref_db,
    BinaryUfuncInfo,
    xfail,
    skip,
    skipOps
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
    OpDTypes,
    skipMeta,
)
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
)
from torch._subclasses.fake_utils import outputs_alias_inputs

import torch._prims as prims
from torch._prims.context import TorchRefsMode

from torch.testing._internal import opinfo
from torch.testing._internal import composite_compliance

from torch.utils._pytree import tree_flatten
from torch.utils._python_dispatch import TorchDispatchMode


if torch.get_default_dtype() != torch.float32:
    raise RuntimeError("default dtype not equals to float32")

# variant testing is only done with torch.float and torch.cfloat to avoid
#   excessive test times and maximize signal to noise ratio
_variant_ops = partial(
    ops, dtypes=OpDTypes.supported, allowed_dtypes=(torch.float, torch.cfloat)
)

# Get names of all the operators which have ref in their entry in OpInfo (testing infra)
#   except for elementwise unary operators (separately implemented in test/test_unary_ufuncs.py),
#   elementwise binary operators (separately implemented in test_binary_ufuncs.py),
#   reduction operations (separately impelemented in test_reductions.py),
#   and Spectral Functions (separately implemented for only 1D as of now, in test/test_spectral_ops.py)
_ref_test_ops = tuple(
    filter(
        lambda op: not isinstance(
            op, (UnaryUfuncInfo, ReductionOpInfo, SpectralFuncInfo, BinaryUfuncInfo)
        )
        and op.ref is not None,
        op_db,
    )
)
_ops_and_refs = op_db + python_ref_db


def reduction_dtype_filter(op):
    if (not isinstance(op, ReductionPythonRefInfo) or not op.supports_out
       or torch.int16 not in op.dtypes):
        return False

    argspec = inspect.getfullargspec(op.op)
    if 'dtype' not in argspec.kwonlyargs:
        return False
    return True


# Create a list of operators that are a subset of _ref_test_ops but don't have a
# numpy ref to compare them too, If both CPU and NPU are compared to numpy
# then they do not need to be compared to each other
_ops_and_refs_with_no_numpy_ref = [op for op in _ops_and_refs if op.ref is None]

aten = torch.ops.aten


class TestCommon(TestCase):
    exact_dtype = True

    # Verifies, on teardown, that no OpInfo is still using dynamic dtypes in CI
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    @unittest.skip("NPU doesn't support yet.")
    def test_pointwise_tag_coverage(self):

        pytorch_dir = os.path.abspath(__file__ + "/../../")
        files = [
            "aten/src/ATen/native/UnaryOps.cpp",
            "aten/src/ATen/native/BinaryOps.cpp",
            "aten/src/ATen/native/PointwiseOps.cpp",
            "aten/src/ATen/native/TensorCompare.cpp",
        ]

        allowed_functions = (
            # reduction version of these operators
            "aten.max.default",
            "aten.max.dim",
            "aten.max.dim_max",
            "aten.max.names_dim",
            "aten.max.names_dim_max",
            "aten.max.unary_out",
            "aten.min.default",
            "aten.min.dim",
            "aten.min.dim_min",
            "aten.min.names_dim",
            "aten.min.names_dim_min",
            "aten.min.unary_out",
            # not pointwise
            "aten.isin.Tensor_Tensor",
            "aten.isin.Tensor_Tensor_out",
            "aten.isin.Tensor_Scalar",
            "aten.isin.Tensor_Scalar_out",
            "aten.isin.Scalar_Tensor",
            "aten.isin.Scalar_Tensor_out",
            "aten.mode.default",
            "aten.mode.dimname",
            "aten.mode.dimname_out",
            "aten.mode.values",
        )

        regex = re.compile(r"DEFINE_DISPATCH\(.*_stub")

        def get_opoverloadpacket_from_dispatch(kernel):
            if hasattr(torch.ops.aten, kernel):
                return kernel
            if hasattr(torch.ops.aten, f"__{kernel}__"):
                return f"__{kernel}__"
            if hasattr(torch.ops.aten, f"special_{kernel}"):
                return f"special_{kernel}"
            if "_" in kernel:
                kernel_split = kernel.split("_")
                new_kernel = "_".join(kernel_split[:-1])
                if hasattr(torch.ops.aten, new_kernel):
                    return new_kernel
                else:
                    return None

            # could not find op from kernel dispatch string
            self.assertTrue(False)

        for file_name in files:
            with open(os.path.join(pytorch_dir, file_name)) as f:
                lines = f.read()
                matches = regex.findall(lines)
                for match in matches:
                    kernel = match[len("DEFINE_DISPATCH("):-len("_stub")]

                    # no op definition for it, but defined with DEFINE_DISPATCH ?
                    if kernel == "trigamma":
                        continue

                    kernel = get_opoverloadpacket_from_dispatch(kernel)
                    overloadpacket = getattr(torch.ops.aten, kernel)

                    for overload_name in overloadpacket.overloads():
                        overload = getattr(overloadpacket, overload_name)

                        if not torch._C._dispatch_has_kernel(overload.name()):
                            continue

                        # and there's no way of specifying them
                        if torch.Tag.generated in overload.tags:
                            continue

                        if str(overload) in allowed_functions:
                            continue

                        self.assertTrue(torch.Tag.pointwise in overload.tags)

    # Tests that the cpu and npu results are consistent
    @suppress_warnings
    @ops(_ops_and_refs_with_no_numpy_ref, dtypes=OpDTypes.any_common_cpu_cuda_one)
    def test_compare_cpu(self, device, dtype, op):

        def to_cpu(arg):
            if isinstance(arg, torch.Tensor):
                return arg.to(device='cpu')
            return arg

        samples = op.reference_inputs(device, dtype)

        for sample in samples:
            cpu_sample = sample.transform(to_cpu)
            npu_results = op(sample.input, *sample.args, **sample.kwargs)
            cpu_results = op(cpu_sample.input, *cpu_sample.args, **cpu_sample.kwargs)

            # output_process_fn_grad has a very unfortunate name
            # We use this function in linalg extensively to postprocess the inputs of functions
            # that are not completely well-defined. Think svd and muliplying the singular vectors by -1.
            # CPU and NPU implementations of the SVD can return valid SVDs that are different.
            # We use this function to compare them.
            npu_results = sample.output_process_fn_grad(npu_results)
            cpu_results = cpu_sample.output_process_fn_grad(cpu_results)

            # Lower tolerance because we are running this as a `@slowTest`
            # Don't want the periodic tests to fail frequently
            self.assertEqual(npu_results, cpu_results, atol=1e-3, rtol=1e-3)

    @unittest.skip("NPU doesn't support yet.")
    @ops([op for op in op_db if op.error_inputs_func is not None], dtypes=OpDTypes.none)
    def test_errors(self, device, op):
        error_inputs = op.error_inputs(device)
        for ei in error_inputs:
            si = ei.sample_input
            with self.assertRaisesRegex(ei.error_type, ei.error_regex):
                out = op(si.input, *si.args, **si.kwargs)
                self.assertFalse(isinstance(out, type(NotImplemented)))

    # Validates ops implement the correct out= behavior
    #   for a description of the correct behavior
    # Validates the following cases:
    #   - Case 0: out has the correct shape, dtype, and device but is full of extremal values
    #   - Case 1: out has the correct shape, dtype, and device but is noncontiguous
    #   - Case 2: out has the correct dtype and device, but is zero elements
    #   - Case 3: out has the correct shape and dtype, but is on a different device type
    #   - Case 4: out has the correct shape and device, but a dtype that cannot
    #       "safely" cast to
    #
    # Case 3 and 4 are slightly different when the op is a factory function:
    #   - if device, dtype are NOT passed, any combination of dtype/device should be OK for out
    #   - if device, dtype are passed, device and dtype should match
    @ops(_ops_and_refs, dtypes=OpDTypes.any_one)
    @skipIfTorchInductor("Inductor does not support complex dtype yet")
    @unittest.skip("skip test_out now")
    def test_out(self, device, dtype, op):
        # Prefers running in float32 but has a fallback for the first listed supported dtype
        samples = op.sample_inputs(device, dtype)
        for sample in samples:
            # calls it normally to get the expected result
            expected = op(sample.input, *sample.args, **sample.kwargs)
            op_out = partial(op, sample.input, *sample.args, **sample.kwargs)

            # Short-circuits if output is not a single tensor or an
            #   iterable of tensors
            if not isinstance(expected, torch.Tensor) and not is_iterable_of_tensors(
                expected, include_empty=True
            ):
                self.skipTest(
                    "Skipped! Only supports single tensor or iterable of tensor outputs."
                )

            # Validates the op doesn't support out if it claims not to
            if not op.supports_out:
                with self.assertRaises(Exception):
                    if op_out(out=expected) == NotImplemented:
                        raise RuntimeError("Except to support out but get not implemented")
                return

            # A wrapper around map that works with single tensors and always
            #   instantiates the map. Used below to apply transforms to
            #   single tensor and iterable tensor outputs.
            def _apply_out_transform(fn, out):
                if isinstance(out, torch.Tensor):
                    return fn(out)

                # assumes (see above) that out is an iterable of tensors
                return tuple(map(fn, out))

            # Extracts strides from a tensor or iterable of tensors into a tuple
            def _extract_strides(out):
                if isinstance(out, torch.Tensor):
                    return (out.stride(),)

                # assumes (see above) that out is an iterable of tensors
                return tuple(t.stride() for t in out)

            # Extracts data pointers from a tensor or iterable of tensors into a tuple
            # NOTE: only extracts on the CPU and NPU device types since some
            #   device types don't have storage
            def _extract_data_ptrs(out):
                if isinstance(out, torch.Tensor):
                    return (out.data_ptr(),)

                # assumes (see above) that out is an iterable of tensors
                return tuple(t.data_ptr() for t in out)

            def _compare_out(transform, *, compare_strides_and_data_ptrs=True):
                out_ = _apply_out_transform(transform, expected)
                original_strides = _extract_strides(out_)
                original_ptrs = _extract_data_ptrs(out_)

                op_out(out=out_)
                final_strides = _extract_strides(out_)
                final_ptrs = _extract_data_ptrs(out_)
                self.assertEqual(expected, out_)

                if compare_strides_and_data_ptrs:
                    stride_msg = "Strides are not the same! Original strides were {} and strides are now {}".format(
                        original_strides, final_strides
                    )
                    self.assertEqual(original_strides, final_strides, msg=stride_msg)
                    self.assertEqual(original_ptrs, final_ptrs)

            # Case 0: out= with the correct shape, dtype, and device
            #   but NaN values for floating point and complex tensors, and
            #   maximum values for integer tensors.
            #   Expected behavior: out= values have no effect on the computation.
            def _case_zero_transform(t):
                try:
                    info = torch.iinfo(t.dtype)
                    return torch.full_like(t, info.max)
                except TypeError as te:
                    # for non-integer types fills with NaN
                    return torch.full_like(t, float("nan"))

            _compare_out(_case_zero_transform)

            # Case 1: out= with the correct shape, dtype, and device,
            #   but noncontiguous.
            #   Expected behavior: strides are respected and `out` storage is not changed.
            def _case_one_transform(t):
                return make_tensor(
                    t.shape, dtype=t.dtype, device=t.device, noncontiguous=True
                )

            _compare_out(_case_one_transform)

            # Case 2: out= with the correct dtype and device, but has no elements.
            #   Expected behavior: resize without warning.
            def _case_two_transform(t):
                return make_tensor((0,), dtype=t.dtype, device=t.device)

            _compare_out(_case_two_transform, compare_strides_and_data_ptrs=False)

            # Also validates that no warning is thrown when this out is resized
            out = _apply_out_transform(_case_two_transform, expected)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                op_out(out=out)

            # Verifies no warning is a resize warning
            for w in caught:
                if "An output with one or more elements" in str(w.message):
                    self.fail(
                        "Resizing an out= argument with no elements threw a resize warning!"
                    )

            # Case 3: out= with correct shape and dtype, but wrong device.
            wrong_device = None
            if torch.device(device).type != "cpu":
                wrong_device = "cpu"
            elif torch.npu.is_available():
                wrong_device = "npu"

            factory_fn_msg = (
                "\n\nNOTE: If your op is a factory function (i.e., it accepts TensorOptions) you should mark its "
                "OpInfo with `is_factory_function=True`."
            )
            if wrong_device is not None:

                def _case_three_transform(t):
                    return make_tensor(t.shape, dtype=t.dtype, device=wrong_device)

                out = _apply_out_transform(_case_three_transform, expected)

                if op.is_factory_function and sample.kwargs.get("device", None) is None:
                    op_out(out=out)
                else:
                    msg_fail = (
                        f"Expected RuntimeError when calling with input.device={device} and out.device={wrong_device}."
                    ) + factory_fn_msg
                    with self.assertRaises(RuntimeError, msg=msg_fail):
                        op_out(out=out)

            # Case 4: out= with correct shape and device, but a dtype
            #   that output cannot be "safely" cast to (long).
            #   Expected behavior: error.
            # NOTE: this case is filtered by dtype since some ops produce
            #   bool tensors, for example, which can be safely cast to any
            #   dtype. It is applied when single tensors are floating point or complex
            #   dtypes, or if an op returns multiple tensors when at least one such
            #   tensor is a floating point or complex dtype.
            _dtypes = floating_and_complex_types_and(torch.float16, torch.bfloat16)
            if (
                isinstance(expected, torch.Tensor)
                and expected.dtype in _dtypes
                or (
                    not isinstance(expected, torch.Tensor)
                    and any(t.dtype in _dtypes for t in expected)
                )
            ):

                def _case_four_transform(t):
                    return make_tensor(t.shape, dtype=torch.long, device=t.device)

                out = _apply_out_transform(_case_four_transform, expected)
                msg_fail = "Expected RuntimeError when doing an unsafe cast!"
                msg_fail = (
                    msg_fail
                    if not isinstance(expected, torch.Tensor)
                    else (
                        "Expected RuntimeError when doing an unsafe cast from a result of dtype "
                        f"{expected.dtype} into an out= with dtype torch.long"
                    )
                ) + factory_fn_msg

                if op.is_factory_function and sample.kwargs.get("dtype", None) is None:
                    op_out(out=out)
                else:
                    with self.assertRaises(RuntimeError, msg=msg_fail):
                        op_out(out=out)

    @ops(filter(reduction_dtype_filter, _ops_and_refs), dtypes=(torch.int16,))
    def test_out_integral_dtype(self, device, dtype, op):
        def helper(with_out, expectFail, op_to_test, inputs, *args, **kwargs):
            out = None
            try:
                if with_out:
                    out = torch.empty(0, dtype=torch.int32, device=device)
                    op_to_test(inputs, out=out, *args, **kwargs)
                else:
                    out = op_to_test(inputs, *args, **kwargs)
                self.assertFalse(expectFail)
            except RuntimeError as err:
                self.assertEqual(
                    str(err), "dtype argument and out dtype must match in reduction")
                self.assertTrue(expectFail)
            return out
        samples = op.sample_inputs(device, dtype)
        for sample in samples:
            if 'dtype' not in sample.kwargs:
                helper(False, False, op, sample.input, *sample.args, **sample.kwargs)
                helper(True, False, op, sample.input, *sample.args, **sample.kwargs)
                sample.kwargs['dtype'] = torch.int16
                helper(False, False, op, sample.input, *sample.args, **sample.kwargs)
                helper(True, True, op, sample.input, *sample.args, **sample.kwargs)
                sample.kwargs['dtype'] = torch.int32
                helper(False, False, op, sample.input, *sample.args, **sample.kwargs)
                helper(True, False, op, sample.input, *sample.args, **sample.kwargs)
            else:
                helper(False, False, op, sample.input, *sample.args, **sample.kwargs)
                helper(True, sample.kwargs['dtype'] != torch.int32, op, sample.input,
                       *sample.args, **sample.kwargs)

    # Tests that the forward and backward passes of operations produce the
    #   same values for the cross-product of op variants (method, inplace)
    #   against eager's gold standard op function variant
    @_variant_ops(op_db)
    @skipIfTorchInductor("Inductor does not support complex dtype yet")
    def test_variant_consistency_eager(self, device, dtype, op):
        # Acquires variants (method variant, inplace variant, operator variant, inplace_operator variant, aliases)

        method = op.method_variant
        inplace = op.inplace_variant
        operator = op.operator_variant
        inplace_operator = op.inplace_operator_variant

        # list of all inplace ops: inplace variant + alias inplace variants if exist
        inplace_ops = [inplace, inplace_operator]
        variants_tmp = [method, inplace, operator, inplace_operator]
        operators = [operator, inplace_operator]

        for a_op in op.aliases:
            variants_tmp.append(a_op.op)
            variants_tmp.append(a_op.method_variant)
            variants_tmp.append(a_op.inplace_variant)
            inplace_ops.append(a_op.inplace_variant)

        inplace_variants = tuple(filter(None, inplace_ops))
        variants = tuple(filter(None, variants_tmp))
        operators = tuple(filter(None, operators))

        _requires_grad = dtype in op.supported_backward_dtypes(
            torch.device(device).type
        )

        include_conjugated_inputs = op.test_conjugated_samples and dtype.is_complex
        samples = op.sample_inputs(
            device,
            dtype,
            requires_grad=_requires_grad,
            include_conjugated_inputs=include_conjugated_inputs,
        )
        samples = list(samples)

        def _test_consistency_helper(samples, variants):
            for sample in samples:
                tensor = (
                    sample.input
                    if isinstance(sample.input, torch.Tensor)
                    else sample.input[0]
                )

                # Computes function forward and backward values
                tensor.grad = None
                expected_forward = op(sample.input, *sample.args, **sample.kwargs)
                expected_grad = None

                output_process_fn_grad = (
                    sample.output_process_fn_grad
                    if sample.output_process_fn_grad
                    else lambda x: x
                )

                # Skips inplace variants if the output dtype is not the same as
                #   the input dtype
                skip_inplace = False
                if (
                    isinstance(expected_forward, torch.Tensor)
                    and expected_forward.dtype is not tensor.dtype
                ):
                    skip_inplace = True

                if isinstance(
                    expected_forward, torch.Tensor
                ) and dtype in op.supported_backward_dtypes(torch.device(device).type):
                    out = output_process_fn_grad(expected_forward).sum()
                    if out.dtype.is_complex:
                        out = out.abs()
                    out.backward()
                    expected_grad = tensor.grad

                # Test eager consistency
                for variant in variants:
                    # Skips inplace ops
                    if variant in inplace_ops and skip_inplace:
                        continue

                    # Compares variant's forward
                    # Note: copies the to-be-modified input when testing the inplace variant
                    tensor.grad = None
                    cloned = (
                        clone_input_helper(sample.input)
                        if variant in inplace_ops
                        else sample.input
                    )

                    if variant in inplace_ops and sample.broadcasts_input:
                        with self.assertRaises(
                            RuntimeError,
                            msg=(
                                "inplace variant either incorrectly allowed "
                                f"resizing or you have marked the sample {sample.summary()}"
                                " incorrectly with `broadcasts_self=True"
                            ),
                        ):
                            variant_forward = variant(
                                cloned, *sample.args, **sample.kwargs
                            )
                        continue

                    if variant in operators and sample.kwargs:
                        # skip samples with kwargs for operator variants
                        continue

                    variant_forward = variant(cloned, *sample.args, **sample.kwargs)
                    self.assertEqual(expected_forward, variant_forward)

                    # Compares variant's backward
                    if expected_grad is not None and (
                        variant not in inplace_ops or op.supports_inplace_autograd
                    ):
                        out = output_process_fn_grad(variant_forward).sum()
                        if out.dtype.is_complex:
                            out = out.abs()
                        out.backward()
                        self.assertEqual(expected_grad, tensor.grad)

        _test_consistency_helper(samples, variants)

        def _test_inplace_preserve_storage(samples, variants):
            for sample in samples:
                # Skips inplace variants if the output dtype is not the same as
                #   the input dtype
                expected_forward = op(sample.input, *sample.args, **sample.kwargs)
                tensor = (
                    sample.input
                    if isinstance(sample.input, torch.Tensor)
                    else sample.input[0]
                )
                skip_inplace = False
                if (
                    isinstance(expected_forward, torch.Tensor)
                    and expected_forward.dtype is not tensor.dtype
                ):
                    skip_inplace = True
                if skip_inplace:
                    return
                for variant in variants:
                    cloned = (
                        clone_input_helper(sample.input)
                        if variant in inplace_ops
                        else sample.input
                    )
                    inp_tensor = (
                        cloned if isinstance(cloned, torch.Tensor) else cloned[0]
                    )
                    data_ptr = inp_tensor.data_ptr()
                    if variant in operators and sample.kwargs:
                        # skip samples with kwargs for operator variants
                        continue

                    variant_forward = variant(cloned, *sample.args, **sample.kwargs)
                    if isinstance(variant_forward, torch.Tensor):
                        self.assertEqual(
                            data_ptr, variant_forward.data_ptr(), atol=0, rtol=0
                        )
                    else:
                        self.assertTrue(
                            False,
                            "Non-tensor outputs for inplace ops are not supported",
                        )

        if len(inplace_ops) > 0:
            inplace_samples = list(
                filter(lambda sample: not sample.broadcasts_input, samples)
            )
            _test_inplace_preserve_storage(inplace_samples, inplace_variants)

    @ops(op_db, allowed_dtypes=(torch.bool,))
    @skipIfTorchInductor("Inductor does not support view with dtype yet")
    def test_non_standard_bool_values(self, device, dtype, op):
        # Test boolean values other than 0x00 and 0x01 (gh-54789)
        def convert_boolean_tensors(x):
            if not isinstance(x, torch.Tensor) or x.dtype != torch.bool:
                return x

            # Map False -> 0 and True -> Random value in [2, 255]
            true_vals = torch.randint(2, 255, x.shape, dtype=torch.uint8, device=x.device)
            false_vals = torch.zeros((), dtype=torch.uint8, device=x.device)
            x_int = torch.where(x, true_vals, false_vals)

            ret = x_int.view(torch.bool)
            self.assertEqual(ret, x)
            return ret

        for sample in op.sample_inputs(device, dtype):
            expect = op(sample.input, *sample.args, **sample.kwargs)

            transformed = sample.transform(convert_boolean_tensors)
            actual = op(transformed.input, *transformed.args, **transformed.kwargs)

            self.assertEqual(expect, actual)

    # Validates that each OpInfo specifies its forward and backward dtypes
    @unittest.skip("NPU doesn't support yet.")
    @skipMeta
    @ops(ops_and_refs, dtypes=OpDTypes.none)
    def test_dtypes(self, device, op):
        # Check complex32 support only if the op claims.
        device_type = torch.device(device).type
        include_complex32 = (
            (torch.complex32,)
            if op.supports_dtype(torch.complex32, device_type)
            else ()
        )

        # dtypes to try to backward in
        allowed_backward_dtypes = floating_and_complex_types_and(
            *((torch.half, torch.bfloat16) + include_complex32)
        )

        # lists for (un)supported dtypes
        supported_dtypes = set()
        unsupported_dtypes = set()
        supported_backward_dtypes = set()
        unsupported_backward_dtypes = set()
        dtype_error: Dict[torch.dtype, Exception] = dict()

        def unsupported(dtype, e):
            dtype_error[dtype] = e
            unsupported_dtypes.add(dtype)
            if dtype in allowed_backward_dtypes:
                unsupported_backward_dtypes.add(dtype)

        for dtype in all_types_and_complex_and(
            *((torch.half, torch.bfloat16, torch.bool) + include_complex32)
        ):
            # tries to acquire samples - failure indicates lack of support
            requires_grad = dtype in allowed_backward_dtypes
            try:
                samples = tuple(
                    op.sample_inputs(device, dtype, requires_grad=requires_grad)
                )
            except Exception as e:
                unsupported(dtype, e)
                continue

            for sample in samples:
                # tries to call operator with the sample - failure indicates
                #   lack of support
                try:
                    result = op(sample.input, *sample.args, **sample.kwargs)
                    supported_dtypes.add(dtype)
                except Exception as e:
                    # NOTE: some ops will fail in forward if their inputs
                    #   require grad but they don't support computing the gradient
                    #   in that type! This is a bug in the op!
                    unsupported(dtype, e)
                    continue

                # Checks for backward support in the same dtype, if the input has
                # one or more tensors requiring grad
                def _tensor_requires_grad(x):
                    if isinstance(x, dict):
                        for v in x.values():
                            if _tensor_requires_grad(v):
                                return True
                    if isinstance(x, (list, tuple)):
                        for a in x:
                            if _tensor_requires_grad(a):
                                return True
                    if isinstance(x, torch.Tensor) and x.requires_grad:
                        return True

                    return False

                requires_grad = _tensor_requires_grad(sample.input) \
                    or _tensor_requires_grad(sample.args) or _tensor_requires_grad(sample.kwargs)
                if not requires_grad:
                    continue

                try:
                    result = sample.output_process_fn_grad(result)
                    if isinstance(result, torch.Tensor):
                        backward_tensor = result
                    elif isinstance(result, Sequence) and isinstance(
                        result[0], torch.Tensor
                    ):
                        backward_tensor = result[0]
                    else:
                        continue

                    # Note: this grad may not have the same dtype as dtype
                    # For functions like complex (float -> complex) or abs
                    #   (complex -> float) the grad tensor will have a
                    #   different dtype than the input.
                    #   For simplicity, this is still modeled as these ops
                    #   supporting grad in the input dtype.
                    grad = torch.randn_like(backward_tensor)
                    backward_tensor.backward(grad)
                    supported_backward_dtypes.add(dtype)
                except Exception as e:
                    dtype_error[dtype] = e
                    unsupported_backward_dtypes.add(dtype)

        # Checks that dtypes are listed correctly and generates an informative
        #   error message

        supported_forward = supported_dtypes - unsupported_dtypes
        partially_supported_forward = supported_dtypes & unsupported_dtypes
        unsupported_forward = unsupported_dtypes - supported_dtypes
        supported_backward = supported_backward_dtypes - unsupported_backward_dtypes
        partially_supported_backward = (
            supported_backward_dtypes & unsupported_backward_dtypes
        )
        unsupported_backward = unsupported_backward_dtypes - supported_backward_dtypes

        device_type = torch.device(device).type

        claimed_forward = set(op.supported_dtypes(device_type))
        supported_but_unclaimed_forward = supported_forward - claimed_forward
        claimed_but_unsupported_forward = claimed_forward & unsupported_forward

        claimed_backward = set(op.supported_backward_dtypes(device_type))
        supported_but_unclaimed_backward = supported_backward - claimed_backward
        claimed_but_unsupported_backward = claimed_backward & unsupported_backward

        # Partially supporting a dtype is not an error, but we print a warning
        if (len(partially_supported_forward) + len(partially_supported_backward)) > 0:
            msg = f"Some dtypes for {op.name} on device type {device_type} are only partially supported!\n"
            if len(partially_supported_forward) > 0:
                msg = (
                    msg
                    + "The following dtypes only worked on some samples during forward: {}.\n".format(
                        partially_supported_forward
                    )
                )
            if len(partially_supported_backward) > 0:
                msg = (
                    msg
                    + "The following dtypes only worked on some samples during backward: {}.\n".format(
                        partially_supported_backward
                    )
                )
            print(msg)

        if (
            len(supported_but_unclaimed_forward)
            + len(claimed_but_unsupported_forward)
            + len(supported_but_unclaimed_backward)
            + len(claimed_but_unsupported_backward)
        ) == 0:
            return

        # Reference operators often support additional dtypes, and that's OK
        if op in python_ref_db:
            if (
                len(claimed_but_unsupported_forward)
                + len(claimed_but_unsupported_backward)
            ) == 0:
                return

        # Generates error msg
        msg = f"The supported dtypes for {op.name} on device type {device_type} are incorrect!\n"
        if len(supported_but_unclaimed_forward) > 0:
            msg = (
                msg
                + "The following dtypes worked in forward but are not listed by the OpInfo: {}.\n".format(
                    supported_but_unclaimed_forward
                )
            )
        if len(supported_but_unclaimed_backward) > 0:
            msg = (
                msg
                + "The following dtypes worked in backward but are not listed by the OpInfo: {}.\n".format(
                    supported_but_unclaimed_backward
                )
            )
        if len(claimed_but_unsupported_forward) > 0:
            msg = (
                msg
                + "The following dtypes did not work in forward but are listed by the OpInfo: {}.\n".format(
                    claimed_but_unsupported_forward
                )
            )
        if len(claimed_but_unsupported_backward) > 0:
            msg = (
                msg
                + "The following dtypes did not work in backward but are listed by the OpInfo: {}.\n".format(
                    claimed_but_unsupported_backward
                )
            )

        all_claimed_but_unsupported = set.union(claimed_but_unsupported_backward, claimed_but_unsupported_forward)
        if all_claimed_but_unsupported:
            msg += "Unexpected failures raised the following errors:\n"
            for dtype in all_claimed_but_unsupported:
                msg += f"{dtype} - {dtype_error[dtype]}\n"

        self.fail(msg)


class TestCompositeCompliance(TestCase):
    # Checks if the operator (if it is composite) is written to support most
    # backends and Tensor subclasses. See "CompositeImplicitAutograd Compliance"
    # in aten/src/ATen/native/README.md for more details
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_operator(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=False)

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            composite_compliance.check_with_mode(op, args, kwargs, self.assertEqual)
            composite_compliance.check_all_permutations(op, args, kwargs, self.assertEqual)

    @ops([op for op in op_db if op.supports_autograd], allowed_dtypes=(torch.float,))
    def test_backward(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            # We pass assertEqual so that decorators like `toleranceOverride`
            # actually work (otherwise they silently do nothing!)
            composite_compliance.check_backward_formula(
                op.get_op(), args, kwargs,
                sample.output_process_fn_grad,
                op.gradcheck_wrapper, self.assertEqual)

    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_forward_ad(self, device, dtype, op):
        if torch.float not in op.supported_backward_dtypes(device):
            raise unittest.SkipTest("Does not support autograd")

        if not op.supports_forward_ad:
            raise unittest.SkipTest("Does not support forward_ad")

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            # We pass assertEqual so that decorators like `toleranceOverride`
            # actually work (otherwise they silently do nothing!)
            composite_compliance.check_forward_ad_formula(
                op.get_op(), args, kwargs, op.gradcheck_wrapper, self.assertEqual)


class TestMathBits(TestCase):
    # Tests that
    # 1. The operator's output for physically conjugated/negated tensors and conjugate/negative view tensors
    # produces the same value
    # 2. The gradients are same in both cases mentioned in (1)
    # 3. If the operator's inplace variant is supported, tests that the inplace operation
    #    produces the correct value when called on a conjugate/negative view tensor and that the output
    #    has its conj/neg bit set to true
    # This test only runs for C -> R and C -> C functions
    # Note: This test runs for functions that take both tensors and tensorlists as input.
    def _test_math_view(
        self,
        device,
        dtype,
        op,
        samples,
        math_op_physical,
        math_op_view,
        is_bit_set,
        out_type,
    ):
        inplace_variant = op.inplace_variant

        # helper function to clone and conjugate/negate the input if its a tensor
        # else clone the sequence and conjugate/negate the first element in the sequence
        # If a requires_grad argument is provided the tensor being conjugated/negated will
        # have its requires_grad set to that value.
        def clone_and_perform_view(input_, **kwargs):
            if isinstance(input_, torch.Tensor):
                requires_grad = kwargs.get("requires_grad", input_.requires_grad)
                with torch.no_grad():
                    # Ensure view represents the original sample input
                    input_ = math_op_physical(input_)
                # Note: .conj() is not called under no_grad mode since it's not allowed to modify a
                # view created in no_grad mode. Here it's ok to do so, so as a workaround we call conj
                # before resetting the requires_grad field for input
                input_ = math_op_view(input_)
                if not input_.is_leaf:
                    raise RuntimeError("input is not leaf node")
                return input_.requires_grad_(requires_grad)

            if isinstance(input_, Sequence):
                out = list(map(clone_input_helper, input_))
                out[0] = clone_and_perform_view(out[0])
                return tuple(out)

        for sample in samples:
            tensor = (
                sample.input
                if isinstance(sample.input, torch.Tensor)
                else sample.input[0]
            )
            cloned1 = clone_and_perform_view(sample.input)

            # Computes function forward value with a physically conjugated/negated tensor and
            # a conj/neg view tensor and verifies that the output in both case are equal.
            expected_forward = op(sample.input, *sample.args, **sample.kwargs)
            forward_with_mathview = op(cloned1, *sample.args, **sample.kwargs)
            self.assertEqual(expected_forward, forward_with_mathview)

            # If the op has an inplace variant, and the input doesn't require broadcasting
            # and has the same dtype as output, verify that the inplace operation on a conjugated/negated
            # input produces correct output, and the output tensor has the conj/neg bit set to True
            if inplace_variant is not None and not sample.broadcasts_input:
                cloned2 = clone_and_perform_view(tensor, requires_grad=False)
                if (
                    isinstance(expected_forward, torch.Tensor)
                    and expected_forward.dtype is tensor.dtype
                ):
                    inplace_forward = inplace_variant(
                        cloned2, *sample.args, **sample.kwargs
                    )
                    self.assertTrue(is_bit_set(inplace_forward))
                    self.assertEqual(inplace_forward, expected_forward)

            if (
                isinstance(expected_forward, torch.Tensor)
                and expected_forward.requires_grad
            ):
                output_process_fn_grad = sample.output_process_fn_grad or (lambda x: x)
                expected_forward = output_process_fn_grad(expected_forward)
                forward_with_mathview = output_process_fn_grad(forward_with_mathview)

                tensor = (
                    sample.input
                    if isinstance(sample.input, torch.Tensor)
                    else sample.input[0]
                )
                expected_forward.sum().abs().backward(retain_graph=True)
                forward_with_mathview.sum().abs().backward(retain_graph=True)
                if tensor.grad is not None:
                    cloned1_tensor = (
                        cloned1 if isinstance(cloned1, torch.Tensor) else cloned1[0]
                    )
                    self.assertEqual(tensor.grad, cloned1_tensor.grad)

                    tensor.grad, cloned1_tensor.grad = None, None

                    # a repeat of the above test if output is not complex valued
                    if out_type(expected_forward):
                        grad = torch.randn_like(expected_forward)
                        expected_forward.backward(grad)
                        forward_with_mathview.backward(
                            math_op_view(math_op_physical(grad))
                        )

                        self.assertEqual(tensor.grad, cloned1_tensor.grad)

    @ops(ops_and_refs, allowed_dtypes=(torch.cfloat,))
    @skipIfTorchInductor("Inductor does not support complex dtype yet")
    def test_conj_view(self, device, dtype, op):
        if not op.test_conjugated_samples:
            self.skipTest("Operation doesn't support conjugated inputs.")
        math_op_physical = torch.conj_physical
        math_op_view = torch.conj
        _requires_grad = torch.cfloat in op.supported_backward_dtypes(
            torch.device(device).type
        )
        is_bit_set = torch.is_conj
        samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad)
        self._test_math_view(
            device,
            dtype,
            op,
            samples,
            math_op_physical,
            math_op_view,
            is_bit_set,
            torch.is_complex,
        )

    @ops(ops_and_refs, allowed_dtypes=(torch.double,))
    @skipIfTorchInductor("Inductor does not support complex dtype yet")
    def test_neg_view(self, device, dtype, op):
        if not op.test_neg_view:
            self.skipTest("Operation not tested with tensors with negative bit.")
        math_op_physical = torch.neg
        math_op_view = torch._neg_view
        is_bit_set = torch.is_neg
        samples = op.sample_inputs(device, dtype, requires_grad=op.supports_autograd)
        self._test_math_view(
            device,
            dtype,
            op,
            samples,
            math_op_physical,
            math_op_view,
            is_bit_set,
            lambda x: True,
        )

    @ops(ops_and_refs, allowed_dtypes=(torch.cdouble,))
    @skipIfTorchInductor("Inductor does not support complex dtype yet")
    def test_neg_conj_view(self, device, dtype, op):
        if not op.test_neg_view:
            self.skipTest("Operation not tested with tensors with negative bit.")
        if not op.test_conjugated_samples:
            self.skipTest("Operation doesn't support conjugated inputs.")

        def math_op_physical(x):
            return -x.conj_physical()

        def math_op_view(x):
            return torch._neg_view(x).conj()

        def is_bit_set(x):
            return torch.is_neg(x) and torch.is_conj(x)

        _requires_grad = dtype in op.supported_backward_dtypes(
            torch.device(device).type
        )
        samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad)
        # Only test one sample
        samples = itertools.islice(samples, 1)
        self._test_math_view(
            device,
            dtype,
            op,
            samples,
            math_op_physical,
            math_op_view,
            is_bit_set,
            torch.is_complex,
        )


def check_inplace_view(func, input_, rs, input_size, input_strides):
    if func is None:
        return
    # which mutate not necessarily the first input.
    if isinstance(rs, torch.Tensor) and rs is input_:
        unequal_size = rs.size() != input_size
        unequal_strides = rs.stride() != input_strides
        # resize_ should probably have inplace_view tag. Not adding the tag since it
        # breaks some codegen logic
        if (unequal_size or unequal_strides):
            if isinstance(func, torch._ops.OpOverloadPacket):
                func = func.default
            if func is not torch.ops.aten.resize_.default:
                if torch.Tag.inplace_view not in func.tags:
                    raise RuntimeError("torch.Tag.inplace_view not in func.tags")


class TestTagsMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if isinstance(args[0], torch.Tensor):
            old_size = args[0].size()
            old_stride = args[0].stride()
            rs = func(*args, **kwargs)
            check_inplace_view(func, args[0], rs, old_size, old_stride)
        else:
            rs = func(*args, **kwargs)
        return rs


fake_skips = (
    "aminmax",  # failing input
    "cov",  # aweights cannot be negtaive
    "istft",  # window overlap add min: 0
    "linalg.eigvals",  # The tensor has a non-zero number of elements, but its data is not allocated yet
    "linalg.eigvalsh",  # aten::linalg_eigvalsh.out' with arguments from the 'Meta' backend
    "linalg.matrix_power",  # Could not run 'aten::eye.m_out' with arguments from the 'Meta' backend
    # "linalg.pinv",  # Could not run 'aten::pinv.out' with arguments from the 'Meta' backen
    "linalg.matrix_rank.hermitian",  # Could not run 'aten::linalg_eigvalsh.out' with arguments from the 'Meta' backend
    "linalg.pinv.hermitian",  # tensor.mH is only supported on matrices or batches of matrices. Got 1-D tensor
    "linalg.solve",  # Could not run 'aten::linalg_solve' with arguments from the 'Meta' backend
    "linalg.tensorsolve",  # Could not run 'aten::linalg_solve' with arguments from the 'Meta'
    "lu_solve",  # MALLOC ERROR: debug
    "multinomial",  # Could not run 'aten::multinomial' with arguments from the 'Meta' backend
    "mvlgamma.mvlgamma_p_1",  # Could not run 'aten::_local_scalar_dense' with arguments from the 'Meta' backend
    "mvlgamma.mvlgamma_p_3",  # Could not run 'aten::_local_scalar_dense' with arguments from the 'Meta' backend
    "mvlgamma.mvlgamma_p_5",  # Could not run 'aten::_local_scalar_dense' with arguments from the 'Meta' backend
    "nanmean",  # logical_not() got an unexpected keyword argument 'out'
    "quantile",  # quantile() q values must be in the range [0, 1]
    "nanquantile",  # quantile() q values must be in the range [0, 1]
    "nn.functional.ctc_loss",  # The tensor has a non-zero number of elements, but its data is not allocated yet
    "nn.functional.embedding_bag",  # sometimes errors
    "nn.functional.nll_loss",  # sometimes errors
    "nn.functional.max_pool1d",  # The tensor has a non-zero number of elements
    "to_sparse",  # Could not run 'aten::_to_sparse' with arguments from the 'Meta' backend
    "tensor_split",  # The tensor has a non-zero number of elements, but its data is not allocated yet
    "repeat_interleave",  # cannot repeat_interleave a meta tensor without output_size
    "_segment_reduce.lengths",  # Could not run 'aten::segment_reduce' with arguments from the 'Meta' backend.
    "sparse.sampled.addmm",  # sparsity not supported
    # Can not infer total number of classes from meta. no way at present to throw DynamicOutputShapeException
    "nn.functional.one_hot",
    "narrow",  # Fails only for one overload with DataDependentOutputException (hence skip).
)

fake_autocast_device_skips = defaultdict(dict)

fake_autocast_device_skips["cpu"] = {"linalg.pinv"}


dynamic_output_op_tests = (
    "argwhere",
    "bincount",
    "combinations",
    "linalg.lstsq",
    "masked_select",
    "nonzero",
    "unique_consecutive",
    "unique",
    "linalg.lstsq.grad_oriented",
)

# some inputs invoke dynamic output shape operators, some do not
sometimes_dynamic_output_op_test = (
    "__getitem__",
    "index_select",
)

data_dependent_op_tests = (
    "equal",
    "corrcoef",
    "nn.functional.gaussian_nll_loss",
    "allclose",
)

aliasing_failures = (
    "histogramdd",
)

fake_backward_skips = {
    "linalg.cond",
    "linalg.matrix_norm",
    "linalg.norm",
    "linalg.svd",
    "linalg.svdvals",
    "pca_lowrank",
    "roll",
    "svd_lowrank",
    "sgn",
}

fake_backward_xfails = {skip(s) for s in fake_backward_skips} | {
    xfail("_segment_reduce", "lengths"),
    xfail("fft.ihfftn"),  # Mismatch in aten._conj_physical.default
    xfail("fft.ihfft2"),  # Mismatch in aten._conj_physical.default
    skip('nn.functional.ctc_loss'),
}

fake_autocast_backward_xfails = {
    skip("nn.functional.binary_cross_entropy"),
    skip("sparse.sampled_addmm"),
    skip("linalg.pinv"),
    skip("linalg.pinv", "hermitian"),
    skip("linalg.pinv", "singular"),
    skip('pinverse'),
}


class TestFakeTensor(TestCase):
    def _test_fake_helper(self, device, dtype, op, context):
        name = op.name
        if op.variant_test_name:
            name += "." + op.variant_test_name
        if name in fake_skips or "sparse" in name or "jiterator" in name:
            self.skipTest("Skip failing test")

        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample in samples:
            try:
                mode = FakeTensorMode()

                def map_to_fake(e):
                    if isinstance(e, torch.Tensor):
                        return mode.from_tensor(e)
                    else:
                        return e

                input_ = tree_map(map_to_fake, sample.input)
                args = tree_map(map_to_fake, sample.args)
                kwargs = tree_map(map_to_fake, sample.kwargs)

                try:
                    with context():
                        res = op(sample.input, *sample.args, **sample.kwargs)
                except Exception as e:
                    continue

                with context():
                    with mode:
                        res_fake = op(input_, *args, **kwargs)

                for fake_out, real_out in zip(
                    tree_flatten(res_fake)[0], tree_flatten(res)[0]
                ):
                    if not isinstance(fake_out, torch.Tensor):
                        self.assertTrue(not isinstance(real_out, torch.Tensor))
                        continue

                    self.assertTrue(isinstance(fake_out, FakeTensor))
                    # if you see a shape exception here, you may need to add
                    # a `dynamic_output_shape` tag to an operator

                    # prims/decomps must correctly model strides,
                    prims.utils.compare_tensor_meta(fake_out, real_out, True)

                    if name not in aliasing_failures:
                        fake_aliasing = outputs_alias_inputs((input_, args, kwargs), res_fake)
                        real_aliasing = outputs_alias_inputs((sample.input, sample, args, sample.kwargs), res)
                        self.assertEqual(fake_aliasing, real_aliasing)

                self.assertTrue(name not in dynamic_output_op_tests and name not in data_dependent_op_tests)

            except torch._subclasses.fake_tensor.UnsupportedFakeTensorException:
                pass
            except torch._subclasses.fake_tensor.UnsupportedOperatorException:
                pass
            except torch._subclasses.fake_tensor.DynamicOutputShapeException:
                self.assertTrue(name in dynamic_output_op_tests or name in sometimes_dynamic_output_op_test)
            except torch._subclasses.fake_tensor.DataDependentOutputException:
                self.assertTrue(name in data_dependent_op_tests)

    @ops(op_db, dtypes=OpDTypes.any_one)
    def test_pointwise_ops(self, device, dtype, op):
        name = op.name
        if op.variant_test_name:
            name += "." + op.variant_test_name
        if name in fake_skips or "sparse" in name or "jiterator" in name:
            self.skipTest("Skip failing test")

        test_self = self

        class TestPointwiseMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}

                out = func(*args, **kwargs)

                if torch.Tag.pointwise in func.tags:
                    shapes = []
                    for inp in tree_flatten((args, kwargs)):
                        if isinstance(inp, torch.Tensor):
                            shapes.append(inp.shape)

                    out_shape = torch._refs._broadcast_shapes(*shapes)

                    for out_elem in tree_flatten(out):
                        if isinstance(out_elem, torch.Tensor):
                            test_self.assertEqual(out_elem.shape, out_shape)

                return out

        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample in samples:
            mode = FakeTensorMode()

            def map_to_fake(e):
                if isinstance(e, torch.Tensor):
                    return mode.from_tensor(e)
                else:
                    return e

            input = tree_map(map_to_fake, sample.input)
            args = tree_map(map_to_fake, sample.args)
            kwargs = tree_map(map_to_fake, sample.kwargs)

            try:
                op(input, *args, **kwargs)
            except Exception as e:
                continue

            with TestPointwiseMode():
                with mode:
                    op(input, *args, **kwargs)

    @ops(op_db, dtypes=OpDTypes.any_one)
    def test_fake(self, device, dtype, op):
        self._test_fake_helper(device, dtype, op, contextlib.nullcontext)

    @ops(op_db, dtypes=OpDTypes.any_one)
    def test_fake_autocast(self, device, dtype, op):
        if op.name in fake_autocast_device_skips[device]:
            self.skipTest("Skip failing test")
        context = torch.npu.amp.autocast if device == "npu" else torch.cpu.amp.autocast
        self._test_fake_helper(device, dtype, op, context)

    def _test_fake_crossref_helper(self, device, dtype, op, context):
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        for _, sample in enumerate(samples):
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            # skip these to speed up tests
            common_skip_ops = (
                aten.detach.default,
                aten.empty_strided.default,
                aten.copy_.default,
                aten.is_same_size.default,
            )

            try:
                with torch._subclasses.CrossRefFakeMode(ignore_op_fn=lambda fn: fn in common_skip_ops, check_aliasing=True):
                    with warnings.catch_warnings(), context(), torch.autograd.set_multithreading_enabled(False):
                        composite_compliance.compute_expected_grads(
                            op.get_op(), args, kwargs,
                            sample.output_process_fn_grad,
                            op.gradcheck_wrapper)
            except torch._subclasses.fake_tensor.UnsupportedOperatorException:
                pass

    @ops([op for op in op_db if op.supports_autograd], allowed_dtypes=(torch.float,))
    @skipOps('TestFakeTensor', 'test_fake_crossref_backward_no_amp', fake_backward_xfails)
    @unittest.skip("skip test fake crossref backward no amp now")
    def test_fake_crossref_backward_no_amp(self, device, dtype, op):
        self._test_fake_crossref_helper(device, dtype, op, contextlib.nullcontext)

    @ops([op for op in op_db if op.supports_autograd], allowed_dtypes=(torch.float,))
    @skipOps('TestFakeTensor', 'test_fake_crossref_backward_amp', fake_backward_xfails | fake_autocast_backward_xfails)
    @unittest.skip("skip test fake crossref backward amp now")
    def test_fake_crossref_backward_amp(self, device, dtype, op):
        self._test_fake_crossref_helper(device, dtype, op, torch.npu.amp.autocast)


instantiate_device_type_tests(TestCommon, globals(), only_for='privateuse1')
instantiate_device_type_tests(TestCompositeCompliance, globals(), only_for='privateuse1')
instantiate_device_type_tests(TestMathBits, globals(), only_for='privateuse1')
instantiate_device_type_tests(TestFakeTensor, globals(), only_for='privateuse1')


"""
Below defines params needed to run ALL test suites and collect corresponding failed cases.
test logs is stored in test_name.log. File will be removed automatically if process exits
with zero.

IO_path: path to store logs. Uses need to manually create a folder to store the log files.
res_log: file name to store all failed test names.
"""

IO_path = "logs"
res_log = "result.log"
flags = os.O_WRONLY | os.O_RDONLY | os.O_CREAT
modes = stat.S_IWUSR | stat.S_IRUSR


def get_list(all_test_name_log):
    all_attr = dir(TestCommonPRIVATEUSE1) + dir(TestCompositeCompliancePRIVATEUSE1) + dir(TestMathBitsPRIVATEUSE1) + \
        dir(TestFakeTensorPRIVATEUSE1)
    with os.fdopen(os.open(all_test_name_log, flags, modes), "a") as f:
        for i in all_attr:
            if i.startswith("test_"):
                f.write(i + "\n")


def check_file_IO(log_file):
    size = os.path.getsize(log_file)
    time.sleep(30)
    new_size = os.path.getsize(log_file)
    return size == new_size


def _read_file(t_name):
    log_name = os.path.join(IO_path, '{}.log'.format(t_name))
    success = False
    if os.path.exists(log_name):
        while not check_file_IO(log_name):
            pass
        with open(log_name, 'r', encoding='utf-8') as f:
            tmp = f.readlines()
            for t in tmp:
                if "OK" in t:
                    success = True
                    os.remove(log_name)
                    return
        if not success:
            with os.fdopen(os.open(res_log, flags, modes), "a") as f:
                f.write(t_name + '\n')
        if os.path.exists(log_name):
            os.remove(log_name)


def start_thread(t_name):
    thread_io = threading.Thread(target=_read_file, args=(t_name,))
    thread_io.start()


if __name__ == "__main__":
    check_end = sys.argv[-1].isdigit()
    if check_end:
        device_id, test_name = sys.argv[-1], sys.argv[-2]
        torch_npu.npu.set_device(int(device_id))
        start_thread(test_name)
        run_tests(sys.argv[:-1])
    else:
        run_tests(sys.argv[:])
