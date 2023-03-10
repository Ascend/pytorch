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

from typing import Sequence, List
from functools import partial

import torch
from torch.testing._internal.common_methods_invocations import SampleInput
from torch.testing._internal.common_dtype import floating_and_complex_types_and
from torch.testing._internal.common_utils import (clone_input_helper, 
                                                  first_sample, 
                                                  is_iterable_of_tensors)

from torch_npu.testing.common_methods_invocations import op_db
from torch_npu.testing.decorator import Dtypes, Formats, instantiate_ops_tests
from torch_npu.testing.testcase import TestCase, run_tests


def trans_device_and_dtype(sample, origin, target, npu_format=2, to_npu=False):
    def transform(sample, f):
        def tt(t):
            def _tt(t):
                return f(t)

            if isinstance(t, torch.Tensor):
                return _tt(t)
            elif isinstance(t, torch.dtype):
                return _tt(t)
            elif isinstance(t, list):
                return list(map(tt, t))
            elif isinstance(t, tuple):
                return tuple(map(tt, t))
            elif isinstance(t, dict):
                return {k: tt(v) for k, v in t.items()}
            else:
                return t

        sample_tt_input, tt_args, tt_kwargs = tt(sample.input), tt(sample.args), tt(sample.kwargs)
        return (sample_tt_input, tt_args, tt_kwargs)

    def _trans_helper(arg):
        if isinstance(arg, torch.Tensor):
            if to_npu:
                arg = arg.to('npu')
            if arg.dtype == origin:
                arg = arg.to(target)
                if to_npu:
                    arg.npu_format_cast(npu_format)

        return arg
    
    sample_helper = transform(sample, _trans_helper)
    res = SampleInput(input=sample_helper[0], 
                      args=sample_helper[1], 
                      kwargs=sample_helper[2], 
                      broadcasts_input=sample.broadcasts_input)
    return res


@instantiate_ops_tests(op_db)
class TestOps(TestCase):

    def test_correctness(self, dtype, op, npu_format):

        def _generate_sample_inputs_requried_grad(sample_input, args):
            res = []

            if isinstance(sample_input, torch.Tensor):
                res.append(sample_input)
            elif isinstance(sample_input, Sequence) and isinstance(sample_input[0], torch.Tensor):
                res.extend(sample_input)
            
            if isinstance(args, torch.Tensor):
                res.append(args)
            elif isinstance(args, Sequence):             
                for arg in args:
                    if isinstance(arg, torch.Tensor) and (arg.grad_fn or arg.requires_grad):
                        res.append(arg)
            
            return res

        unsupported_dtypes_cpu = {dtype for dtype in op.dtypesIfNPU if dtype not in op.dtypes}
        allowed_backward_dtypes = floating_and_complex_types_and(*(torch.half, torch.bfloat16))
        requires_grad = (dtype in allowed_backward_dtypes and op.supports_autograd)

        samples = op.sample_inputs('cpu', dtype, requires_grad=requires_grad)

        for index, sample in enumerate(samples):
            if op.skipSample and index in op.skipSample.get('test_correctness', {}):
                continue

            cpu_sample = sample
            if dtype in unsupported_dtypes_cpu and dtype == torch.float16:
                cpu_sample = trans_device_and_dtype(sample, dtype, torch.float32)

            expected = op(cpu_sample.input, *cpu_sample.args, **cpu_sample.kwargs)

            npu_sample = trans_device_and_dtype(sample, dtype, dtype, npu_format, to_npu=True)
            actual = op(npu_sample.input, *npu_sample.args, **npu_sample.kwargs)

            self.assertRtolEqual(expected, actual, auto_trans_dtype = True, message=f'sampleinput #{index} fail')

            if not requires_grad:
                continue

            expected = cpu_sample.output_process_fn_grad(expected)
            actual = npu_sample.output_process_fn_grad(actual)

            backward_cpu_tensor = expected.sum()
            backward_npu_tensor = actual.sum()

            sample_input_required_grad_cpu = _generate_sample_inputs_requried_grad(cpu_sample.input, cpu_sample.args)
            sample_input_required_grad_npu = _generate_sample_inputs_requried_grad(npu_sample.input, npu_sample.args)

            grads_cpu = torch.autograd.grad(outputs=backward_cpu_tensor, 
                                            inputs=sample_input_required_grad_cpu, 
                                            grad_outputs=torch.ones_like(backward_cpu_tensor))
            grads_npu = torch.autograd.grad(outputs=backward_npu_tensor, 
                                            inputs=sample_input_required_grad_npu, 
                                            grad_outputs=torch.ones_like(backward_npu_tensor))

            self.assertRtolEqual(grads_cpu, grads_npu, auto_trans_dtype=True, message=f'sampleinput #{index} fail')


    @Formats(2)
    @Dtypes(torch.float32)
    def test_variant_consistency_eager(self, dtype, op, npu_format):
        
        method = op.method_variant
        inplace = op.inplace_variant

        # list of all inplace ops: inplace variant + alias inplace variants if exist
        inplace_ops = [inplace, ]
        variants = [method, inplace, ]

        for a_op in op.aliases:
            variants.append(a_op.op)
            variants.append(a_op.method_variant)
            variants.append(a_op.inplace_variant)
            inplace_ops.append(a_op.inplace_variant)

        inplace_variants = tuple(filter(None, inplace_ops))
        variants = tuple(filter(None, variants))

        allowed_backward_dtypes = floating_and_complex_types_and(
            *(torch.half, torch.bfloat16))

        requires_grad = (dtype in allowed_backward_dtypes and op.supports_autograd)

        samples = op.sample_inputs('cpu',
                                   dtype,
                                   requires_grad=requires_grad,
                                   include_conjugated_inputs=True)

        def _test_consistency_helper(samples, variants):
            for index, sample in enumerate(samples):
                if op.skipSample and index in op.skipSample.get('test_variant_consistency_eager', {}):
                    continue

                sample = trans_device_and_dtype(sample, dtype, dtype, npu_format, to_npu=True)

                tensor = (
                    sample.input
                    if isinstance(sample.input, torch.Tensor)
                    else sample.input[0]
                )

                # Computes function forward and backward values
                tensor.grad = None
                expected_forward = op(sample.input, *sample.args, **sample.kwargs)
                expected_grad = None

                output_process_fn_grad = sample.output_process_fn_grad or (lambda x: x)

                # Skips inplace variants if the output dtype is not the same as
                #   the input dtype
                skip_inplace = False
                if isinstance(expected_forward, torch.Tensor) and expected_forward.dtype is not tensor.dtype:
                    skip_inplace = True

                if isinstance(expected_forward, torch.Tensor) and requires_grad:
                    output_process_fn_grad(expected_forward).sum().backward()
                    expected_grad = tensor.grad

                for variant in variants:
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
                                f"inplace variant either incorrectly allowed "
                                f"resizing or you have marked the sample {index}"
                                " incorrectly with `broadcasts_self = True".format(
                                    sample.summary()
                                )
                            )):
                            variant_forward = variant(
                                cloned, *sample.args, **sample.kwargs
                            )
                        continue

                    variant_forward = variant(cloned, *sample.args, **sample.kwargs)
                    self.assertRtolEqual(expected_forward, variant_forward, message=f'sampleinput #{index} fail')

                    if not requires_grad:
                        continue

                    if expected_grad is not None and (
                        variant not in inplace_ops or op.supports_inplace_autograd
                    ):  
                        output_process_fn_grad(variant_forward).sum().backward()
                        self.assertRtolEqual(expected_grad, tensor.grad, message=f'sampleinput #{index} fail')

        _test_consistency_helper(samples, variants)

        def _test_inplace_preserve_storage(samples, variants):
            for sample in samples:

                sample = trans_device_and_dtype(sample, dtype, dtype, npu_format, to_npu=True)
                # Skips inplace variants if the output dtype is not the same as
                #   the input dtype
                expected_forward = op(sample.input, *sample.args, **sample.kwargs)
                tensor = sample.input if isinstance(sample.input, torch.Tensor) else sample.input[0]
                skip_inplace = False

                if isinstance(expected_forward, torch.Tensor) and expected_forward.dtype is not tensor.dtype:
                    skip_inplace = True

                if skip_inplace:
                    return

                for variant in variants:
                    cloned = clone_input_helper(sample.input) if variant in inplace_ops else sample.input
                    inp_tensor = cloned if isinstance(cloned, torch.Tensor) else cloned[0]
                    data_ptr = inp_tensor.data_ptr()
                    variant_forward = variant(cloned,
                                              *sample.args,
                                              **sample.kwargs)

                    if isinstance(variant_forward, torch.Tensor):
                        self.assertRtolEqual(data_ptr, variant_forward.data_ptr())
                    else:
                        self.assertTrue(False, "Non-tensor outputs for inplace ops are not supported")

        if inplace_ops:
            inplace_samples = list(filter(lambda sample: not sample.broadcasts_input, samples))
            _test_inplace_preserve_storage(inplace_samples, inplace_variants)


    @Formats(2)
    @Dtypes(torch.float32)
    def test_out(self, op, dtype, npu_format):

        if not op.supports_out:
            self.skipTest("Skipped! Op doesn't support out= kwarg.")

        # Prefers running in float32 but has a fallback for the first listed supported dtype
        supported_dtypes = op.supported_dtypes('npu')
        if len(supported_dtypes) == 0:
            self.skipTest("Skipped! Op has not supported dtypes on this device.")
        dtype = torch.float32 if torch.float32 in supported_dtypes else list(supported_dtypes)[0]

        # NOTE: only tests on first sample
        samples = op.sample_inputs('cpu', dtype)
        sample = first_sample(self, samples)
        sample = trans_device_and_dtype(sample, dtype, dtype, npu_format, to_npu=True)

        # calls it normally to get the expected result
        expected = op(sample.input, *sample.args, **sample.kwargs)
        op_out = partial(op, sample.input, *sample.args, **sample.kwargs)

        if not isinstance(expected, torch.Tensor) and not is_iterable_of_tensors(expected, include_empty=True):
            self.skipTest("Skipped! Only supports single tensor or iterable of tensor outputs.")

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
            return tuple(map(lambda t: t.stride(), out))

        # Extracts data pointers from a tensor or iterable of tensors into a tuple
        # NOTE: only extracts on the CPU and CUDA device types since some
        #   device types don't have storage
        def _extract_data_ptrs(out):
            if isinstance(out, torch.Tensor):
                return (out.data_ptr(),)

            # assumes (see above) that out is an iterable of tensors
            return tuple(map(lambda t: t.data_ptr(), out))

        def _compare_out(transform, *, compare_strides_and_data_ptrs=True):
            out = _apply_out_transform(transform, expected)
            original_strides = _extract_strides(out)
            original_ptrs = _extract_data_ptrs(out)

            op_out(out=out)

            final_strides = _extract_strides(out)
            final_ptrs = _extract_data_ptrs(out)

            self.assertRtolEqual(expected, out)

            if compare_strides_and_data_ptrs:
                self.assertRtolEqual(original_strides, final_strides)
                self.assertRtolEqual(original_ptrs, final_ptrs)

        # Case 0: out= with the correct shape, dtype, and device
        #   but NaN values for floating point and complex tensors, and
        #   maximum values for integer tensors.
        #   Expected behavior: out= values have no effect on the computation.
        def _case_zero_transform(t):
            return t

        _compare_out(_case_zero_transform)

if __name__ == "__main__":
    run_tests()
