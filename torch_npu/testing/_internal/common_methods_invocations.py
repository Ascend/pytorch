from typing import List
from functools import wraps, partial
import unittest
import numpy as np

import torch
from torch.testing import make_tensor
from torch.testing._internal import common_methods_invocations
from torch.testing._internal.common_methods_invocations import sample_inputs_normal_common


def sample_inputs_normal_tensor_second(self, device, dtype, requires_grad, **kwargs):
    cases = [
        ([3, 4], 0.3, {}),
        ([3, 55], 0, {}),
        ([5, 6, 7, 8], [5, 6, 7, 8], {})
    ]
    return sample_inputs_normal_common(self, device, dtype, requires_grad, cases, **kwargs)


def sample_inputs_median_custom(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    empty_tensor_shape = [(2, 0), (0, 2)]
    for shape in empty_tensor_shape:
        yield common_methods_invocations.SampleInput(make_arg(shape))

    for torch_sample in \
            common_methods_invocations.sample_inputs_reduction(self, device, dtype, requires_grad, **kwargs):
        yield torch_sample
