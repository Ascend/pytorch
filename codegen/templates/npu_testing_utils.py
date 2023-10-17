import unittest
from collections import defaultdict
import torch
from torch.testing._internal.common_methods_invocations import op_db, python_ref_db
from torch.testing._internal.opinfo.core import DecorateInfo

"""
strategy: Due to the restriction of NPU operators. 
patch the data classes to avoid unsupported cases.
"""

skip_list = {${skip_detail}}


def update_skip_list():
    for item in op_db:
        op_name = item.name
        if op_name in skip_list:
            if isinstance(item.skips, tuple):
                new_skips = tuple(skip_list[op_name]) + item.skips
            elif isinstance(item.skips, list):
                new_skips = skip_list[op_name] + item.skips
            else:
                new_skips = tuple(skip_list[op_name])
            item.skips = new_skips

    for item in python_ref_db:
        op_name = item.name
        if op_name in skip_list:
            if isinstance(item.skips, tuple):
                new_skips = tuple(skip_list[op_name]) + item.skips
            elif isinstance(item.skips, list):
                new_skips = skip_list[op_name] + item.skips
            else:
                new_skips = tuple(skip_list[op_name])
            item.skips = new_skips


def get_decorators(self, test_class, test_name, device, dtype, param_kwargs):
    self.decorators = (*self.decorators, *self.skips)
    result = []
    for decorator in self.decorators:
        if isinstance(decorator, DecorateInfo):
            if decorator.is_active(test_class, test_name, device, dtype, param_kwargs):
                result.extend(decorator.decorators)
        else:
            result.append(decorator)
    return result
