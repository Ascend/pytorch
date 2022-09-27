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

from functools import wraps

import os
import logging
import inspect
import itertools
import torch


def feed_data(func, new_name, *args, **kwargs):
    """
    This internal method decorator feeds the test data item to the test.
    """
    @wraps(func)
    def wrapper(self):
        return func(self, *args, **kwargs)
    wrapper.__name__ = new_name
    wrapper.__wrapped__ = func
    return wrapper


def instantiate_tests(arg=None, **kwargs):

    def wrapper(cls):
        def gen_testcase(cls, func, name, key_list, func_args, value):
            new_kwargs = dict(device="npu") if "device" in func_args else {}
            test_name = name
            for k, v in zip(key_list, value):
                func_key = None
                if k == "format":
                    test_name += ("_" + str(v))
                if k == "dtype":
                    test_name += ("_" + str(v).split('.')[1])
                for _func_key in func_args:
                    if k in _func_key:
                        assert func_key is None, f"Multiple matches for {k}"
                        func_key = _func_key
                new_kwargs[func_key] = v
            setattr(cls, test_name, feed_data(func, test_name, **new_kwargs))

        for name, func in list(cls.__dict__.items()):
            data = {}
            if hasattr(func, "dtypes"):
                data['dtype'] = func.dtypes
            if hasattr(func, "formats"):
                data['format'] = func.formats

            key_list = data.keys()
            if not key_list:
                continue

            func_args = inspect.getfullargspec(func).args
            value_list = [data.get(key) for key in key_list]
            for value in itertools.product(*value_list):
                gen_testcase(cls, func, name, key_list, func_args, value)

            delattr(cls, name)
        return cls

    return wrapper(arg)

def graph_mode(func):
    if os.getenv("GRAPH_MODE_TEST") == '1':
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.info("graph mode on")
        def wrapper(*args, **kw):
            logging.info("runing: {}".format(func.__name__))
            torch.npu.enable_graph_mode()
            func(*args, **kw)
            logging.info("graph mode off")
            torch.npu.disable_graph_mode()
        return wrapper
    
    def wrapper(*args, **kw):
        func(*args, **kw)
    return wrapper

class Dtypes(object):

    def __init__(self, *args):
        assert args is not None and len(args) != 0, "No dtypes given"
        assert all(isinstance(arg, torch.dtype) for arg in args), "Unknown dtype in {0}".format(str(args))
        self.args = args

    def __call__(self, fn):
        fn.dtypes = self.args
        return fn


class Formats(object):

    def __init__(self, *args):
        assert args is not None and len(args) != 0, "No formats given"
        self.args = args

    def __call__(self, fn):
        fn.formats = self.args
        return fn
