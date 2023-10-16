from functools import wraps, partialmethod

import os
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
                elif k == "dtype":
                    test_name += ("_" + str(v).split('.')[1])
                for _func_key in func_args:
                    if k in _func_key:
                        if func_key is not None:
                            raise RuntimeError(f"Multiple matches for {k}")
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


def gen_ops_testcase(cls, func, name, keys, value, op_info):
    new_kwargs = {}
    test_name = f'{func.__name__}_{name}'

    for k, v in zip(keys, value):
        if k == "npu_format":
            test_name += ("_" + str(v))
        elif k == "dtype":
            test_name += ("_" + str(v).split('.')[1])
        new_kwargs[k] = v

    new_kwargs['op'] = op_info
    new_func = partialmethod(func, **new_kwargs)

    setattr(cls, test_name, new_func)
    for decorator in op_info.get_decorators(cls.__name__, func.__name__, 'cpu', value[0], {}):
        setattr(cls, test_name, decorator(new_func))


def gen_op_input(testcase, func, op_info):
    data = {
        'dtype': func.dtypes if hasattr(func, "dtypes") else op_info.dtypesIfNPU, 
        'npu_format': func.formats if hasattr(func, "formats") else op_info.formats
    }

    if 'test_variant_consistency_eager' in testcase:
        if torch.float32 in op_info.dtypesIfNPU:
            data['dtype'] = {torch.float32}
        else:
            data['dtype'] = {list(op_info.dtypesIfNPU)[-1]}

    return data


def instantiate_ops_tests(op_db):

    def wrapper(cls):
        testcases = [x for x in dir(cls) if x.startswith('test_')]
        for testcase in testcases: 
            if hasattr(cls, testcase):
                func = getattr(cls, testcase)
                for op_info in op_db:
                    data = gen_op_input(testcase, func, op_info)
                    keys = data.keys()
                    values = [data.get(key) for key in keys]

                    for value in itertools.product(*values):
                        gen_ops_testcase(cls, func, op_info.name, keys, value, op_info)

                delattr(cls, testcase)

        return cls
        
    return wrapper


class Dtypes(object):

    def __init__(self, *args):
        if (args is None or len(args) == 0):
            raise RuntimeError("No dtypes given")
        if not all(isinstance(arg, torch.dtype) for arg in args):
            raise RuntimeError("Unknown dtype in {0}".format(str(args)))
        self.args = args

    def __call__(self, fn):
        fn.dtypes = self.args
        return fn


class Formats(object):

    def __init__(self, *args):
        if args is None or len(args) == 0:
            raise RuntimeError("No formats given")
        self.args = args

    def __call__(self, fn):
        fn.formats = self.args
        return fn
