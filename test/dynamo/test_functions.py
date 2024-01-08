# Owner(s): ["module: dynamo"]
# flake8: noqa
import collections
import functools
import inspect
import itertools
import math
import operator
import sys
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple
from unittest.mock import patch

import numpy as np

import torch
import torch_npu
import torch._dynamo.test_case
import torch._dynamo.testing
from torch import sub
from torch._dynamo.testing import expectedFailureDynamic
from torch._dynamo.utils import ifdynstaticdefault, same

from torch._higher_order_ops.triton_kernel_wrap import (
    triton_kernel_wrapper_functional,
    triton_kernel_wrapper_mutation,
)
from torch._inductor import metrics
from torch.nn import functional as F
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    disable_translation_validation_if_dynamic_shapes,
    skipIfRocm,
)

# Defines all the kernels for tests
from torch.testing._internal.triton_utils import *  # noqa: F403

if HAS_CUDA:
    import triton
    from triton import language as tl

requires_cuda = functools.partial(
    unittest.skipIf, not torch.cuda.is_available(), "requires cuda"
)

d = torch.ones(10, 10)
e = torch.nn.Linear(10, 10)
flag = True


class CustomDictSubclass(collections.OrderedDict):
    pass


clip01 = functools.partial(torch.clip, min=0.0, max=1.0)


def constant3(a, b):
    return a - b + (1.0 + 2)


def func_with_default(a, b, some_default_arg=True):
    if some_default_arg:
        return a - b


def make_test(fn):
    nargs = len(inspect.signature(fn).parameters)

    def test_fn(self):
        return torch._dynamo.testing.standard_test(self, fn=fn, nargs=nargs)

    return test_fn


@torch.jit.script_if_tracing
def inline_script_if_tracing(x):
    return x + 1.2


@torch.jit.ignore
def inline_ignore(x):
    return x + 3.4


@torch.jit.unused
def inline_unused(x):
    return x + 5.6


class FunctionTests(torch._dynamo.test_case.TestCase):
    @make_test
    def test_inline_jit_annotations(x):
        x = inline_script_if_tracing(x)
        x = inline_ignore(x)
        x = inline_unused(x)
        return

    @make_test
    def test_add(a, b):
        return a + b

    @make_test
    def test_add_(a, b):
        a_copy = torch.tensor(a)
        return a_copy.add_(b, alpha=5.0)

    @make_test
    def test_addcdiv(a, b, c):
        # dynamo decomposes this to avoid a graph break when
        # the value kwarg is populated
        return torch.addcdiv(a, b, c, value=5.0)

    @make_test
    def test_addcdiv_(a, b, c):
        a_copy = torch.tensor(a)
        return a_copy.addcdiv_(b, c, value=5.0)

    @make_test
    def test_is_not_null(a, b):
        if a is not None and b is not None:
            return a + b

    @make_test
    def test_functools_partial(a, b):
        return clip01(a + b)

    @make_test
    def test_itertools_product(a, b):
        v = a
        for x, i in itertools.product([a, b], [1, 2]):
            v = v + x * i
        return v

    @make_test
    def test_itertools_chain(a, b):
        v = a
        for x in itertools.chain([a, b], [1, 2]):
            v = v + x
        return v

    @make_test
    def test_itertools_combinations(a, b):
        combs = []
        for size in itertools.combinations((1, 2, 3, 4), 2):
            combs.append(torch.ones(size))
        return combs

    @make_test
    def test_constant1(a, b, c):
        return a - b * c + 1.0

    @make_test
    def test_constant2(a, b, c):
        return a - b * c + 1

    @make_test
    def test_constant3(a):
        b = 1
        c = 2
        f = 3
        return b + c - f + a

    @make_test
    def test_constant4(a, b):
        c = 2
        f = 3
        if c > f:
            return a - b
        return b - a

    @make_test
    def test_finfo(a, b):
        if torch.iinfo(torch.int32).bits == 32:
            return torch.finfo(a.dtype).min * b

    @make_test
    def test_globalfn(a, b):
        return sub(a, b)

    @make_test
    def test_viatorch(a, b):
        return torch.sub(a, b)

    @make_test
    def test_viamethod(a, b):
        return a.sub(b)

    @make_test
    def test_indirect1(a, b):
        t = a.sub
        return t(b)

    @make_test
    def test_indirect2(a, b):
        t = a.sub
        args = (b,)
        return t(*args)

    @make_test
    def test_indirect3(a, b):
        t = a.sub
        args = (b,)
        kwargs = {}
        return t(*args, **kwargs)

    @make_test
    def test_methodcall1(a, b, c):
        return constant3(a, b) * c

    @make_test
    def test_methodcall2(a, b):
        return constant3(a=b, b=a) + 1

    @make_test
    def test_methodcall3(a, b):
        return constant3(a, b=1.0) + b

    @make_test
    def test_device_constant(a):
        return a + torch.ones(1, device=torch.device("cpu"))

    @make_test
    def test_tuple1(a, b):
        args = (a, b)
        return sub(*args)

    @make_test
    def test_tuple2(a, b):
        args = [a, b]
        return sub(*args)

    @make_test
    def test_is_in_onnx_export(x, y):
        if torch.onnx.is_in_onnx_export():
            return x - 1
        else:
            return y + 1

    @make_test
    def test_is_fx_tracing(x, y):
        if torch.fx._symbolic_trace.is_fx_tracing():
            return x - 1
        else:
            return y + 1

    @make_test
    def test_listarg1(a, b):
        return torch.cat([a, b])

    @make_test
    def test_listarg2(a, b):
        return torch.cat((a, b), dim=0)

    @make_test
    def test_listarg3(a, b):
        kwargs = {"tensors": (a, b), "dim": 0}
        return torch.cat(**kwargs)

    @make_test
    def test_listarg4(a, b):
        return torch.cat(tensors=[a, b], dim=0)

    @make_test
    def test_listarg5(a, b):
        args = [(a, b)]
        kwargs = {"dim": 0}
        return torch.cat(*args, **kwargs)

    @make_test
    def test_deque(a, b):
        f = collections.deque([a, b])
        f.append(a + 1)
        f.extend([a, b])
        f.insert(0, "foo")
        tmp = f.pop()

        another_deque = collections.deque([tmp])
        f.extendleft(another_deque)
        another_deque.clear()
        f.extend(another_deque)

        f[2] = "setitem"
        f = f.copy()
        f.append(f.popleft())

        empty = collections.deque()
        f.extend(empty)

        # dynamo same() util doesn't support deque so just return a list
        return list(f)

    @make_test
    def test_slice1(a):
        return a[5]

    @make_test
    def test_slice2(a):
        return a[:5]

    @make_test
    def test_slice3(a):
        return a[5:]

    @make_test
    def test_slice4(a):
        return a[2:5]

    @make_test
    def test_slice5(a):
        return a[::2]

    @make_test
    def test_slice6(a):
        return torch.unsqueeze(a, 0)[:, 2:]

    @make_test
    def test_range1(a):
        return torch.tensor(range(a.size(0)))

    @make_test
    def test_range2(x, y):
        r = x + y
        for i in range(x.size(0) + 2):
            r = r / y
        return r

    @make_test
    def test_unpack1(a):
        a, b = a[:5], a[5:]
        return a - b

    @make_test
    def test_unpack2(a):
        packed = [a[:5], a[5:]]
        a, b = packed
        return a - b

    @make_test
    def test_unpack3(a):
        packed = (a[:5], a[5:])
        a, b = packed
        return a - b

    @make_test
    def test_fn_with_self_set(a, b):
        # avg_pool2d is an odd one with __self__ set
        return F.avg_pool2d(
            torch.unsqueeze(a, 0) * torch.unsqueeze(b, 1), kernel_size=2, padding=1
        )

    @make_test
    def test_return_tuple1(a, b):
        res = (a - b, b - a, a, b)
        return res

    @make_test
    def test_globalvar(a, b):
        return a - b + d

    @make_test
    def test_globalmodule(x):
        return e(x)

    @make_test
    def test_inline_with_default(a, b, c):
        return func_with_default(a, b) * c

    @make_test
    def test_inner_function(x):
        def fn(x):
            return torch.add(x, x)

        return fn(x)

    @make_test
    def test_transpose_for_scores(x):
        new_x_shape = x.size()[:-1] + (2, 5)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)

    @make_test
    def test_return_tuple2(x):
        return (torch.add(x, x), x)

    @make_test
    def test_load_global_bool(x):
        if flag:
            return torch.add(x, x)
        else:
            return x

    @make_test
    def test_len_tensor(x):
        z = len(x)
        return torch.add(x, z)

    @make_test
    def test_len_constant_list(x):
        z = len([1, 2, 3])
        return torch.add(x, z)

    @make_test
    def test_len_constant_dict(x):
        z = len({"foo": "bar"})
        return torch.add(x, z)

    @make_test
    def test_dict_copy(x):
        z = dict({"foo": x + 1})
        return z

    @make_test
    def test_callable_lambda(x):
        if callable(lambda x: True):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_callable_torch(x):
        if callable(torch.abs):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_callable_builtin(x):
        if callable(sum):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_len_constant_misc_iterables(x):
        a = len((1, 2, 3))
        b = len("test str")
        c = a + b
        return torch.add(x, c)

    @make_test
    def test_dict_kwargs(x):
        z = dict(text_embed=x + 1, other=x + 2)
        return z

    @make_test
    def test_ordered_dict_kwargs(x):
        z = collections.OrderedDict(sample=torch.ones(10))
        return z

    @make_test
    def test_custom_dict_kwargs(x):
        z = CustomDictSubclass(sample=torch.ones(10))
        return z

    @make_test
    def test_float(x):
        y = float(1.2)
        y += float("1.2")
        return torch.add(x, y)

    @make_test
    def test_is_floating_point(x):
        y = x + 1
        return torch.is_floating_point(y), torch.is_floating_point(input=y)

    @make_test
    def test_dtype(x):
        if x.dtype == torch.float32:
            return x + 1

    @make_test
    def test_get_default_dtype(x):
        if x.dtype == torch.get_default_dtype():
            return x + 1
        else:
            return x - 1

    @make_test
    def test_get_autocast_gpu_dtype(x):
        dtype = torch.get_autocast_gpu_dtype()
        return x.type(dtype)

    @make_test
    def test_promote_types(x):
        if x.dtype == torch.promote_types(torch.int32, torch.float32):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_get_calculate_correct_fan(x):
        fan_in = torch.nn.init._calculate_correct_fan(x, "fan_in")
        return x + fan_in

    @make_test
    def test_is_complex(x):
        if torch.is_complex(x):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_get_privateuse1_name(x):
        if torch._C._get_privateuse1_backend_name() == "privateuseone":
            return x + 1
        else:
            return x - 1

    @make_test
    def test_device(x):
        if not x.is_npu:
            return x + 1

    @make_test
    def test_tensor_type(a, b):
        m = a.to(torch.float16)
        return b.type(m.type())

    @unittest.skipIf(not torch.npu.is_available(), "requires npu")
    @make_test
    def test_tensor_type2(a, b):
        m = a.to("npu")
        return m + b.type(m.type())

    @make_test
    def test_tensor_type3(a, b):
        m = a.type(torch.HalfTensor)
        return b.type(m.type())

    @make_test
    def test_tensor_type4(a, b):
        m = a.type("torch.HalfTensor")
        return b.type(m.type())

    @unittest.skipIf(not torch.npu.is_available(), "requires npu")
    @make_test
    def test_tensor_type5(a, b):
        m = a.type(torch.npu.HalfTensor)
        return b.type(m.type())

    @make_test
    def test_ndim(x):
        if x.ndim == 2 and x.ndimension() == 2 and x.dim() == 2:
            return x + 1

    @make_test
    def test_T(x):
        return torch.ones_like(x.T)

    @make_test
    def test_mT(x):
        return torch.ones_like(x.mT)

    @make_test
    def test_is_sparse(x):
        if not x.is_sparse:
            return x + 1

    @make_test
    def test_shape1(x):
        if x.shape[0] == 10:
            return x + 1

    @make_test
    def test_shape2(x):
        if x.size(1) == 10:
            return x + 1

    @make_test
    def test_del(a, b):
        c = a + 1
        f = c + 2
        del c, a
        return b + f

    @make_test
    def test_chunks1(x):
        chunk_size = 5
        assert x.shape[0] % chunk_size == 0
        assert x.shape[0] // chunk_size == 2
        return x[:chunk_size] - x[chunk_size:]

    @make_test
    def test_import1(x, y):
        return sub(torch.add(x, y), y)

    @make_test
    def test_return_dict(x, y):
        z = [x + y, y, False]
        return {"x": x, "z": z, "a": x, "b": z, "c": x}

    @make_test
    def test_return_dict2(x, y):
        tmp = {"x": x}
        tmp["z"] = [x + y, y]
        tmp["y"] = y
        tmp["z"].append(False)
        return tmp

    @make_test
    def test_funcdef_closure(x, y):
        x = x + y + 1.0

        def inner(z):
            nonlocal x, y
            y = x + z + 20.0
            x = y + z + 10.0

        inner(2.0)
        inner(3.0)

        return x, y

    @make_test
    def test_module_constant(x, y):
        r = x + y
        for i in range(torch._dynamo.testing.three):
            r = r / y
        return r

    @make_test
    def test_inline_softmax(x, y):
        # This is common in sme huggingface models
        return torch.nn.Softmax(dim=-1)(x + y * 2)

    @make_test
    def test_dtype_compare(a, b):
        if a.dtype == torch.float16:
            return a + 10
        if a.dtype == torch.float32:
            return a - b * 32

    @make_test
    def test_build_list_unpack(a, b):
        it1 = (x + 1 for x in (a, b))
        it2 = (x - 1 for x in (a, b))
        return torch.cat([*it1, *it2], dim=-1)

    @make_test
    def test_tensor_len(a, b):
        return a + b + len(a) + b.__len__()

    @make_test
    def test_pop(a, b):
        ll = [a, b]
        ll.append(a + 1)
        ll.extend(
            [
                b + 2,
                a + b,
            ]
        )
        ll.pop(-1)
        ll.pop(0)
        ll.pop()
        v1, v2 = ll
        return v1 - v2

    @make_test
    def test_list_convert(a, b):
        ll = [a + 2, b]
        ll = tuple(ll)
        tmp = b + 3
        ll = list(ll)
        v1, v2 = ll
        return v1 - v2 + tmp

    @make_test
    def test_list_add(a, b):
        l1 = (a, b)
        l2 = ()  # being a LOAD_CONST in the bytecode
        l3 = l1 + l2
        return l3[0] + l3[1]

    @make_test
    def test_list_index_with_constant_tensor(a, b):
        l1 = [a, b, a + 1, b + 1]
        return l1[torch.as_tensor(2)]

    @make_test
    def test_startswith(a, b):
        x = a + b
        if "foobar".startswith("foo") and "test" in constant3.__module__:
            x = x + 1
        return x

    @make_test
    def test_dict_ops(a, b):
        tmp = {"a": a + 1, "b": b + 2}
        assert tmp.get("zzz") is None
        v = tmp.pop("b") + tmp.get("a") + tmp.get("missing", 3) + tmp.pop("missing", 4)
        tmp.update({"d": 3})
        tmp["c"] = v + tmp.get('d')
        if "c" in tmp and "missing" not in tmp:
            return tmp["c"] - tmp["a"] + len(tmp)

    def test_dict_param_keys(self):
        a_param = torch.nn.Parameter(torch.ones([4, 4]))

        def fn(a):
            tmp = {"a": a, a_param: 3}
            return tmp["a"] + tmp[a_param]

        test = make_test(fn)
        test(self)

    def _test_default_dict_helper(self, factory):
        dd = collections.defaultdict(factory)
        param = torch.nn.Parameter(torch.ones([2, 2]))

        def fn(x):
            dd["a"] = x + 1
            dd[param] = 123
            dd["c"] = x * 2
            return dd["b"], dd

        x = torch.randn(10, 10)
        ref = fn(x)
        opt_fn = torch._dynamo.optimize_assert("eager")(fn)
        res = opt_fn(x)

        self.assertTrue(same(ref[0], res[0]))
        self.assertTrue(same(ref[1]["a"], res[1]["a"]))
        self.assertTrue(same(ref[1]["c"], res[1]["c"]))
        self.assertTrue(same(ref[1][param], res[1][param]))

    def test_default_dict(self):
        self._test_default_dict_helper(dict)

    def test_default_dict_lambda(self):
        self._test_default_dict_helper(lambda: dict())

    def test_default_dict_closure(self):
        def factory():
            return dict()

        self._test_default_dict_helper(factory)

    def test_default_dict_constr(self):
        param = torch.nn.Parameter(torch.ones([2, 2]))

        def fn(x):
            dd = collections.defaultdict(lambda: dict())
            dd["a"] = x + 1
            dd[param] = 123
            dd["c"] = x * 2
            dd.update({"b": x * 3})
            dd.update([["d", x - 2], ("e", x + 2)])
            dd.update(zip("ab", [x + 3, x + 4]))
            return dd["b"], dd

        x = torch.randn(10, 10)
        ref = fn(x)
        opt_fn = torch._dynamo.optimize_assert("eager")(fn)
        res = opt_fn(x)

        self.assertTrue(same(ref[0], res[0]))
        self.assertTrue(same(ref[1]["a"], res[1]["a"]))
        self.assertTrue(same(ref[1]["b"], res[1]["b"]))
        self.assertTrue(same(ref[1]["c"], res[1]["c"]))
        self.assertTrue(same(ref[1]["d"], res[1]["d"]))
        self.assertTrue(same(ref[1]["e"], res[1]["e"]))
        self.assertTrue(same(ref[1][param], res[1][param]))

    @make_test
    def test_call_dict1(x):
        d1 = dict()
        d1["x"] = x + 1
        d2 = collections.OrderedDict()
        d2["x"] = x + 2
        return d1["x"] + d2["x"] + 1

    @make_test
    def test_call_dict2(x):
        d1 = dict()
        d1["x"] = x
        d2 = collections.OrderedDict(d1)
        if isinstance(d2, collections.OrderedDict):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_call_dict3(x):
        my_list = [("a", x), ("b", x + 1), ("c", x + 2)]
        d1 = dict(my_list)
        d1["a"] = x + 10
        d2 = collections.OrderedDict(my_list)
        d2["c"] = x + 20
        return d1["a"] + d2["c"] + 1

    @make_test
    def test_call_dict4(x):
        my_list = (("a", x), ("b", x + 1), ("c", x + 2))
        d1 = dict(my_list)
        d1["a"] = x + 10
        d2 = collections.OrderedDict(my_list)
        d2["c"] = x + 20
        return d1["a"] + d2["c"] + 1

    @make_test
    def test_call_dict5(x):
        my_list = iter([("a", x), ("b", x + 1), ("c", x + 2)])
        d1 = dict(my_list)
        d1["a"] = x + 10
        d2 = collections.OrderedDict(my_list)
        d2["c"] = x + 20
        return d1["a"] + d2["c"] + 1

    @make_test
    def test_dict_fromkeys(x, y):
        lst = ["a", "b"]
        dd = dict.fromkeys(lst)
        d1 = dict.fromkeys(dd, x + 1)
        d2 = collections.defaultdict.fromkeys(iter(d1), x - 2)
        d3 = collections.OrderedDict.fromkeys(tuple(lst), value=y)
        return d1.get('a') * d2.get('b') + d2.get('a') + d1.get('b') + d3.get('a')+ d3.get('b') + 1

    @make_test
    def test_dict_copy(x):
        my_list = [("a", x), ("b", x + 1), ("c", x + 2)]
        d1 = dict(my_list)
        d1["a"] = x + 10
        d2 = d1.copy()
        d2["a"] = x - 5
        d2["b"] = x + 3
        d3 = collections.OrderedDict(my_list)
        d3["c"] = x + 20
        d4 = d3.copy()
        d4["c"] = x - 10
        return d1.get('a') * d2.get('b') + d2.get('a') + d3.get('c') + d4.get('c') + 1

    @make_test
    def test_dict_update(x, y, z):
        dd = {"a": x, "b": y}
        dd.update({"a": y - 1})
        dd.update([("b", z + 1), ["c", z]])
        dd.update(zip("ab", [z + 3, y + 2]))

        od = collections.OrderedDict(a=x * 3, b=y + 2)
        od.update({"a": y + 5})
        od.update([["b", z + 6], ("c", z - 7)])
        od.update(zip("ab", [z - 3, x + 2]))
        return dd.get("a") * od.get("a") + od.get("c") + dd.get("b") + od.get("b") * dd.get("c")

    @make_test
    def test_min_max(a, b):
        c = a + b
        a = a.sum()
        b = b.sum()
        a = min(max(a, 0), 1)
        b = max(0, min(1, b))
        return max(a, b) - min(a, b) + c

    @make_test
    def test_symbool_to_int(x):
        # this is roughly the pattern found in einops.unpack()
        if sum(s == -1 for s in x.size()) == 0:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_map_sum(a, b, c, f):
        return sum(map(lambda x: x + 1, [a, b, c, f]))

    @make_test
    def test_reduce(a, b, c, f):
        return functools.reduce(operator.add, [a, b, c, f])

    @make_test
    def test_tuple_contains(a, b):
        v1 = "a"
        v2 = "b"
        v3 = "c"
        vals1 = (v1, v2, v3)
        vals2 = ("d", "e", "f")
        if "a" in vals1 and "b" not in vals2:
            return a + b
        return a - b

    @make_test
    def test_set_contains(a, b):
        vals = set(["a", "b", "c"])
        if "a" in vals:
            x = a + b
        else:
            x = a - b
        if "d" in vals:
            y = a + b
        else:
            y = a - b
        return x, y

    @make_test
    def test_tuple_iadd(a, b):
        output = (a, b)
        output += (a + b, a - b)
        return output

    @make_test
    def test_unpack_ex1(x):
        output = (x, x + 1, x + 2, x + 3)
        a, b, *cd = output
        return a - b / cd[0]

    @make_test
    def test_unpack_ex2(x):
        output = (x, x + 1, x + 2, x + 3)
        *ab, c, dd = output
        return c - dd / ab[0]

    @make_test
    def test_unpack_ex3(x):
        output = (x, x + 1, x + 2, x + 3)
        a, *bc, dd = output
        return a - dd / bc[0]

    @make_test
    def test_const_tuple_add1(x):
        output = (x, x + 1, x + 2, x + 3)
        output = () + output + ()
        return output[2] + output[3]

    @make_test
    def test_const_tuple_add2(x):
        output = (x, x + 1, x + 2, x + 3)
        output = (None,) + output + (None,)
        return output[2] + output[3]

    @make_test
    def test_list_truth(a, b):
        tmp = [1, 2, 3]
        if tmp:
            return a + b
        else:
            return a - b

    @make_test
    def test_list_reversed(a, b):
        tmp = [a + 1, a + 2, a + 3]
        return a + b + next(iter(reversed(tmp)))

    @make_test
    def test_list_sorted1(x):
        tmp = [1, 10, 3, 0]
        return x + 1, sorted(tmp), sorted(tmp, reverse=True)

    @make_test
    def test_list_sorted2(x):
        y = [
            ("john", "A", 8),
            ("jane", "B", 5),
            ("dave", "B", 10),
        ]
        res =  (
            x + 1,
            sorted(y),
            sorted(y, key=lambda student: student[2]),
            sorted(y, key=lambda student: student[2], reverse=True),
        )
        return res

    @make_test
    def test_tuple_sorted(x):
        tmp = (1, 10, 3, 0)
        return x + 1, sorted(tmp), sorted(tmp, reverse=True)

    @make_test
    def test_dict_sorted(x):
        tmp = {1: "D", 10: "B", 3: "E", 0: "F"}
        return x + 1, sorted(tmp), sorted(tmp, reverse=True)

    @make_test
    def test_list_clear(a, b):
        tmp = [a + 1, a + 2]
        tmp.clear()
        tmp.append(a + b)
        return tmp

    @make_test
    def test_not_list(a):
        return not [a + 1]

    @make_test
    def test_islice_chain(a, b):
        tmp1 = [a + 1, a + 2]
        tmp2 = [a + 3, a + 4]
        a, b = list(itertools.islice(itertools.chain(tmp1, tmp2), 1, 3))
        c = next(itertools.islice(tmp1, 1, None))
        return a - b / c

    @make_test
    def test_namedtuple(a, b):
        mytuple = collections.namedtuple("mytuple", ["x", "y", "xy"])
        tmp = mytuple(a, b, a + b)
        return mytuple(tmp.x, tmp[1], tmp.xy + b)

    @make_test
    def test_namedtuple_defaults(a, b):
        mytuple = collections.namedtuple(
            "mytuple", ["x", "y", "xy"], defaults=(None, 1, None)
        )
        tmp = mytuple(a, xy=b)
        return mytuple(tmp.x, tmp[1], tmp.xy + b)

    class MyNamedTuple(NamedTuple):
        first: torch.Tensor
        second: torch.Tensor

        def add(self) -> torch.Tensor:
            return self.first + self.second

        @staticmethod
        def static_method() -> int:
            return 1

        @classmethod
        def class_method(cls) -> str:
            return cls.__name__

    @make_test
    def test_namedtuple_user_methods(a, b):
        mytuple = FunctionTests.MyNamedTuple(a, b)
        return mytuple.add(), mytuple.static_method(), mytuple.class_method()

    @make_test
    def test_is_quantized(a, b):
        if not a.is_quantized:
            return a + b

    @make_test
    def test_fstrings1(a, b):
        x = 1.229
        tmp = f"{x:.2f} bar"
        if tmp.startswith("1.23"):
            return a + b

    # See pytorch/pytorch/issues/103602
    @expectedFailureDynamic
    @make_test
    def test_fstrings2(x):
        tmp = f"{x.shape[0]} bar"
        if tmp.startswith("10"):
            return x + 1

    @make_test
    def test_fstrings3(x):
        tmp = f"{x.__class__.__name__} foo"
        if tmp.startswith("Tensor"):
            return x + 1

    @make_test
    def test_tensor_new_with_size(x):
        y = torch.rand(5, 8)
        z = x.new(y.size())
        assert z.size() == y.size()

    @make_test
    def test_tensor_new_with_shape(x):
        y = torch.rand(5, 8)
        z = x.new(y.shape)
        assert z.size() == y.size()

    @make_test
    def test_jit_annotate(x):
        y = torch.jit.annotate(Any, x + 1)
        return y + 2

    @make_test
    def test_is_contiguous_memory_format(tensor):
        if torch.jit.is_scripting():
            return None
        elif tensor.is_contiguous(memory_format=torch.contiguous_format):
            return tensor + 1

    def test_is_contiguous_frame_counts(self):
        data = [
            torch.rand(10),
            torch.rand(2, 3, 32, 32),
            torch.rand(2, 3, 32, 32).contiguous(memory_format=torch.channels_last),
            torch.rand(10)[::2],
            torch.rand(12),
            torch.rand(2, 3, 24, 24).contiguous(memory_format=torch.channels_last),
            torch.rand(50)[::2],
            torch.rand(2, 3, 32, 32)[:, :, 2:-2, 3:-3],
        ]
        # dynamo should recompile for all inputs in static shapes mode
        expected_frame_counts_static = [1, 2, 3, 4, 5, 6, 7, 8]
        # dynamo should recompile for items 0, 1, 2, 6 in dynamic shapes mode
        expected_frame_counts_dynamic = [1, 2, 3, 4, 4, 4, 4, 5]
        expected_frame_counts = ifdynstaticdefault(
            expected_frame_counts_static, expected_frame_counts_dynamic
        )
        dynamic = ifdynstaticdefault(False, True)

        def func(x):
            if x.is_contiguous():
                return x + 1
            elif x.is_contiguous(memory_format=torch.channels_last):
                return x + 2
            else:
                return x + 3

        cnt = torch._dynamo.testing.CompileCounter()
        cfunc = torch._dynamo.optimize_assert(cnt, dynamic=dynamic)(func)

        assert cnt.frame_count == 0
        for i, x in enumerate(data):
            expected = func(x)
            output = cfunc(x)
            self.assertTrue(same(output, expected))
            assert cnt.frame_count == expected_frame_counts[i]

    @make_test
    def test_list_slice_assignment(x):
        m = [1, 2, 3, 4]
        m[1:] = [6] * (len(m) - 1)
        return x + 1

    @make_test
    def test_distributed_is_available(x):
        if torch.distributed.is_available():
            return x + 1
        else:
            return x - 1

    @unittest.skipIf(
        not torch.distributed.is_available(), "requires distributed package"
    )
    @make_test
    def test_distributed_is_initialized(x):
        if torch.distributed.is_initialized():
            return x + 1
        else:
            return x - 1

    @disable_translation_validation_if_dynamic_shapes
    @make_test
    def test_torch_distributions_functions(x):
        normal = torch.distributions.Normal(x, torch.tensor(1))
        independent = torch.distributions.Independent(normal, 1)
        return independent.log_prob(x)

    @make_test
    def test_context_wrapping_nested_functions_no_closure(x):
        @torch.no_grad()
        def augment(x: torch.Tensor) -> torch.Tensor:
            return (x + 1) * 2

        return augment(x)

    # # This is to test the new syntax for pattern matching
    # # ("match ... case ...") added on python 3.10.
    # # Uncomment these test cases if you run on 3.10+
    # @make_test
    # def test_match_sequence(a):
    #     point = (5, 8)
    #     match point:
    #         case (0, 0):
    #             return a
    #         case (0, y):
    #             return a - y
    #         case (x, 0):
    #             return a + x
    #         case (x, y):
    #             return a + x - y

    # @make_test
    # def test_match_mapping_and_match_keys(x):
    #     param = {"a": 0.5}
    #     match param:
    #         case {"a": param}:
    #             return x * param
    #         case {"b": param}:
    #             return x / param

    def test_math_radians(self):
        def func(x, a):
            return x + math.radians(a)

        cnt = torch._dynamo.testing.CompileCounter()
        cfunc = torch._dynamo.optimize_assert(cnt)(func)

        assert cnt.frame_count == 0
        x = torch.rand(10)
        expected = func(x, 12)
        output = cfunc(x, 12)
        self.assertTrue(same(output, expected))
        assert cnt.frame_count == 1

    @make_test
    def test_numpy_meshgrid(x, y):
        r1, r2 = np.meshgrid(x.numpy(), y.numpy())
        return torch.from_numpy(r1), torch.from_numpy(r2)

    @make_test
    def test_torch_from_numpy(x):
        a = x.numpy()
        b = torch.from_numpy(a)
        if b.size(0) == 1:
            return torch.tensor(True)
        else:
            return torch.tensor(False)

    @make_test
    def test_numpy_size(x):
        a = x.numpy()
        return a.size

    @make_test
    def test_numpy_attributes(x):
        a = x.numpy()
        res =  (
            a.itemsize,
            a.strides,
            a.shape,
            a.ndim,
            a.size,
            torch.from_numpy(a.T),
            torch.from_numpy(a.real),
            torch.from_numpy(a.imag),
        )
        return res

    @make_test
    def test_mean_sum_np(x: torch.Tensor):
        x_mean = np.mean(x.numpy(), 1)
        x_sum = np.sum(x_mean)
        x_sum_array = np.asarray(x_sum)
        return torch.from_numpy(x_sum_array)

    @make_test
    def test_return_numpy_ndarray(x):
        a = x.numpy()
        return a.T

    @make_test
    def test_return_multiple_numpy_ndarray(x):
        a = x.numpy()
        return a.T, a.imag, a.real

    @make_test
    def test_ndarray_method(x):
        a = x.numpy()
        return a.copy()

    @make_test
    def test_ndarray_transpose(x):
        a = x.numpy()
        return a.transpose(0, 1)

    @make_test
    def test_ndarray_reshape(x):
        a = x.numpy()
        return a.reshape([1, a.size])

    @make_test
    def test_ndarray_methods_returning_scalar(x):
        a = x.numpy()
        return a.max(axis=0), a.all(axis=0)

    @make_test
    def test_ndarray_builtin_functions(x):
        a = x.numpy()
        return a + a, a - a

    @make_test
    def test_numpy_dtype_argument_to_function(x):
        return np.ones_like(x, dtype=np.float64)

    @make_test
    def test_numpy_linalg(x):
        return np.linalg.norm(x.numpy(), axis=0)

    @make_test
    def test_numpy_fft(x):
        return np.fft.fftshift(x.numpy())

    @make_test
    def test_numpy_random():
        x = np.random.randn(2, 2)
        return x - x

    @make_test
    def test_partials_torch_op_kwarg(x):
        par_mul = functools.partial(torch.mul, other=torch.ones(10, 10))
        return par_mul(x)

    @make_test
    def test_partials_torch_op_arg(x):
        par_mul = functools.partial(torch.mul, torch.ones(10, 10))
        return par_mul(x)

    @make_test
    def test_partials_udf_arg(x):
        par_mul = functools.partial(udf_mul, torch.ones(10, 10))
        return par_mul(x)

    @make_test
    def test_partials_udf_kwarg(x):
        par_mul = functools.partial(udf_mul, y=torch.ones(10, 10))
        return par_mul(x)

    @make_test
    def test_partials_udf_kwarg_module(x, y):
        par_mod = functools.partial(udf_module, mod=SmallNN())
        return par_mod(x=x, y=y)

    @make_test
    def test_partials_udf_kwarg_method(x, y):
        par_mod = functools.partial(udf_module, mod=SmallNN().forward)
        return par_mod(x=x, y=y)

    @make_test
    def test_partials_lambda(x):
        multiply = lambda x, y: x * y
        triple = functools.partial(multiply, y=3)
        return triple(x)

    def test_pow_int(self):
        def fn(a, b):
            return torch.pow(a, b)

        x = torch.ones(2, 2)
        opt_fn = torch.compile(fullgraph=True, backend="eager", dynamic=True)(fn)
        self.assertEqual(opt_fn(x, 2), fn(x, 2))

    def test_tensor_size_indexed_by_symint(self):
        def fn(x, y):
            index = x.shape[-1]
            return x + y.shape[index]

        x = torch.rand(10, 2)
        y = torch.rand(10, 8, 6)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_partials_as_input_partials_lambda(self):
        def fn(f0, f1, x):
            return f0(x) * f1(x)

        multiply = lambda x, y: x * y
        lambda0 = functools.partial(multiply, y=3)
        lambda1 = functools.partial(multiply, y=2)

        cnts = torch._dynamo.testing.CompileCounter()
        torch._dynamo.optimize(cnts, nopython=True)(fn)(
            lambda0, lambda1, torch.randn(2, 2)
        )
        self.assertEqual(cnts.frame_count, 1)

    def test_partials_as_input_partials_mod(self):
        def fn(f0, f1, x):
            return f0(x) * f1(x)

        lambda0 = functools.partial(SmallNN(), y=torch.randn(2, 2))
        lambda1 = functools.partial(SmallNN(), y=torch.randn(2, 2))

        cnts = torch._dynamo.testing.CompileCounter()
        x = torch.randn(2, 2)
        dynamo_result = torch._dynamo.optimize(cnts, nopython=True)(fn)(
            lambda0, lambda1, x
        )
        self.assertEqual(cnts.frame_count, 1)

        eager_result = fn(lambda0, lambda1, x)
        self.assertEqual(eager_result, dynamo_result)

    def test_partials_as_input_UDF(self):
        def fn(f0, f1, x):
            return f0(x) * f1(x)

        lambda0 = functools.partial(udf_mul, y=torch.randn(2, 2))
        lambda1 = functools.partial(udf_mul, y=torch.randn(2, 2))

        cnts = torch._dynamo.testing.CompileCounter()
        x = torch.randn(2, 2)
        dynamo_result = torch._dynamo.optimize(cnts, nopython=True)(fn)(
            lambda0, lambda1, x
        )
        self.assertEqual(cnts.frame_count, 1)

        eager_result = fn(lambda0, lambda1, x)
        self.assertEqual(eager_result, dynamo_result)

    def test_partials_recompilation(self):
        def fn(f0, f1, x):
            return f0(x) * f1(x)

        lambda0 = functools.partial(udf_mul, y=torch.randn(2, 2))
        lambda1 = functools.partial(udf_mul, y=torch.randn(2, 2))

        cnts = torch._dynamo.testing.CompileCounter()
        x = torch.randn(2, 2)
        fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        dynamo_result = fn(lambda0, lambda1, x)
        self.assertEqual(cnts.frame_count, 1)

        fn(lambda1, lambda0, x)
        self.assertEqual(
            cnts.frame_count, 1
        )  # No recompile! Tensor and udf_mul guarded

        lambda2 = functools.partial(udf_mul, y=torch.randn(3, 3))
        x = torch.randn(3, 3)
        fn(lambda2, lambda2, x)
        self.assertEqual(cnts.frame_count, 2)  # Recompile! Tensor size changed

        multiply = lambda x, y: x * y
        lambda3 = functools.partial(multiply, y=torch.randn(3, 3))
        x = torch.randn(3, 3)
        fn(lambda3, lambda3, x)

        self.assertEqual(cnts.frame_count, 3)  # Recompile! func id changed

        def fn2(f0, f1, args):
            return f0(*args) * f1(*args)

        cnts = torch._dynamo.testing.CompileCounter()

        x = torch.randn(2, 2)
        fn2 = torch._dynamo.optimize(cnts, nopython=True)(fn2)
        dynamo_result = fn2(lambda0, lambda1, [x])
        self.assertEqual(cnts.frame_count, 1)  # start over

        lambda4 = functools.partial(multiply, y=3, x=torch.randn(3, 3))
        fn2(lambda4, lambda4, [])

        self.assertEqual(cnts.frame_count, 2)  # Recompile! Different kwarg keys

        lambda5 = functools.partial(multiply, 1)
        x = torch.randn(3, 3)
        fn2(lambda5, lambda5, [x])

        self.assertEqual(cnts.frame_count, 3)  # Recompile! Different arg keys

        lambda6 = lambda x: x + x
        fn2(lambda6, lambda6, [x])
        self.assertEqual(
            cnts.frame_count, 4
        )  # Recompile! input is no longer a functools partial

    def test_manual_seed(self):
        @torch.compile
        def foo():
            torch.manual_seed(3)
            return torch.randint(0, 5, (5,))

        self.assertEqual(foo(), foo())
        self.assertEqual(foo(), foo())

    def test_partial_across_graph_break_uninvoked(self):
        from functools import partial

        def bar(x, **kwargs):
            return x + x

        @torch.compile(backend="eager", dynamic=True)
        def foo(x, i):
            def inner():
                print("this is a graph_break")
                return op(x)

            op = partial(bar, dim=10)
            x = inner()
            op = partial(bar, other=10)
            return inner() + x

        foo(torch.rand(1), 10)

    def test_no_recompile_inner_function(self):
        def forward(inp):
            def g(y):
                return inp + y

            print("graph break")
            return g(torch.rand([1]))

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(forward)

        ipt = torch.rand([2])
        _ = opt_fn(ipt)
        _ = opt_fn(ipt)
        _ = opt_fn(ipt)
        # Should not have recompiled
        self.assertEqual(cnts.frame_count, 1)

    def test_no_recompile_inner_lambda(self):
        def forward(inp):
            g = lambda y: inp + y
            print("graph break")
            return g(torch.rand([1]))

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(forward)

        ipt = torch.rand([2])
        _ = opt_fn(ipt)
        _ = opt_fn(ipt)
        _ = opt_fn(ipt)
        # Should not have recompiled
        self.assertEqual(cnts.frame_count, 1)

    def test_complex_closure(self):
        @torch.compile
        def forward(y):
            def a():
                def x(z):
                    return y + z

                return x

            return a()

        input1 = torch.rand([2])
        input2 = torch.rand([2])
        res = forward(input1)(input2)
        self.assertTrue(same(res, input1 + input2))

    def test_non_inlined_closure(self):
        @torch.compile()
        def program(x, y):
            one = lambda x, y: x + y

            def inner():
                # Force no inlining
                torch._dynamo.graph_break()
                return one(x, y)

            res = inner()
            one = lambda x, y: x - y
            res += inner()
            return res

        input1 = torch.randn(1)
        input2 = torch.randn(1)

        self.assertTrue(same(program(input1, input2), input1 + input1))


def udf_mul(x, y):
    return x * y


class SmallNN(torch.nn.Module):
    def forward(self, x, y):
        combined = torch.cat((x, y), dim=1)
        out = torch.nn.ReLU()(combined)
        out = torch.nn.ReLU()(out)
        return out


def udf_module(mod, x, y):
    return mod(x, y)


def global_func_with_default_tensor_args(
    x=torch.zeros((2, 2)), *, kw_x=torch.zeros((1, 2))
):
    x.add_(1)
    kw_x.add_(1)
    return x, kw_x


class ModuleWithDefaultTensorArgsMethod(torch.nn.Module):
    def forward(self, x=torch.zeros((2, 2)), *, kw_x=torch.zeros((1, 2))):
        x.add_(1)
        kw_x.add_(1)
        return x, kw_x


class WrapperModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = ModuleWithDefaultTensorArgsMethod()

    def forward(self):
        return self.m()


# Define shared triton constants here.
CONSTANT_C = 4
STRING_CONSTANT_C = "CONSTANT_C"
BOOL_CONSTANT_C = True


class DefaultsTests(torch._dynamo.test_case.TestCase):
    def test_func_default_tensor_args(self):
        """
        Tests that we indeed reference (and mutate) "the one" default tensor arg
        stored on the globally allocated function object, both from the orig and
        compiled function
        """

        def func():
            return global_func_with_default_tensor_args()

        cnts = torch._dynamo.testing.CompileCounter()
        compiled_func = torch.compile(func, backend=cnts)
        for i in range(4):
            if i % 2 == 0:
                x, kw_x = func()
            else:
                x, kw_x = compiled_func()
            # the inner func mutates += 1 each call
            self.assertTrue(same(x, torch.ones_like(x) + i))
            self.assertTrue(same(kw_x, torch.ones_like(kw_x) + i))
        # Calling compiled_func twice does not recompile
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

        # But with a change to the guarded default tensor, we do recompile
        with patch.object(
            global_func_with_default_tensor_args,
            "__defaults__",
            (torch.ones((3, 4, 5)),),
        ):
            x, kw_x = compiled_func()
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

        with patch.object(
            global_func_with_default_tensor_args,
            "__kwdefaults__",
            {"kw_x": torch.ones((3, 4, 5))},
        ):
            x, kw_x = compiled_func()
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 6)

    def test_meth_default_tensor_args(self):
        """
        Tests that we indeed reference (and mutate) "the one" default tensor arg
        stored on the globally allocated function object, both from the orig and
        compiled function
        """
        mod = WrapperModule()
        cnts = torch._dynamo.testing.CompileCounter()
        compiled_mod = torch.compile(mod, backend=cnts)
        for i in range(4):
            if i % 2 == 0:
                x, kw_x = mod()
            else:
                x, kw_x = compiled_mod()
            # the inner func mutates += 1 each call
            self.assertTrue(same(x, torch.ones_like(x) + i))
            self.assertTrue(same(kw_x, torch.ones_like(kw_x) + i))
        # Calling compiled_func twice does not recompile
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

        # But with a change to the guarded default tensor, we do recompile
        with patch.object(
            ModuleWithDefaultTensorArgsMethod.forward,
            "__defaults__",
            (torch.ones((3, 4, 5)),),
        ):
            x, kw_x = compiled_mod()
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

        with patch.object(
            ModuleWithDefaultTensorArgsMethod.forward,
            "__kwdefaults__",
            {"kw_x": torch.ones((3, 4, 5))},
        ):
            x, kw_x = compiled_mod()
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 6)

    def test_func_default_torch_args(self):
        """
        Tests other types of torch types as function default (size, dtype, device)
        """

        def func_with_default_torch_args(
            dt=torch.float16, ds=torch.Size((1, 2, 3)), dd=torch.device("cpu")
        ):
            return torch.ones(ds, dtype=dt, device=dd)

        def func():
            return func_with_default_torch_args()

        cnts = torch._dynamo.testing.CompileCounter()
        compiled_func = torch.compile(func, backend=cnts)
        out = func()
        compiled_out = compiled_func()
        self.assertEqual(out.dtype, compiled_out.dtype)
        self.assertEqual(out.device, compiled_out.device)
        self.assertEqual(out.size(), compiled_out.size())
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    @requires_cuda()
    def test_triton_kernel_with_kernel_param(self):
        @triton.jit
        def pass_kernel(kernel):
            pass

        @torch.compile(backend="eager")
        def f(x):
            grid = (x.numel(),)
            pass_kernel[grid](kernel=x)

        t1 = torch.rand(5, device="cuda")
        f(t1)
        # No need to assert anything, the goal is to make sure dynamo does
        # not crash

    @requires_cuda()
    def test_triton_kernel_higher_order_func(self):
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        add_kernel_id = kernel_side_table.add_kernel(add_kernel)

        t1 = torch.rand(5, device="cuda")
        t2 = torch.rand(5, device="cuda")

        torch_add = t1 + t2

        # Test higher order function with mutation
        output = torch.zeros_like(t1)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        triton_kernel_wrapper_mutation(
            kernel_idx=add_kernel_id,
            grid=[grid],
            kwargs={
                "in_ptr0": t1,
                "in_ptr1": t2,
                "out_ptr": output,
                "n_elements": n_elements,
                "BLOCK_SIZE": 16,
            },
        )
        self.assertEqual(output, torch_add)
        # Make sure it is modified
        self.assertNotEqual(output, torch.zeros_like(t1))

        # Test higher order function without mutation
        output = torch.zeros_like(t1)
        out_dict = triton_kernel_wrapper_functional(
            kernel_idx=add_kernel_id,
            grid=[grid],
            kwargs={
                "in_ptr0": t1,
                "in_ptr1": t2,
                "out_ptr": output,
                "n_elements": n_elements,
                "BLOCK_SIZE": 16,
            },
            tensors_to_clone=["in_ptr0", "in_ptr1", "out_ptr"],
        )
        self.assertEqual(out_dict["out_ptr"], torch_add)
        # Make sure it is NOT modified
        self.assertEqual(output, torch.zeros_like(t1))

    @requires_cuda()
    @skipIfRocm
    def test_triton_kernel_functionalize(self):
        import functorch
        from functorch import make_fx
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        from torch._subclasses.functional_tensor import (
            CppFunctionalizeAPI,
            FunctorchFunctionalizeAPI,
            PythonFunctionalizeAPI,
        )

        kernel_side_table.reset_table()

        def f(x, output):
            out = triton_kernel_wrapper_functional(
                kernel_idx=kernel_side_table.add_kernel(mul2_kernel),
                grid=[(x.numel(),)],
                kwargs={
                    "in_ptr0": x,
                    "out_ptr": output,
                    "n_elements": output.numel(),
                    "BLOCK_SIZE": 16,
                },
                tensors_to_clone=["in_ptr0", "out_ptr"],
            )
            return out["out_ptr"]

        t1 = torch.rand(5, device="cuda")
        t2 = torch.rand(5, device="cuda")

        gm = make_fx(PythonFunctionalizeAPI().functionalize(f))(t1, t2)
        # Make sure t2 was not modified
        self.assertNotEqual(gm(t1, t2), t2)

        gm = make_fx(CppFunctionalizeAPI().functionalize(f))(t1, t2)
        # Make sure t2 was not modified
        self.assertNotEqual(gm(t1, t2), t2)

        gm = make_fx(torch.func.functionalize(f))(t1, t2)
        # Make sure t2 was not modified
        self.assertNotEqual(gm(t1, t2), t2)

        gm = make_fx(f, tracing_mode="fake")(t1, t2)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1, output_1):
    triton_kernel_wrapper_functional_proxy = torch._higher_order_ops.triton_kernel_wrap.triton_kernel_wrapper_functional(kernel_idx = 0, grid = [(5,)], kwargs = {'in_ptr0': x_1, 'out_ptr': output_1, 'n_elements': 5, 'BLOCK_SIZE': 16}, tensors_to_clone = ['in_ptr0', 'out_ptr']);  x_1 = output_1 = None
    getitem = triton_kernel_wrapper_functional_proxy['in_ptr0']
    getitem_1 = triton_kernel_wrapper_functional_proxy['out_ptr']
    getitem_2 = triton_kernel_wrapper_functional_proxy['n_elements']
    getitem_3 = triton_kernel_wrapper_functional_proxy['BLOCK_SIZE'];  triton_kernel_wrapper_functional_proxy = None
    return getitem_1""",
        )

    @requires_cuda()
    @skipIfRocm
    def test_triton_kernel_mutation_type(self):
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        def prep():
            x = torch.ones(4, device="cuda", requires_grad=True)
            x_func = FunctionalTensor.to_functional(x)
            self.assertTrue(torch._is_functional_tensor(x_func.elem))
            return x_func

        # normal mutation only
        with FakeTensorMode():
            x_func = prep()

            with FunctionalTensorMode():
                x_func.mul_(2)

            self.assertFalse(
                torch._functionalize_are_all_mutations_hidden_from_autograd(x_func.elem)
            )

        # triton kernel mutation only
        with FakeTensorMode():
            x_func = prep()

            with FunctionalTensorMode():
                triton_kernel_wrapper_mutation(
                    kernel_idx=kernel_side_table.add_kernel(mul2_inplace_kernel),
                    grid=[(x_func.numel(),)],
                    kwargs={
                        "ptr": x_func,
                        "n_elements": x_func.numel(),
                        "BLOCK_SIZE": 16,
                    },
                )

            self.assertTrue(
                torch._functionalize_are_all_mutations_hidden_from_autograd(x_func.elem)
            )

        # normal mutation + triton kernel mutation
        with FakeTensorMode():
            x_func = prep()

            with FunctionalTensorMode():
                x_func.mul_(2)
                triton_kernel_wrapper_mutation(
                    kernel_idx=kernel_side_table.add_kernel(mul2_inplace_kernel),
                    grid=[(x_func.numel(),)],
                    kwargs={
                        "ptr": x_func,
                        "n_elements": x_func.numel(),
                        "BLOCK_SIZE": 16,
                    },
                )

            self.assertFalse(
                torch._functionalize_are_all_mutations_hidden_from_autograd(x_func.elem)
            )

    @requires_cuda()
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_with_views(self, dynamic, backend):
        def call_triton_take_view(x: torch.Tensor):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            mul2_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
            return output

        def call_triton_return_view(x: torch.Tensor):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            mul2_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
            return output.view(4, 4)

        t = torch.rand(4, 4, device="cuda")
        t_view = t.view(16)

        compiled_func = torch.compile(
            call_triton_take_view, backend=backend, fullgraph=True, dynamic=dynamic
        )
        self.assertEqual(2 * t_view, compiled_func(t_view))
        self.assertEqual(2 * t, compiled_func(t_view).view(4, 4))

        compiled_func = torch.compile(
            call_triton_return_view, backend=backend, fullgraph=True, dynamic=dynamic
        )
        self.assertEqual(2 * t_view, compiled_func(t).view(16))
        self.assertEqual(2 * t, compiled_func(t))

    @requires_cuda()
    @common_utils.parametrize("grad_fn", [torch.no_grad, torch.enable_grad])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_with_grad_option(self, grad_fn, backend):
        def call_triton(x: torch.Tensor):
            with grad_fn():
                output = torch.zeros_like(x)
                n_elements = output.numel()
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
                mul2_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
                return output

        t = torch.rand(5, device="cuda")
        compiled_func = torch.compile(call_triton, backend=backend, fullgraph=True)
        self.assertEqual(2 * t, compiled_func(t))

    @requires_cuda()
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_inner_triton_function(self, backend):
        def f(x: torch.Tensor):
            @triton.jit
            def pow2_kernel(
                in_ptr0,
                out_ptr,
                n_elements,
                BLOCK_SIZE: "tl.constexpr",
            ):
                pid = tl.program_id(axis=0)
                block_start = pid * BLOCK_SIZE
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(in_ptr0 + offsets, mask=mask)
                output = x * x
                tl.store(out_ptr + offsets, output, mask=mask)

            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            pow2_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
            return output

        t = torch.rand(5, device="cuda")

        compiled_func = torch.compile(f, backend=backend, fullgraph=True)
        # do for later(oulgen): NYI - Support this
        # self.assertEqual(t * t, compiled_func(t))

    @requires_cuda()
    @common_utils.parametrize("grad", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    @patch.object(torch._inductor.config, "implicit_fallbacks", False)
    def test_triton_kernel_no_clones(self, grad, dynamic):
        from torch._inductor.utils import run_and_get_code

        def call_triton(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
            n_elements = output.numel()

            tmp = torch.add(x, 1)
            grid = (x.numel(),)
            add_kernel.run(x, y, output, n_elements, grid=grid, BLOCK_SIZE=16)

            return output, tmp

        t1 = torch.rand(5, device="cuda", requires_grad=grad)
        t2 = torch.rand(5, device="cuda", requires_grad=grad)
        o1 = torch.zeros_like(t1, requires_grad=grad)

        torch_add = call_triton(t1, t2, o1)
        metrics.reset()
        o2 = torch.zeros_like(t1, requires_grad=grad)
        test, codes = run_and_get_code(
            torch.compile(call_triton, dynamic=dynamic), t1, t2, o2
        )
        if not grad:
            self.assertEqual(metrics.generated_kernel_count, 1)
        self.assertEqual(torch_add, test)
        # These two asserts are not optimal since it requires original aten
        # to be in the metadata, so there might be false negatives
        self.assertTrue("aten.copy" not in codes[0])
        self.assertTrue("aten.clone" not in codes[0])
        # The following checks that there are only the tensor output is in
        # the compiled graph
        if dynamic and grad:
            self.assertTrue("return (buf0, s0, )" in codes[0])
        else:
            self.assertTrue("return (buf0, )" in codes[0])

    @requires_cuda()
    @skipIfRocm
    def test_triton_kernel_caching(self):
        from torch._inductor.utils import run_and_get_code

        def add_in_loop(
            x: torch.Tensor,
            y: torch.Tensor,
        ):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel_autotuned[grid](x, y, output, n_elements)
            return output

        def call_triton_add(
            x: torch.Tensor,
            y: torch.Tensor,
        ):
            for i in range(4):
                x = add_in_loop(x, y)
            return x

        t1 = torch.ones(5, device="cuda")
        t2 = torch.ones(5, device="cuda")

        test, (code,) = run_and_get_code(torch.compile(call_triton_add), t1, t2)
        self.assertEqual(test, 5 * torch.ones(5, device="cuda"))
        self.assertTrue("add_kernel_autotuned_1.run" not in code)

    @requires_cuda()
    @skipIfRocm
    def test_triton_kernel_caching_duplicate(self):
        from torch._inductor.utils import run_and_get_code

        class C:
            @triton.jit
            def pass_kernel(
                in_ptr0,
                out_ptr,
                n_elements,
                BLOCK_SIZE: "tl.constexpr",
            ):
                pass

        class D:
            @triton.jit
            def pass_kernel(
                in_ptr0,
                out_ptr,
                n_elements,
                BLOCK_SIZE: "tl.constexpr",
            ):
                pass

        def call_triton(x: torch.Tensor):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            C.pass_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
            D.pass_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)

        t = torch.ones(5, device="cuda")
        test, (code,) = run_and_get_code(torch.compile(call_triton), t)
        # Make sure we emitted two kernels here
        self.assertTrue("pass_kernel_0.run" in code)
        self.assertTrue("pass_kernel_1.run" in code)

    @requires_cuda()
    @skipIfRocm
    def test_triton_kernel_various_args(self):
        @triton.autotune(
            configs=[triton.Config({"BLOCK_SIZE": 128})],
            key=[],
        )
        @triton.jit
        def pass_kernel(
            out_ptr,
            n_elements,
            dummy_None,
            dummy_empty,
            dummy_float,
            BLOCK_SIZE: "tl.constexpr",
            RANDOM_SIZE: "tl.constexpr",
        ):
            pass

        @torch.compile
        def call_triton(output):
            n_elements = output.numel()
            grid = (n_elements,)
            pass_kernel[grid](
                output,
                n_elements,
                None,
                torch.empty_like(output),
                3.1415926,
                RANDOM_SIZE=0,
            )
            return output

        output = torch.randn(5, device="cuda")
        # Make sure this does not crash
        call_triton(output)

    @requires_cuda()
    @skipIfRocm
    def test_triton_kernel_dependancies(self):
        def call_triton(
            x: torch.Tensor,
            y: torch.Tensor,
        ):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel_autotuned[grid](x, y, output, n_elements)
            output2 = torch.zeros_like(output)
            add_kernel_autotuned[grid](output, y, output2, n_elements)
            output3 = torch.add(output2, 1)
            return output3

        t1 = torch.rand(5, device="cuda")
        t2 = torch.rand(5, device="cuda")
        torch_result = call_triton(t1, t2)
        compiled_result = torch.compile(call_triton)(t1, t2)
        self.assertEqual(torch_result, compiled_result)

    @requires_cuda()
    @common_utils.parametrize("grad", [False, True])
    def test_triton_kernel_multi_kernel(self, grad):
        @triton.jit
        def mul2_and_add_and_zero_negatives_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
            ACTIVATION: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            indirection_kernel(
                in_ptr0,
                in_ptr0,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                ACTIVATION="mul2_inplace_kernel",
            )
            indirection_kernel(
                in_ptr1,
                in_ptr1,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                ACTIVATION="mul2_inplace_kernel",
            )
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            if ACTIVATION == "zero_negs":
                output = zero_negs(output)
            tl.store(out_ptr + offsets, output, mask=mask)

        @torch.compile
        def call_triton(
            x: torch.Tensor,
            y: torch.Tensor,
            xi: torch.Tensor,
            yi: torch.Tensor,
            output: torch.Tensor,
            outputi: torch.Tensor,
        ):
            n_elements = output.numel()

            grid = (x.numel(),)
            mul2_and_add_and_zero_negatives_kernel[grid](
                x, y, output, n_elements, BLOCK_SIZE=16, ACTIVATION="zero_negs"
            )
            mul2_and_add_and_zero_negatives_kernel[grid](
                xi, yi, outputi, n_elements, BLOCK_SIZE=16, ACTIVATION=None
            )

            return (output, outputi)

        t1 = torch.tensor(
            [-2.0, -1.0, 0.0, 1.0, 2.0], device="cuda", requires_grad=grad
        )
        t2 = torch.tensor(
            [-2.0, -1.0, 0.0, 1.0, 2.0], device="cuda", requires_grad=grad
        )
        float_result = 2 * t1 + 2 * t2
        float_result = float_result.where(float_result >= 0, 0.0)

        t1i = torch.randint(-2, 2, (5,), device="cuda")
        t2i = torch.randint(-2, 2, (5,), device="cuda")
        o_tensor = torch.zeros_like(t1, requires_grad=grad)
        oi = torch.zeros_like(t1i)
        int_result = 2 * t1i + 2 * t2i

        (result, resulti) = call_triton(t1, t2, t1i, t2i, o_tensor, oi)
        self.assertEqual(float_result, result)
        self.assertEqual(int_result, resulti)

    @requires_cuda()
    def test_triton_kernel_constants(self):
        @triton.jit
        def mulC_kernel(
            in_ptr0,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
            CONSTANT_NAME: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            if CONSTANT_NAME.value == STRING_CONSTANT_C:
                output = CONSTANT_C * x
            if BOOL_CONSTANT_C:
                output *= CONSTANT_C
            tl.store(out_ptr + offsets, output, mask=mask)

        def call_triton(
            x: torch.Tensor,
        ):
            output = torch.zeros_like(x)
            n_elements = output.numel()

            grid = (x.numel(),)
            mulC_kernel[grid](
                x, output, n_elements, BLOCK_SIZE=16, CONSTANT_NAME="CONSTANT_C"
            )
            return output

        # Triton kernels capture global constants by their parse time value
        # not runtime value
        global CONSTANT_C
        prev_c = CONSTANT_C
        # If the behavior of triton kernels change, this test will fail
        CONSTANT_C = 10
        assert CONSTANT_C != prev_c

        t = torch.randn(5, device="cuda")
        torch_result = call_triton(t)
        compiled_result = torch.compile(call_triton)(t)

        self.assertEqual(torch_result, compiled_result)

        # reset back
        CONSTANT_C = prev_c

    @requires_cuda()
    @skipIfRocm
    @common_utils.parametrize("grad", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("grid_type", [1, 2, 3])
    def test_triton_kernel_autotune(self, grad, dynamic, backend, grid_type):
        def call_triton(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
            n_elements = output.numel()

            def grid_fn(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            if grid_type == 1:
                grid = (n_elements,)
            elif grid_type == 2:
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            elif grid_type == 3:
                grid = grid_fn

            add_kernel_autotuned[grid](x, y, output, n_elements)
            return output

        t1 = torch.rand(256, device="cuda", requires_grad=grad)
        t2 = torch.rand(256, device="cuda", requires_grad=grad)
        output = torch.zeros_like(t1, requires_grad=grad)

        torch_add = call_triton(t1, t2, output)
        compiled_func = torch.compile(
            call_triton, backend=backend, fullgraph=True, dynamic=dynamic
        )

        output2 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, output2), torch_add)

    @requires_cuda()
    @skipIfRocm
    @common_utils.parametrize("grad", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("grid_type", [1, 2, 3])
    def test_triton_kernel_2d_autotune(self, grad, dynamic, backend, grid_type):
        def call_triton(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
            x_elements = output.size()[0]
            y_elements = output.size()[1]

            def grid_fn(meta):
                return (
                    triton.cdiv(x_elements, meta["BLOCK_SIZE_X"]),
                    triton.cdiv(y_elements, meta["BLOCK_SIZE_Y"]),
                )

            if grid_type == 1:
                grid = (x_elements, y_elements)
            elif grid_type == 2:
                grid = lambda meta: (
                    triton.cdiv(x_elements, meta["BLOCK_SIZE_X"]),
                    triton.cdiv(y_elements, meta["BLOCK_SIZE_Y"]),
                )
            elif grid_type == 3:
                grid = grid_fn

            add_kernel_2d_autotuned[grid](x, y, output, x_elements, y_elements)
            return output

        t1 = torch.rand((512, 256), device="cuda", requires_grad=grad)
        t2 = torch.rand((512, 256), device="cuda", requires_grad=grad)
        output = torch.zeros_like(t1, requires_grad=grad)

        torch_result = call_triton(t1, t2, output)
        compiled_func = torch.compile(
            call_triton, backend=backend, fullgraph=True, dynamic=dynamic
        )
        output2 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, output2), torch_result)

    @requires_cuda()
    @common_utils.parametrize("grad", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @patch.object(torch._inductor.config, "implicit_fallbacks", False)
    def test_triton_kernel_native(self, grad, dynamic, backend):
        def call_triton_add(
            x: torch.Tensor,
            y: torch.Tensor,
            output: torch.Tensor,
            grid_type: int,
            num=1,
            positional=False,
        ):
            n_elements = output.numel()

            def grid_fn(meta):
                return (triton.cdiv(num, meta["BLOCK_SIZE"]),)

            if grid_type == 0:
                grid = (x.numel(),)
            elif grid_type == 1:
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            else:
                grid = grid_fn

            if positional:
                add_kernel[grid](x, y, output, n_elements, 16)
            else:
                add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)

            return output

        t1 = torch.rand(5, device="cuda", requires_grad=grad)
        t2 = torch.rand(5, device="cuda", requires_grad=grad)
        o1 = torch.zeros_like(t1, requires_grad=grad)

        torch_add = t1 + t2

        # No Dynamo -- Make sure triton kernel works
        self.assertEqual(call_triton_add(t1, t2, o1, 1), torch_add)
        # No Dynamo -- Make sure triton kernel works (with positional BLOCK_SIZE)
        o2 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(call_triton_add(t1, t2, o2, 1, True), torch_add)

        # With Dynamo
        compiled_func = torch.compile(
            call_triton_add, backend=backend, fullgraph=True, dynamic=dynamic
        )
        # With simple kernel
        o3 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, o3, 0), torch_add)
        # With lambda kernel
        o4 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, o4, 1), torch_add)
        # With lambda kernel (with positional BLOCK_SIZE)
        o5 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, o5, 1, 1, True), torch_add)
        # With user defined function kernel
        o6 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, o6, 2, 200), torch_add)

    def test_dataclass_factory(self):
        @dataclass
        class Output:
            scalar: int = 2
            named_tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
            lists: List[torch.Tensor] = field(default_factory=list)

            def scale(self):
                return self.scalar * 2

        def fn(x):
            # Check default dict assignment
            a = Output(1)
            # Check that dataclass methods can be inlined
            scaled_value = a.scale()

            # Check that normal assignment works
            b = Output(5, named_tensors={"x": x})

            # Check default int assignment
            c = Output()

            # Check that the default members are properly initialized
            if isinstance(a.named_tensors, dict):
                x = torch.sin(x)

            # Change dataclass
            c.scalar = 6
            c.named_tensors["x"] = x

            # Return dataclaass as well to check reconstruction
            return c, torch.cos(x) * scaled_value + b.named_tensors["x"] + c.scalar

        cnts = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.randn(4)
        eager_dataclass, out = fn(x)
        compiled_dataclass, compiled_out = compiled_fn(x)
        self.assertEqual(eager_dataclass.scalar, compiled_dataclass.scalar)
        self.assertEqual(
            eager_dataclass.named_tensors["x"], compiled_dataclass.named_tensors["x"]
        )
        self.assertTrue(same(out, compiled_out))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 5)

    def test_dataclass_nested(self):
        @dataclass
        class Base:
            outer_a: int
            outer_b: int

        @dataclass
        class Derived(Base):
            inner_a: Any = field(default_factory=list)

        def fn(x):
            l_derived = Derived(1, 2)
            return l_derived.outer_a * x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        res = fn(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)

    def test_listlike_of_tensors_contains_constant(self):
        for listlike in [set, list]:

            def fn(x):
                x.add_(1)
                s = listlike([x])
                res = 1 in s
                return res

            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            x = torch.randn(1)
            ref = opt_fn(x)
            res = fn(x)
            self.assertEqual(ref, res)

    def test_cast_tensor_single_elem(self):
        with torch._dynamo.config.patch({"capture_scalar_outputs": True}):
            for t, val in [
                (float, 1.0),
                (float, 1),
                (float, True),
                (int, 1),
                (int, False),
                # (int, 1.0), # fails due to a >= 0 comparison in sym_int
            ]:  # , bool, complex]: no casting for sym_bool, no sym_complex

                def fn(x):
                    x = x + 1
                    return t(x)

                opt_fn = torch.compile(
                    fn, backend="eager", fullgraph=True, dynamic=False
                )
                x = torch.tensor([val])
                res = fn(x)
                ref = opt_fn(x)
                self.assertEqual(ref, res)

                # Cannot handle non single-elem
                with self.assertRaises(ValueError):
                    fn(torch.tensor([val] * 2))
                with self.assertRaises(torch._dynamo.exc.TorchRuntimeError):
                    opt_fn(torch.tensor([val] * 2))

    def test_set_construction(self):
        def fn(x):
            y = x.add_(1)
            s = set({x})
            s.add(y)
            return len(s)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        res = fn(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)

    def test_is_tensor_tensor(self):
        def fn(x, y):
            if x is y:
                return x * 2
            else:
                return x + y

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        x = torch.zeros(2)
        y = torch.ones(2)

        self.assertEqual(fn(x, y), fn_opt(x, y))
        self.assertEqual(fn(x, x), fn_opt(x, x))

    def test_is_mutated_tensor_tensor(self):
        def fn(x):
            y = x.add_(1)
            return x is y

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        z = torch.ones(4)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_mutated_tensor_tensor_across_graph_break(self):
        def fn(x):
            y = x.add_(1)
            cond = x is y
            x.add_(1)
            # The real tensor values are recovered when graph breaking.
            # Hence we recover the invariant.
            torch._dynamo.graph_break()
            x.add_(1)
            return x is y, cond

        fn_opt = torch.compile(backend="eager", dynamic=True)(fn)

        z = torch.ones(4)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_mutated_tensor_tensor(self):
        def fn(x):
            y = x.add_(1)
            return y is x

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        z = torch.ones(4, 1)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_init_in_compile_mutated_tensor_tensor(self):
        def fn(x):
            z = x.clone()
            y = z.add_(1)
            return y is z

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        z = torch.ones(4, 1)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_init_in_compile_vmapped_mutated_tensor_tensor(self):
        def fn(z):
            x = z.clone()
            y = torch.vmap(torch.Tensor.acos_)(x)
            _ = y is z
            return y is x

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        z = torch.ones(4, 1)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_vmapped_mutated_tensor_tensor(self):
        def fn(x):
            y = torch.vmap(torch.Tensor.acos_)(x)
            return y is x

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        z = torch.ones(4, 1)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_init_in_compile_vmapped_mutated_tensor_tensor_multi_arg(self):
        def fn(y, z):
            a = y.clone()
            b = z.clone()

            def g(a, b):
                return a.acos_(), b.acos_()

            c, dd = torch.vmap(g)(a, b)
            return a is c is b is dd

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        y = torch.ones(4, 2)
        z = torch.ones(4, 10)

        self.assertEqual(fn(y, z), fn_opt(y, z))
        self.assertEqual(fn(y, y), fn_opt(y, y))

    def test_in_set_would_fail_broadcast(self):
        param = torch.zeros(5)
        param2 = torch.zeros(5, 10)

        tensor_list = set()
        tensor_list.add(param2)
        assert param not in tensor_list

        def fn(param, param2):
            param.add_(1)
            tensor_list = set([param2])
            return param in tensor_list

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        self.assertEqual(opt_fn(param, param2), fn(param, param2))
        self.assertEqual(cnts.frame_count, 1)
        # Test aliased
        self.assertEqual(opt_fn(param, param), fn(param, param))
        self.assertEqual(cnts.frame_count, 2)  # Recompiles

    def test_in_set_inplace(self):
        param = torch.zeros(5)
        param2 = torch.zeros(5, 10)

        tensor_list = set()
        tensor_list.add(param2)
        assert param not in tensor_list

        def fn(param, param2):
            y = param.add_(1)  # Tensor method
            z = torch.Tensor.add_(y, 1)  # torch function
            tensor_list = set([param2])
            return y in tensor_list and z in tensor_list

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        self.assertEqual(opt_fn(param, param2), fn(param, param2))
        self.assertEqual(cnts.frame_count, 1)
        # Test aliased
        self.assertEqual(opt_fn(param, param), fn(param, param))
        self.assertEqual(cnts.frame_count, 2)  # Recompiles

    @unittest.skipIf(
        sys.version_info < (3, 10),
        "zip strict kwargs not implemented for Python < 3.10",
    )
    def test_zip_strict(self):
        def fn(x, ys, zs):
            x = x.clone()
            for y, z in zip(ys, zs, strict=True):
                x += y * z
            return x

        opt_fn = torch._dynamo.optimize(backend="eager")(fn)
        nopython_fn = torch._dynamo.optimize(backend="eager", nopython=True)(fn)

        x = torch.ones(3)
        ys = [1.0, 2.0, 3.0]
        zs = [2.0, 5.0, 8.0]

        self.assertEqual(opt_fn(x, ys, zs), fn(x, ys, zs))

        # If nopython, should raise UserError
        with self.assertRaisesRegex(torch._dynamo.exc.UserError, "zip()"):
            nopython_fn(x, ys[:1], zs)

        # Should cause fallback if allow graph break
        with self.assertRaisesRegex(ValueError, "zip()"):
            opt_fn(x, ys[:1], zs)

    def test_compare_constant_and_tensor(self):
        for op in [
            operator.lt,
            operator.le,
            operator.gt,
            operator.ge,
            operator.ne,
            operator.eq,
        ]:

            def fn(x):
                return op(-10, x)

            opt_fn = torch.compile(fullgraph=True)(fn)

            x = torch.randn(10)
            self.assertEqual(opt_fn(x), fn(x))


common_utils.instantiate_parametrized_tests(DefaultsTests)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
