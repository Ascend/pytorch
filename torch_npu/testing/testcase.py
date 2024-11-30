r"""Importing this file must **not** initialize NPU context. test_distributed
relies on this assumption to properly run. This means that when this is imported
no NPU calls shall be made, including torch.npu.device_count(), etc.

torch.testing._internal.common_npu.py can freely initialize NPU context when imported.
"""
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import contextmanager
from numbers import Number
from unittest.result import TestResult
from unittest.util import strclass

import sys
import os
import re
import unittest
import time
import warnings
import random
import inspect
import torch
import expecttest
import numpy as np

from torch import inf

from torch_npu.testing.common_utils import set_npu_device, is_iterable, iter_indices
from torch_npu.testing.common_utils import PERF_TEST_ENABLE, PerfBaseline

# Environment variables set in ci script.
IS_IN_CI = os.getenv('IN_CI') == '1'
TEST_REPORT_PATH = os.getenv("TEST_REPORT_PATH", "test-reports")


def run_tests():
    argv = sys.argv
    if IS_IN_CI:
        # import here so that non-CI doesn't need xmlrunner installed
        import xmlrunner
        filename = inspect.getfile(sys._getframe(1))
        strip_py = re.sub(r'.py$', '', filename)
        test_filename = re.sub('/', r'.', strip_py)
        test_report_path = os.path.join(TEST_REPORT_PATH, test_filename)
        verbose = '--verbose' in argv or '-v' in argv
        if verbose:
            print(f'Test results will be stored in {test_report_path}')
        unittest.main(argv=argv, testRunner=xmlrunner.XMLTestRunner(output=test_report_path,
                                                                    verbosity=2 if verbose else 1))
    else:
        unittest.main(argv=argv)


class TestCase(expecttest.TestCase):
    _precision = 1e-5
    maxDiff = None
    exact_dtype = False

    def __init__(self, method_name='runTest'):
        super(TestCase, self).__init__(method_name)

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, prec):
        self._precision = prec

    @classmethod
    def setUpClass(cls):
        cls.npu_device = set_npu_device()

    def setUp(self):
        seed = int(os.getenv('SEED', "666"))
        torch.manual_seed(seed)
        random.seed(seed)

    def assertTensorsSlowEqual(self, x, y, prec=None, message=''):
        self.assertEqual(x.size(), y.size())
        self.assertEqual(x.dtype, y.dtype)
        y = y.type_as(x)
        if x.dtype == torch.bool:
            self.assertEqual(x, y)
        else:
            max_err = 0
            for index in iter_indices(x):
                max_err = max(max_err, abs(x[index] - y[index]))
            self.assertLessEqual(max_err, prec, message)

    def genSparseTensor(self, size, sparse_dim, nnz, is_uncoalesced, device='cpu'):
        # Assert not given impossible combination, where the sparse dims have
        # empty numel, but nnz > 0 makes the indices containing values.
        if not (all(size[d] > 0 for d in range(sparse_dim)) or nnz == 0):
            raise RuntimeError('invalid arguments')

        v_size = [nnz] + list(size[sparse_dim:])
        v = torch.randn(*v_size, device=device)
        i = torch.rand(sparse_dim, nnz, device=device)
        i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
        i = i.to(torch.long)
        if is_uncoalesced:
            v = torch.cat([v, torch.randn_like(v)], 0)
            i = torch.cat([i, i], 1)

        x = torch.sparse_coo_tensor(i, v, torch.Size(size))

        if not is_uncoalesced:
            x = x.coalesce()
        else:
            # NOTE: We do clone() after detach() here because we need to be able to change size/storage of x afterwards
            x = x.detach().clone()
        return x, x._indices().clone(), x._values().clone()

    def safeToDense(self, t):
        r = self.safeCoalesce(t)
        return r.to_dense()

    def safeCoalesce(self, t):
        tc = t.coalesce()
        self.assertEqual(tc.to_dense(), t.to_dense())
        self.assertTrue(tc.is_coalesced())

        # Our code below doesn't work when nnz is 0, because
        # then it's a 0D tensor, not a 2D tensor.
        if t._nnz() == 0:
            self.assertEqual(t._indices(), tc._indices())
            self.assertEqual(t._values(), tc._values())
            return tc

        value_map = {}
        for idx, val in zip(t._indices().t(), t._values()):
            idx_tup = tuple(idx.tolist())
            if idx_tup in value_map:
                value_map[idx_tup] += val
            else:
                value_map[idx_tup] = val.clone() if isinstance(val, torch.Tensor) else val

        new_indices = sorted(list(value_map.keys()))
        new_values = [value_map.get(idx) for idx in new_indices]
        if t._values().ndimension() < 2:
            new_values = t._values().new(new_values)
        else:
            new_values = torch.stack(new_values)

        new_indices = t._indices().new(new_indices).t()
        tg = t.new(new_indices, new_values, t.size())

        self.assertEqual(tc._indices(), tg._indices())
        self.assertEqual(tc._values(), tg._values())

        if t.is_coalesced():
            self.assertEqual(tc._indices(), t._indices())
            self.assertEqual(tc._values(), t._values())

        return tg
    
    def assertRtolEqual(self, x, y, prec=1.e-4, prec16=1.e-3, auto_trans_dtype=False, message=None):

        def _assertRtolEqual(x, y, prec, prec16, message):
            def compare_res(pre, minimum):
                diff = y - x
                # check that NaNs are in the same locations
                nan_mask = np.isnan(x)
                if not np.equal(nan_mask, np.isnan(y)).all():
                    self.fail(message)
                if nan_mask.any():
                    diff[nan_mask] = 0
                result = np.abs(diff)
                deno = np.maximum(np.abs(x), np.abs(y))
                result_atol = np.less_equal(result, pre)
                result_rtol = np.less_equal(result / np.add(deno, minimum), pre)
                if not result_rtol.all() and not result_atol.all():
                    if np.sum(result_rtol == False) > size * pre and np.sum(result_atol == False) > size * pre:
                        self.fail("result error")

            minimum16 = 6e-8
            minimum = 10e-10

            if isinstance(x, Sequence) and isinstance(y, Sequence):
                for x_, y_ in zip(x, y):
                    _assertRtolEqual(x_, y_, prec, prec16, message)
                return

            if isinstance(x, torch.Tensor) and isinstance(y, Sequence):
                y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
            elif isinstance(x, Sequence) and isinstance(y, torch.Tensor):
                x = torch.as_tensor(x, dtype=y.dtype, device=y.device)

            if torch.is_tensor(x) and torch.is_tensor(y):
                if auto_trans_dtype:
                    x = x.to(y.dtype)
                if (x.dtype == torch.bfloat16) and (y.dtype == torch.bfloat16):
                    if (x.shape != y.shape):
                        self.fail("shape error!")
                    result = torch.allclose(x.cpu(), y.cpu(), rtol=prec16, atol=prec16)
                    if not result:
                        self.fail("result error!")
                    return
                x = x.detach().cpu().numpy()
                y = y.detach().cpu().numpy()  
            elif isinstance(x, Number) and isinstance(y, Number):
                x = np.array(x)
                y = np.array(y)

            size = x.size
            if (x.shape != y.shape):
                self.fail("shape error")
            if (x.dtype != y.dtype):
                self.fail("dtype error")
            dtype_list = [np.bool_, np.uint16, np.int16, np.int32, np.float16, 
                        np.float32, np.int8, np.uint8, np.int64, np.float64]
            if x.dtype not in dtype_list:
                self.fail("required dtype in [np.bool_, np.uint16, np.int16, " +
                        "np.int32, np.float16, np.float32, np.int8, np.uint8, np.int64]")
            if x.dtype == np.bool_:
                result = np.equal(x, y)
                if not result.all():
                    self.fail("result error")
            elif (x.dtype == np.float16):
                compare_res(prec16, minimum16)
            elif (x.dtype in [np.float32, np.int8, np.uint8, np.uint16, np.int16, np.int32, np.int64, np.float64]):
                compare_res(prec, minimum)
            else:
                self.fail("required numpy object")

        _assertRtolEqual(x, y, prec, prec16, message)

    def _assert_tensor_equal(self, a, b, message, exact_dtype, allow_inf, prec):
        super(TestCase, self).assertEqual(a.size(), b.size(), message)
        if exact_dtype:
            self.assertEqual(a.dtype, b.dtype)
        if a.numel() > 0:
            if (a.device.type == 'cpu' and (a.dtype == torch.float16 or a.dtype == torch.bfloat16)):
                # CPU half and bfloat16 tensors don't have the methods we need below
                a = a.to(torch.float32)
            b = b.to(a)

            if (a.dtype == torch.bool) != (b.dtype == torch.bool):
                raise TypeError("Was expecting both tensors to be bool type.")

            if a.dtype == torch.bool and b.dtype == torch.bool:
                # we want to respect precision but as bool doesn't support subtraction,
                # boolean tensor has to be converted to int
                a = a.to(torch.int)
                b = b.to(torch.int)

            diff = a - b
            if a.dtype.is_complex or a.dtype.is_floating_point:
                # check that NaNs are in the same locations
                nan_mask = torch.isnan(a)
                self.assertTrue(torch.equal(nan_mask, torch.isnan(b)), message)
                diff[nan_mask] = 0
                # inf check if allow_inf=True
                if allow_inf:
                    inf_mask = torch.isinf(a)
                    inf_sign = inf_mask.sign()
                    self.assertTrue(torch.equal(inf_sign, torch.isinf(b).sign()), message)
                    diff[inf_mask] = 0

            if diff.is_signed() and diff.dtype != torch.int8:
                diff = diff.abs()
                # if diff is complex, the imaginary component for diff will be 0
                # from the previous step, hence converting it to float and double is fine.
                if diff.dtype == torch.complex64:
                    diff = diff.to(torch.float)
                elif diff.dtype == torch.complex128:
                    diff = diff.to(torch.double)
            max_err = diff.max()
            self.assertLessEqual(max_err, prec, message)

    def _assertNumberEqual(self, x, y, prec=None, message='', allow_inf=False, exact_dtype=None):
        if isinstance(x, torch.Tensor) and isinstance(y, Number):
            self._assertNumberEqual(x.item(), y, prec=prec, message=message,
                                    allow_inf=allow_inf, exact_dtype=exact_dtype)

        elif isinstance(y, torch.Tensor) and isinstance(x, Number):
            self._assertNumberEqual(x, y.item(), prec=prec, message=message,
                                    allow_inf=allow_inf, exact_dtype=exact_dtype)

        else:
            if abs(x) == inf or abs(y) == inf:
                if allow_inf:
                    super(TestCase, self).assertEqual(x, y, message)
                else:
                    self.fail("Expected finite numeric values - x={}, y={}".format(x, y))
                return
            super(TestCase, self).assertLessEqual(abs(x - y), prec, message)

    def _assertBoolEqual(self, x, y, prec=None, message='', allow_inf=False, exact_dtype=None):
        if isinstance(x, torch.Tensor) and isinstance(y, np.bool_):
            self._assertBoolEqual(x.item(), y, prec=prec, message=message,
                                  allow_inf=allow_inf, exact_dtype=exact_dtype)
        elif isinstance(y, torch.Tensor) and isinstance(x, np.bool_):
            self._assertBoolEqual(x, y.item(), prec=prec, message=message,
                                  allow_inf=allow_inf, exact_dtype=exact_dtype)
        else:
            super(TestCase, self).assertEqual(x, y, message)

    def _assertTensorsEqual(self, x, y, prec=None, message='', allow_inf=False, exact_dtype=None):
        super(TestCase, self).assertEqual(x.is_sparse, y.is_sparse, message)
        super(TestCase, self).assertEqual(x.is_quantized, y.is_quantized, message)
        if x.is_sparse:
            x = self.safeCoalesce(x)
            y = self.safeCoalesce(y)
            self._assert_tensor_equal(x._indices(), y._indices(), message, exact_dtype, allow_inf, prec)
            self._assert_tensor_equal(x._values(), y._values(), message, exact_dtype, allow_inf, prec)
        elif x.is_quantized and y.is_quantized:
            self.assertEqual(x.qscheme(), y.qscheme(), prec=prec,
                                message=message, allow_inf=allow_inf,
                                exact_dtype=exact_dtype)
            if x.qscheme() == torch.per_tensor_affine:
                self.assertEqual(x.q_scale(), y.q_scale(), prec=prec,
                                    message=message, allow_inf=allow_inf,
                                    exact_dtype=exact_dtype)
                self.assertEqual(x.q_zero_point(), y.q_zero_point(),
                                    prec=prec, message=message,
                                    allow_inf=allow_inf, exact_dtype=exact_dtype)
            elif x.qscheme() == torch.per_channel_affine:
                self.assertEqual(x.q_per_channel_scales(), y.q_per_channel_scales(), prec=prec,
                                    message=message, allow_inf=allow_inf,
                                    exact_dtype=exact_dtype)
                self.assertEqual(x.q_per_channel_zero_points(), y.q_per_channel_zero_points(),
                                    prec=prec, message=message,
                                    allow_inf=allow_inf, exact_dtype=exact_dtype)
                self.assertEqual(x.q_per_channel_axis(), y.q_per_channel_axis(),
                                    prec=prec, message=message)
            self.assertEqual(x.dtype, y.dtype)
            self.assertEqual(x.int_repr().to(torch.int32),
                                y.int_repr().to(torch.int32), prec=prec,
                                message=message, allow_inf=allow_inf,
                                exact_dtype=exact_dtype)
        else:
            self._assert_tensor_equal(x, y, message, exact_dtype, allow_inf, prec)

    def assertEqual(self, x, y, prec=None, message='', allow_inf=False, exact_dtype=None):
        if exact_dtype is None:
            exact_dtype = self.exact_dtype

        if isinstance(prec, str) and message == '':
            message = prec
            prec = None
        if prec is None:
            prec = self.precision

        def _assertEqual(x, y, prec=None, message='', allow_inf=False, exact_dtype=None):
            if isinstance(x, Number) or isinstance(y, Number):
                self._assertNumberEqual(x, y, prec=prec, message=message,
                                        allow_inf=allow_inf, exact_dtype=exact_dtype)
            elif isinstance(x, np.bool_) or isinstance(y, np.bool_):
                self._assertBoolEqual(x, y, prec=prec, message=message,
                                    allow_inf=allow_inf, exact_dtype=exact_dtype)
            elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                self._assertTensorsEqual(x, y, prec=prec, message=message,
                                        allow_inf=allow_inf, exact_dtype=exact_dtype)
            elif isinstance(x, (str, bytes)) and isinstance(y, (str, bytes)):
                super(TestCase, self).assertEqual(x, y, message)
            elif type(x) == set and type(y) == set:
                super(TestCase, self).assertEqual(x, y, message)
            elif isinstance(x, dict) and isinstance(y, dict):
                if isinstance(x, OrderedDict) and isinstance(y, OrderedDict):
                    _assertEqual(x.items(), y.items(), prec=prec,
                                 message=message, allow_inf=allow_inf,
                                 exact_dtype=exact_dtype)
                else:
                    _assertEqual(set(x.keys()), set(y.keys()), prec=prec,
                                 message=message, allow_inf=allow_inf,
                                 exact_dtype=exact_dtype)
                    key_list = list(x.keys())
                    _assertEqual([x[k] for k in key_list],
                                 [y[k] for k in key_list],
                                 prec=prec, message=message,
                                 allow_inf=allow_inf, exact_dtype=exact_dtype)
            elif is_iterable(x) and is_iterable(y):
                super(TestCase, self).assertEqual(len(x), len(y), message)
                for x_, y_ in zip(x, y):
                    _assertEqual(x_, y_, prec=prec, message=message,
                                 allow_inf=allow_inf, exact_dtype=exact_dtype)
            else:
                super(TestCase, self).assertEqual(x, y, message)

        _assertEqual(x, y, prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)

    def assertAlmostEqual(self, x, y, places=None, msg=None, delta=None, allow_inf=None):
        prec = delta
        if places:
            prec = 10**(-places)
        self.assertEqual(x, y, prec, msg, allow_inf)

    def assertNotEqual(self, x, y, prec=None, message=''):
        if isinstance(prec, str) and message == '':
            message = prec
            prec = None
        if prec is None:
            prec = self.precision

        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            if x.size() != y.size():
                super(TestCase, self).assertNotEqual(x.size(), y.size())
            self.assertGreater(x.numel(), 0)
            y = y.type_as(x)
            nan_mask = x != x
            if torch.equal(nan_mask, y != y):
                if x.dtype == torch.bool and y.dtype == torch.bool:
                    x = x.to(torch.int)
                    y = y.to(torch.int)
                diff = x - y
                if diff.is_signed():
                    diff = diff.abs()
                diff[nan_mask] = 0
                max_err = diff.max().item()
                self.assertGreaterEqual(max_err, prec, message)
        elif type(x) == str and type(y) == str:
            super(TestCase, self).assertNotEqual(x, y)
        elif is_iterable(x) and is_iterable(y):
            super(TestCase, self).assertNotEqual(x, y)
        else:
            try:
                self.assertGreaterEqual(abs(x - y), prec, message)
                return
            except (TypeError, AssertionError):
                pass
            super(TestCase, self).assertNotEqual(x, y, message)

    def assertObjectIn(self, obj, iterable):
        for elem in iterable:
            if id(obj) == id(elem):
                return
        raise AssertionError("object not found in iterable")

    # NB: The kwargs forwarding to callable robs the 'subname' parameter.
    # If you need it, manually apply your call_fn in a lambda instead.
    def assertExpectedRaises(self, exc_type, call_fn, *args, **kwargs):
        subname = None
        if 'subname' in kwargs:
            subname = kwargs.get('subname')
            del kwargs['subname']
        try:
            call_fn(*args, **kwargs)
        except exc_type as e:
            self.assertExpected(str(e), subname)
            return
        # Don't put this in the try block; the AssertionError will catch it
        self.fail(msg="Did not raise when expected to")

    def assertNotWarn(self, call_fn, msg=''):
        r"""
        Test if :attr:`call_fn` does not raise a warning.
        """
        with self._reset_warning_registry(), warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            call_fn()
            self.assertTrue(len(ws) == 0, msg)

    def assertWarns(self, call_fn, msg=''):
        r"""
        Test if :attr:`call_fn` raises a warning.
        """
        with self._reset_warning_registry(), warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            call_fn()
            self.assertTrue(len(ws) > 0, msg)

    @contextmanager
    def maybeWarnsRegex(self, category, regex=''):
        """Context manager for code that *may* warn, e.g. ``TORCH_NPU_WARN_ONCE``.

        This filters expected warnings from the test log and fails the test if
        any unexpected warnings are caught.
        """
        with self._reset_warning_registry(), warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            # Ignore expected warnings
            warnings.filterwarnings("ignore", message=regex, category=category)
            try:
                yield
            finally:
                msg = 'Caught unexpected warnings:\n' if len(ws) != 0 else None
                for w in ws:
                    msg += warnings.formatwarning(
                        w.message, w.category, w.filename, w.lineno, w.line)
                    msg += '\n'
                if msg is not None:
                    self.fail(msg)

    @contextmanager
    def _reset_warning_registry(self):
        r"""
        warnings.catch_warnings() in Python 2 misses already registered
        warnings. We need to manually clear the existing warning registries to
        ensure catching warnings in a scope.
        """
        # Python 3 has no problem.
        if sys.version_info >= (3,):
            yield
            return

    def assertExpectedStripMangled(self, s, subname=None):
        s = re.sub(r'__torch__[^ ]+', '', s)
        self.assertExpected(s, subname)

    def run(self, result=None):
        # run test to precompile operators
        super(TestCase, self).run(result)
        
        if PERF_TEST_ENABLE:
            performanceResult = TestResult()
            startTime = time.perf_counter()
            super(TestCase, self).run(performanceResult)
            stopTime = time.perf_counter()
            runtime = stopTime - startTime

            if len(performanceResult.errors) == len(performanceResult.failures) == 0:
                methodId = strclass(self.__class__) + "." + self._testMethodName
                baseline = PerfBaseline.get_baseline(methodId)

                if baseline and runtime > baseline * 1.2:
                    errMsg = "Performance test failed. Performance baseline: " \
                            + str(baseline) + "s, current time: " + str(runtime) + "s"
                    perfErr = (self.failureException, self.failureException(errMsg), None)
                    self._feedErrorsToResult(result, [(self, perfErr)])

                if baseline is None or runtime < baseline * 0.9:
                    PerfBaseline.set_baseline(methodId, runtime)
