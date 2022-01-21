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

import inspect
import threading
from functools import wraps
import unittest
import torch
from torch.testing._internal.common_utils import TestCase, TEST_MKL

# Note: Generic Device-Type Testing
#
# [WRITING TESTS]
#
# Write your test class as usual except:
#   (1) Each test method should have one of four signatures:
#
#           (1a) testX(self, device)
#
#           (1b) @DeviceCountAtLeast(<minimum number of devices to run test with>)
#                testX(self, devices)
#
#           (1c) @Dtypes(<list of dtypes>)
#                testX(self, device, dtype)
#
#           (1d) @DeviceCountAtLeast(<minimum number of devices to run test with>)
#                @Dtypes(<list of dtypes>)
#                testX(self, devices, dtype)
#
#
#       Note that the decorators are required for signatures (1b), (1c) and
#       (1d).
#
#       When a test like (1a) is called it will be given a device string,
#       like 'cpu' or 'npu:0.'
#
#       Tests like (1b) are called with a list of device strings, like
#       ['npu:0', 'npu:1']. The first device string will be the
#       primary device. These tests will be skipped if the device type
#       has fewer available devices than the argument to @DeviceCountAtLeast.
#
#       Tests like (1c) are called with a device string and a torch.dtype from
#       the list of dtypes specified in the @Dtypes decorator. Device-specific
#       dtype overrides can be specified using @DtypesIfCPU and @DtypesIfNPU.
#
#       Tests like (1d) take a devices argument like (1b) and a dtype
#       argument from (1c).
#
#   (2) Prefer using test decorators defined in this file to others.
#       For example, using the @SkipIfNoLapack decorator instead of the
#       @SkipCPUIfNoLapack will cause the test to not run on NPU if
#       LAPACK is not available, which is wrong. If you need to use a decorator
#       you may want to ask about porting it to this framework.
#
#   See the TestTorchDeviceType class in test_torch.py for an example.
#
# [RUNNING TESTS]
#
# After defining your test class call instantiate_device_type_tests on it
# and pass in globals() for the second argument. This will instantiate
# discoverable device-specific test classes from your generic class. It will
# also hide the tests in your generic class so they're not run.
#
# If you device-generic test class is TestClass then new classes with names
# TestClass<DEVICE_TYPE> will be created for each available device type.
# TestClassCPU and TestClassNPU, for example. Tests in these classes also
# have the device type and dtype, if provided, appended to their original
# name. testX, for instance, becomes testX_<device_type> or
# testX_<device_type>_<dtype>.
#
# More concretely, TestTorchDeviceType becomes TestTorchDeviceTypeCPU,
# TestTorchDeviceTypeNPU, ... test_diagonal in TestTorchDeviceType becomes
# test_diagonal_cpu, test_diagonal_npu, ... test_erfinv, which accepts a dtype,
# becomes test_erfinv_cpu_float, test_erfinv_cpu_double, test_erfinv_npu_half,
# ...
#
# In short, if you write a test signature like
#   def textX(self, device)
# You are effectively writing
#   def testX_cpu(self, device='cpu')
#   def textX_npu(self, device='npu')
#   def testX_xla(self, device='xla')
#   ...
#
# These tests can be run directly like normal tests:
# "python test_torch.py TestTorchDeviceTypeCPU.test_diagonal_cpu"
#
# All the tests for a particular device type can be run using the class, and
# other collections of tests can be run using pytest filtering, like
#
# "pytest test_torch.py -k 'test_diag'"
#
# which will run test_diag on every available device.
#
# To specify particular device types the 'and' keyword can be used:
#
# "pytest test_torch.py -k 'test_erfinv and cpu'"
#
# will run test_erfinv on all cpu dtypes.
#
# [ADDING A DEVICE TYPE]
#
# To add a device type:
#
#   (1) Create a new "TestBase" extending DeviceTypeTestBase.
#       See CPUTestBase and NPUTestBase below.
#   (2) Define the "device_type" attribute of the base to be the
#       appropriate string.
#   (3) Add logic to this file that appends your base class to
#       device_type_test_bases when your device type is available.
#   (4) (Optional) Write setUpClass/tearDownClass class methods that
#       instantiate dependencies (see MAGMA in NPUTestBase).
#   (5) (Optional) Override the "instantiate_test" method for total
#       control over how your class creates tests.
#
# setUpClass is called AFTER tests have been created and BEFORE and ONLY IF
# they are run. This makes it useful for initializing devices and dependencies.
#
# Note [Overriding methods in generic tests]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Device generic tests look a lot like normal test classes, but they differ
# from ordinary classes in some important ways.  In particular, overriding
# methods in generic tests doesn't work quite the way you expect.
#
#     class TestFooDeviceType(TestCase):
#         # Intention is to override
#         def assertEqual(self, x, y):
#             # This DOESN'T WORK!
#             super(TestFooDeviceType, self).assertEqual(x, y)
#
# If you try to run this code, you'll get an error saying that TestFooDeviceType
# is not in scope.  This is because after instantiating our classes, we delete
# it from the parent scope.  Instead, you need to hardcode a direct invocation
# of the desired subclass call, e.g.,
#
#     class TestFooDeviceType(TestCase):
#         # Intention is to override
#         def assertEqual(self, x, y):
#             TestCase.assertEqual(x, y)
#
# However, a less error-prone way of customizing the behavior of TestCase
# is to either (1) add your functionality to TestCase and make it toggled
# by a class attribute, or (2) create your own subclass of TestCase, and
# then inherit from it for your generic test.
#

# List of device type test bases that can be used to instantiate tests.
# See below for how this list is populated. If you're adding a device type
# you should check if it's available and (if it is) add it to this list.
device_type_test_bases = []


class DeviceTypeTestBase(TestCase):
    device_type = 'generic_device_type'

    # Precision is a thread-local setting since it may be overridden per test
    _tls = threading.local()
    _tls.precision = TestCase.precision

    @property
    def precision(self):
        return self._tls.precision

    @precision.setter
    def precision(self, prec):
        self._tls.precision = prec

    # Returns a string representing the device that single device tests should use.
    # Note: single device tests use this device exclusively.
    @classmethod
    def get_primary_device(cls):
        return cls.device_type

    # Returns a list of strings representing all available devices of this
    # device type. The primary device must be the first string in the list
    # and the list must contain no duplicates.
    # Note: UNSTABLE API. Will be replaced once PyTorch has a device generic
    #   mechanism of acquiring all available devices.
    @classmethod
    def get_all_devices(cls):
        return [cls.get_primary_device()]

    # Returns the dtypes the test has requested.
    # Prefers device-specific dtype specifications over generic ones.
    @classmethod
    def _get_dtypes(cls, test):
        if not hasattr(test, 'dtypes'):
            return None
        return test.dtypes.get(cls.device_type, test.dtypes.get('all', None))

    @classmethod
    def _get_formats(cls, test):
        if not hasattr(test, 'formats'):
            return None
        return test.formats.get(cls.device_type, test.formats.get('all', None))

    def _get_precision_override(self, test, dtype):
        if not hasattr(test, 'precision_overrides'):
            return self.precision
        return test.precision_overrides.get(dtype, self.precision)

    # Creates device-specific tests.
    @classmethod
    def instantiate_test(cls, name, test):
        test_name = name + "_" + cls.device_type

        dtypes = cls._get_dtypes(test)
        formats = cls._get_formats(test)
        if dtypes is None and formats is None:  # Test has no dtype and npu_format variants
            assert not hasattr(cls, test_name), "Redefinition of test {0}".format(test_name)

            @wraps(test)
            def instantiated_test(self, test=test):
                device_arg = cls.get_primary_device() if not hasattr(test, 'num_required_devices') else cls.get_all_devices()
                return test(self, device_arg)

            setattr(cls, test_name, instantiated_test)

        elif dtypes is None and formats: # Test has npu_format variants
            for npu_format in formats:
                format_str = str(npu_format)
                format_test_name = test_name + "_" + format_str
                assert not hasattr(cls, format_test_name), "Redefinition of test {0}".format(format_test_name)

                @wraps(test)
                def instantiated_test(self, test=test, npu_format=npu_format):
                    device_arg = cls.get_primary_device() if not hasattr(test,
                                                                         'num_required_devices') else cls.get_all_devices()
                    # Sets precision and runs test
                    # Note: precision is reset after the test is run
                    guard_precision = self.precision
                    try:
                        result = test(self, device_arg, npu_format)
                    finally:
                        self.precision = guard_precision

                    return result

                setattr(cls, format_test_name, instantiated_test)

        elif formats and dtypes: # Test has dtype and npu_format variants
            for npu_format in formats:
                for dtype in dtypes:
                    dtype_str = str(dtype).split('.')[1]
                    format_str = str(npu_format)
                    format_dtype_test_name = test_name + "_" + dtype_str + "_" + format_str
                    assert not hasattr(cls, format_dtype_test_name), "Redefinition of test {0}".format(format_dtype_test_name)

                    @wraps(test)
                    def instantiated_test(self, test=test, dtype=dtype, npu_format=npu_format):
                        device_arg = cls.get_primary_device() if not hasattr(test,
                                                                             'num_required_devices') else cls.get_all_devices()
                        # Sets precision and runs test
                        # Note: precision is reset after the test is run
                        guard_precision = self.precision
                        try:
                            self.precision = self._get_precision_override(test, dtype)
                            result = test(self, device_arg, dtype, npu_format)
                        finally:
                            self.precision = guard_precision

                        return result

                    setattr(cls, format_dtype_test_name, instantiated_test)

        elif formats is None and dtypes:  # Test has dtype variants
            for dtype in dtypes:
                dtype_str = str(dtype).split('.')[1]
                dtype_test_name = test_name + "_" + dtype_str
                assert not hasattr(cls, dtype_test_name), "Redefinition of test {0}".format(dtype_test_name)

                @wraps(test)
                def instantiated_test(self, test=test, dtype=dtype):
                    device_arg = cls.get_primary_device() if not hasattr(test, 'num_required_devices') else cls.get_all_devices()
                    # Sets precision and runs test
                    # Note: precision is reset after the test is run
                    guard_precision = self.precision
                    try :
                        self.precision = self._get_precision_override(test, dtype)
                        result = test(self, device_arg, dtype)
                    finally:
                        self.precision = guard_precision

                    return result

                setattr(cls, dtype_test_name, instantiated_test)


class NPUTestBase(DeviceTypeTestBase):
    device_type = 'npu'


class CPUTestBase(DeviceTypeTestBase):
    device_type = 'cpu'


# Adds available device-type-specific test base classes
device_type_test_bases.append(CPUTestBase)
device_type_test_bases.append(NPUTestBase)


# Adds 'instantiated' device-specific test cases to the given scope.
# The tests in these test cases are derived from the generic tests in
# generic_test_class.
# See note "Generic Device Type Testing."
def instantiate_device_type_tests(generic_test_class, scope, except_for=None):
    # Removes the generic test class from its enclosing scope so its tests
    # are not discoverable.
    del scope[generic_test_class.__name__]

    # Creates an 'empty' version of the generic_test_class
    # Note: we don't inherit from the generic_test_class directly because
    #   that would add its tests to our test classes and they would be
    #   discovered (despite not being runnable). Inherited methods also
    #   can't be removed later, and we can't rely on load_tests because
    #   pytest doesn't support it (as of this writing).
    empty_name = generic_test_class.__name__ + "_base"
    empty_class = type(empty_name, generic_test_class.__bases__, {})

    # Acquires members names
    # See Note [Overriding methods in generic tests]
    generic_members = set(generic_test_class.__dict__.keys()) - set(empty_class.__dict__.keys())
    generic_tests = [x for x in generic_members if x.startswith('test')]

    # Creates device-specific test cases
    for base in device_type_test_bases:
        # Skips bases listed in except_for
        if except_for is not None and base.device_type in except_for:
            continue

        class_name = generic_test_class.__name__ + base.device_type.upper()
        device_type_test_class = type(class_name, (base, empty_class), {})

        for name in generic_members:
            if name in generic_tests:  # Instantiates test member
                # Requires tests be a function for Python2 compat
                # (In Python2 tests are type checked methods wrapping functions)
                test = getattr(generic_test_class, name)
                if hasattr(test, '__func__'):
                    test = test.__func__
                assert inspect.isfunction(test), "Couldn't extract function from '{0}'".format(name)

                # Instantiates the device-specific tests
                device_type_test_class.instantiate_test(name, test)
            else:  # Ports non-test member
                assert name not in device_type_test_class.__dict__, "Redefinition of directly defined member {0}".format(name)

                # Unwraps to functions (when available) for Python2 compat
                nontest = getattr(generic_test_class, name)
                if hasattr(nontest, '__func__'):
                    nontest = nontest.__func__

                setattr(device_type_test_class, name, nontest)

        # Mimics defining the instantiated class in the caller's file
        # by setting its module to the given class's and adding
        # the module to the given scope.
        # This lets the instantiated class be discovered by unittest.
        device_type_test_class.__module__ = generic_test_class.__module__
        scope[class_name] = device_type_test_class


# Decorator that skips a test if the given condition is true.
# Notes:
#   (1) Skip conditions stack.
#   (2) Skip conditions can be bools or strings. If a string the
#       test base must have defined the corresponding attribute to be False
#       for the test to run. If you want to use a string argument you should
#       probably define a new decorator instead (see below).
#   (3) Prefer the existing decorators to defining the 'device_type' kwarg.
class SkipIf(object):

    def __init__(self, dep, reason, device_type=None):
        self.dep = dep
        self.reason = reason
        self.device_type = device_type

    def __call__(self, fn):

        @wraps(fn)
        def dep_fn(slf, device, *args, **kwargs):
            if self.device_type is None or self.device_type == slf.device_type:
                if (isinstance(self.dep, str) and getattr(slf, self.dep, True)) or (isinstance(self.dep, bool) and self.dep):
                    raise unittest.SkipTest(self.reason)

            return fn(slf, device, *args, **kwargs)
        return dep_fn


# Skips a test on CPU if the condition is true.
class SkipCPUIf(SkipIf):

    def __init__(self, dep, reason):
        super(SkipCPUIf, self).__init__(dep, reason, device_type='cpu')


class ExpectedFailure(object):

    def __init__(self, device_type):
        self.device_type = device_type

    def __call__(self, fn):

        @wraps(fn)
        def efail_fn(slf, device, *args, **kwargs):
            if self.device_type is None or self.device_type == slf.device_type:
                try:
                    fn(slf, device, *args, **kwargs)
                except Exception:
                    return
                else:
                    slf.fail('expected test to fail, but it passed')

            return fn(slf, device, *args, **kwargs)
        return efail_fn


class OnlyOn(object):

    def __init__(self, device_type):
        self.device_type = device_type

    def __call__(self, fn):

        @wraps(fn)
        def only_fn(slf, device, *args, **kwargs):
            if self.device_type != slf.device_type:
                reason = "Only runs on {0}".format(self.device_type)
                raise unittest.SkipTest(reason)

            return fn(slf, device, *args, **kwargs)

        return only_fn


# Decorator that provides all available devices of the device type to the test
# as a list of strings instead of providing a single device string.
# Skips the test if the number of available devices of the variant's device
# type is less than the 'num_required_devices' arg.
class DeviceCountAtLeast(object):

    def __init__(self, num_required_devices):
        self.num_required_devices = num_required_devices

    def __call__(self, fn):
        assert not hasattr(fn, 'num_required_devices'), "DeviceCountAtLeast redefinition for {0}".format(fn.__name__)
        fn.num_required_devices = self.num_required_devices

        @wraps(fn)
        def multi_fn(slf, devices, *args, **kwargs):
            if len(devices) < self.num_required_devices:
                reason = "fewer than {0} devices detected".format(self.num_required_devices)
                raise unittest.SkipTest(reason)

            return fn(slf, devices, *args, **kwargs)

        return multi_fn


# Specifies per-dtype precision overrides.
# Ex.
#
# @PrecisionOverride(torch.half : 1e-2, torch.float : 1e-4)
# @Dtypes(torch.half, torch.float, torch.double)
# def test_X(self, device, dtype):
#   ...
#
# When the test is instantiated its class's precision will be set to the
# corresponding override, if it exists.
# self.precision can be accessed directly, and it also controls the behavior of
# functions like self.assertEqual().
#
# Note that self.precision is a scalar value, so if you require multiple
# precisions (or are working with multiple dtypes) they should be specified
# explicitly and computed using self.precision (e.g.
# self.precision *2, max(1, self.precision)).
class PrecisionOverride(object):

    def __init__(self, d):
        assert isinstance(d, dict), "PrecisionOverride not given a dtype : precision dict!"
        for dtype, prec in d.items():
            assert isinstance(dtype, torch.dtype), "PrecisionOverride given unknown dtype {0}".format(dtype)

        self.d = d

    def __call__(self, fn):
        fn.precision_overrides = self.d
        return fn


# Decorator that instantiates a variant of the test for each given dtype.
# Notes:
#   (1) Tests that accept the dtype argument MUST use this decorator.
#   (2) Can be overridden for the CPU or NPU, respectively, using DtypesIfCPU
#       or DtypesIfNPU.
#   (3) Prefer the existing decorators to defining the 'device_type' kwarg.
class Dtypes(object):

    # Note: *args, **kwargs for Python2 compat.
    # Python 3 allows (self, *args, device_type='all').
    def __init__(self, *args, **kwargs):
        assert args is not None and len(args) != 0, "No dtypes given"
        assert all(isinstance(arg, torch.dtype) for arg in args), "Unknown dtype in {0}".format(str(args))
        self.args = args
        self.device_type = kwargs.get('device_type', 'all')

    def __call__(self, fn):
        d = getattr(fn, 'dtypes', {})
        assert self.device_type not in d, "dtypes redefinition for {0}".format(self.device_type)
        d[self.device_type] = self.args
        fn.dtypes = d
        return fn

class Formats(object):

    def __init__(self, *args, **kwargs):
        assert args is not None and len(args) != 0, "No formats given"
        self.args = args
        self.device_type = kwargs.get('device_type', 'all')

    def __call__(self, fn):
        d = getattr(fn, 'formats', {})
        assert self.device_type not in d, "formats redefinition for {0}".format(self.device_type)
        d[self.device_type] = self.args
        fn.formats = d
        return fn

# Overrides specified Dtypes on the CPU.
class DtypesIfCPU(Dtypes):

    def __init__(self, *args):
        super(DtypesIfCPU, self).__init__(*args, device_type='cpu')


def only_npu(fn):
    return OnlyOn('npu')(fn)


def only_cpu(fn):
    return OnlyOn('cpu')(fn)


# Skips a test on CPU if LAPACK is not available.
class SkipCPUIfNoLapack(object):

    def __call__(self, fn):
        return SkipCPUIf(not torch._C.has_lapack, "PyTorch compiled without Lapack")(fn)


# Skips a test on CPU if MKL is not available.
class SkipCPUIfNoMkl(object):

    def __call__(fn):
        return SkipCPUIf(not TEST_MKL, "PyTorch is built without MKL support")(fn)

