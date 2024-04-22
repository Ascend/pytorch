import json
import copy
import os
import re
import sys
import unittest
import warnings
import inspect
from typing import Any
from torch.testing._internal import common_utils, common_device_type
from torch.testing._internal.opinfo.core import OpInfo
from torch.testing._internal.common_utils import remove_device_and_dtype_suffixes, TEST_WITH_SLOW, \
        IS_SANDCASTLE, TEST_SKIP_FAST, RERUN_DISABLED_TESTS, DISABLED_TESTS_FILE, SLOW_TESTS_FILE, maybe_load_json, \
        TEST_MPS, IS_FBCODE
from torch.testing._internal.common_device_type import device_type_test_bases, MPSTestBase, \
    filter_desired_device_types, LazyTestBase, PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, \
    PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY
from torch_npu.testing._npu_testing_utils import update_skip_list, get_decorators

__all__ = []


# import test files
disabled_tests_dict = {}
slow_tests_dict = {}
# set them here in case the tests are running in a subprocess that doesn't call run_tests
if os.getenv("SLOW_TESTS_FILE", ""):
    slow_tests_dict = maybe_load_json(os.getenv("SLOW_TESTS_FILE", ""))
if os.getenv("DISABLED_TESTS_FILE", ""):
    disabled_tests_dict = maybe_load_json(os.getenv("DISABLED_TESTS_FILE", ""))
if SLOW_TESTS_FILE:
    if os.path.exists(SLOW_TESTS_FILE):
        with open(SLOW_TESTS_FILE) as fp:
            slow_tests_dict = json.load(fp)
            # use env vars so pytest-xdist subprocesses can still access them
            os.environ['SLOW_TESTS_FILE'] = SLOW_TESTS_FILE
    else:
        warnings.warn(f'slow test file provided but not found: {SLOW_TESTS_FILE}')
if DISABLED_TESTS_FILE:
    if os.path.exists(DISABLED_TESTS_FILE):
        with open(DISABLED_TESTS_FILE) as fp:
            disabled_tests_dict = json.load(fp)
            os.environ['DISABLED_TESTS_FILE'] = DISABLED_TESTS_FILE
    else:
        warnings.warn(f'disabled test file provided but not found: {DISABLED_TESTS_FILE}')


def _check_if_enable_npu(test: unittest.TestCase):
    classname = str(test.__class__).split("'")[1].split(".")[-1]
    sanitized_testname = remove_device_and_dtype_suffixes(test._testMethodName)

    def matches_test(target: str):
        target_test_parts = re.split(" (?=\\(__main__)", target) if "__main__" in target else target.split()
        if len(target_test_parts) < 2:
            # poorly formed target test name
            return False
        target_testname = target_test_parts[0]
        target_classname = target_test_parts[1][1:-1].split(".")[-1]

        if "_npu" in test._testMethodName:
            testname_device_replace = test._testMethodName.replace("_npu", "_privateuse1")
        elif "_privateuse1" in test._testMethodName:
            testname_device_replace = test._testMethodName.replace("_privateuse1", "_npu")
        else:
            testname_device_replace = test._testMethodName

        # if test method name or its sanitized version exactly matches the disabled
        # test method name AND allow non-parametrized suite names to disable
        # parametrized ones (TestSuite disables TestSuiteCPU)
        return classname.startswith(target_classname) \
                and (target_testname in (test._testMethodName, sanitized_testname, testname_device_replace))

    if any(matches_test(x) for x in slow_tests_dict.keys()):
        getattr(test, test._testMethodName).__dict__['slow_test'] = True
        if not TEST_WITH_SLOW:
            raise unittest.SkipTest("test is slow; run with PYTORCH_TEST_WITH_SLOW to enable test")

    if not IS_SANDCASTLE:
        should_skip = False
        skip_msg = ""

        for disabled_test, _ in disabled_tests_dict.items():
            if matches_test(disabled_test):
                should_skip = True
                skip_msg = "this test is disabled now"
                break

        if should_skip and not RERUN_DISABLED_TESTS:
            # Skip the disabled test when not running under --rerun-disabled-tests verification mode
            raise unittest.SkipTest(skip_msg)

        if not should_skip and RERUN_DISABLED_TESTS:
            skip_msg = "Test is enabled but --rerun-disabled-tests verification mode is set, so only" \
                " disabled tests are run"
            raise unittest.SkipTest(skip_msg)

    if TEST_SKIP_FAST:
        if hasattr(test, test._testMethodName) and not getattr(test, test._testMethodName).__dict__.get('slow_test', False):
            raise unittest.SkipTest("test is fast; we disabled it with PYTORCH_TEST_SKIP_FAST")


def _instantiate_device_type_tests(generic_test_class, scope, except_for=None, only_for=None, include_lazy=False, allow_mps=False):
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

    # allow callers to specifically opt tests into being tested on MPS, similar to `include_lazy`
    test_bases = device_type_test_bases.copy()
    if allow_mps and TEST_MPS and MPSTestBase not in test_bases:
        test_bases.append(MPSTestBase)
    desired_device_type_test_bases = filter_desired_device_types(test_bases, except_for, only_for)
    if include_lazy:
        # Note [Lazy Tensor tests in device agnostic testing]
        # Right now, test_view_ops.py runs with LazyTensor.
        # We don't want to opt every device-agnostic test into using the lazy device,
        # because many of them will fail.
        # So instead, the only way to opt a specific device-agnostic test file into
        # lazy tensor testing is with include_lazy=True
        if IS_FBCODE:
            print("TorchScript backend not yet supported in FBCODE/OVRSOURCE builds", file=sys.stderr)
        else:
            desired_device_type_test_bases.append(LazyTestBase)

    def split_if_not_empty(x: str):
        return x.split(",") if len(x) != 0 else []

    # Filter out the device types based on environment variables if available
    # Usage:
    # export PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda,cpu
    # export PYTORCH_TESTING_DEVICE_EXCEPT_FOR=xla
    env_only_for = split_if_not_empty(os.getenv(PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, ''))
    env_except_for = split_if_not_empty(os.getenv(PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY, ''))
    if env_only_for:
        desired_device_type_test_bases += filter(lambda x: x.device_type in env_only_for, test_bases)
        desired_device_type_test_bases = list(set(desired_device_type_test_bases))
    desired_device_type_test_bases = filter_desired_device_types(desired_device_type_test_bases,
                                                                 env_except_for, env_only_for)

    # Creates device-specific test cases
    for base in desired_device_type_test_bases:
        class_name = generic_test_class.__name__ + base.device_type.upper()

        device_type_test_class: Any = type(class_name, (base, empty_class), {})

        for name in generic_members:
            if name in generic_tests:  # Instantiates test member
                test = getattr(generic_test_class, name)
                # XLA-compat shim (XLA's instantiate_test takes doesn't take generic_cls)
                sig = inspect.signature(device_type_test_class.instantiate_test)
                if len(sig.parameters) == 3:
                    # Instantiates the device-specific tests
                    device_type_test_class.instantiate_test(name, copy.deepcopy(test), generic_cls=generic_test_class)
                else:
                    device_type_test_class.instantiate_test(name, copy.deepcopy(test))
            else:  # Ports non-test member
                assert name not in device_type_test_class.__dict__, f"Redefinition of directly defined member {name}"
                nontest = getattr(generic_test_class, name)
                setattr(device_type_test_class, name, nontest)

        # Mimics defining the instantiated class in the caller's file
        # by setting its module to the given class's and adding
        # the module to the given scope.
        # This lets the instantiated class be discovered by unittest.
        device_type_test_class.__module__ = generic_test_class.__module__
        scope[class_name] = device_type_test_class


def _test_for_npu():
    os.environ['PYTORCH_TESTING_DEVICE_ONLY_FOR'] = 'privateuse1'
    os.environ['PYTORCH_TESTING_DEVICE_EXCEPT_FOR'] = 'cuda,cpu'
    common_device_type.onlyCUDA = common_device_type.onlyPRIVATEUSE1
    common_utils.TEST_CUDA = common_utils.TEST_PRIVATEUSE1
    common_device_type.instantiate_device_type_tests = _instantiate_device_type_tests


def _apply_test_patchs():
    update_skip_list()
    OpInfo.get_decorators = get_decorators
    common_utils.check_if_enable = _check_if_enable_npu


#apply test_ops related patch
_apply_test_patchs()
_test_for_npu()
