import json
import os
import unittest
import warnings
import torch
from torch.testing._internal import common_utils
from torch.testing._internal.opinfo.core import OpInfo
from torch.testing._internal.common_utils import remove_device_and_dtype_suffixes, TEST_WITH_SLOW, \
        IS_SANDCASTLE, TEST_SKIP_FAST, RERUN_DISABLED_TESTS, DISABLED_TESTS_FILE, SLOW_TESTS_FILE, maybe_load_json
from torch.testing._internal.common_dtype import floating_and_complex_types_and
from torch_npu.testing.npu_testing_utils import update_skip_list, get_decorators

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


def check_if_enable_npu(test: unittest.TestCase):
    classname = str(test.__class__).split("'")[1].split(".")[-1]
    sanitized_testname = remove_device_and_dtype_suffixes(test._testMethodName)

    def matches_test(target: str):
        target_test_parts = target.split()
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


def _supported_dtypes(self, device_type):
    return self.dtypes


def _supported_backward_dtypes(self, device_type):
    if not self.supports_autograd:
        return set()

    backward_dtypes = self.backward_dtypes

    allowed_backward_dtypes = floating_and_complex_types_and(
        torch.bfloat16, torch.float16, torch.complex32
    )
    return set(allowed_backward_dtypes).intersection(backward_dtypes)


def apply_test_patchs():
    update_skip_list()
    OpInfo.get_decorators = get_decorators
    OpInfo.supported_dtypes = _supported_dtypes
    OpInfo.supported_backward_dtypes = _supported_backward_dtypes
    common_utils.check_if_enable = check_if_enable_npu

#apply test_ops related patch
apply_test_patchs()
