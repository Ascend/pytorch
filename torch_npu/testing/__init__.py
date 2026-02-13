import json
import os
import re
import unittest
import warnings
from functools import lru_cache
import torch
from torch.testing._internal import common_utils, common_device_type
from torch.testing._internal.opinfo.core import OpInfo
from torch.testing._internal.common_utils import remove_device_and_dtype_suffixes, TEST_WITH_SLOW, \
    IS_SANDCASTLE, TEST_SKIP_FAST, RERUN_DISABLED_TESTS, DISABLED_TESTS_FILE, SLOW_TESTS_FILE, maybe_load_json, \
    TEST_MPS, IS_FBCODE
from torch.testing._internal.common_dtype import floating_and_complex_types_and
import torch_npu
from torch_npu.testing._npu_testing_utils import update_skip_list, get_decorators

__all__ = []


@lru_cache(maxsize=1)
def _load_npu_opinfo_dtypes():
    try:
        from .npu_opinfo_dtypes import NPU_OPINFO_DTYPES
        return dict(NPU_OPINFO_DTYPES)
    except ModuleNotFoundError as e:
        # Only downgrade when the config module itself is missing.
        # Other failures (syntax errors, import-time errors, bad data, etc.)
        # should be surfaced in tests instead of being silently ignored.
        warnings.warn(f"npu_opinfo_dtypes config module not found: {e}")
        return {}


def _dtype_from_name(name):
    try:
        dtype = getattr(torch, name)
    except AttributeError:
        return None
    return dtype if isinstance(dtype, torch.dtype) else None


def _merge_dtypes(dtypes, extra):
    if not extra:
        return dtypes
    if isinstance(dtypes, set):
        dtypes = set(dtypes)
        dtypes.update(extra)
        return dtypes
    if isinstance(dtypes, tuple):
        dtypes = list(dtypes)
        for dtype in extra:
            if dtype not in dtypes:
                dtypes.append(dtype)
        return tuple(dtypes)
    if isinstance(dtypes, list):
        dtypes = list(dtypes)
        for dtype in extra:
            if dtype not in dtypes:
                dtypes.append(dtype)
        return dtypes
    try:
        dtypes = set(dtypes)
        dtypes.update(extra)
        return dtypes
    except TypeError:
        return set(extra)


def _get_tests_dict():

    def _filter_json(data):
        if _is_910A():
            return {key: val for key, val in data.items() if len(val) > 1 and not (val[1] and "A2" in val[1])}
        return {key: val for key, val in data.items() if len(val) > 1 and not (val[1] and "910A" in val[1])}

    def _is_910A():
        device_name = torch_npu.npu.get_device_name(0)
        if "Ascend910A" in device_name or "Ascend910P" in device_name:
            return True
        return False

    def _load_disabled_json(filename):
        if os.path.isfile(filename):
            with open(filename) as fp0:
                disabled_dict = json.load(fp0, object_hook=_filter_json)
                return disabled_dict
        warnings.warn(f"Attempted to load json file {filename} but it does not exist.")
        return {}

    # import test files
    disabled_tests_dict = {}
    slow_tests_dict = {}
    # set them here in case the tests are running in a subprocess that doesn't call run_tests
    if os.getenv("SLOW_TESTS_FILE", ""):
        slow_tests_dict = maybe_load_json(os.getenv("SLOW_TESTS_FILE", ""))
    if os.getenv("DISABLED_TESTS_FILE", ""):
        disabled_tests_dict = _load_disabled_json(os.getenv("DISABLED_TESTS_FILE", ""))
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
            disabled_tests_dict = _load_disabled_json(DISABLED_TESTS_FILE)
            os.environ['DISABLED_TESTS_FILE'] = DISABLED_TESTS_FILE
        else:
            warnings.warn(f'disabled test file provided but not found: {DISABLED_TESTS_FILE}')

    return disabled_tests_dict, slow_tests_dict


def _check_if_enable_npu(test: unittest.TestCase):
    disabled_tests_dict, slow_tests_dict = _get_tests_dict()

    classname = str(test.__class__).split("'")[1].split(".")[-1]
    sanitized_testname = remove_device_and_dtype_suffixes(test._testMethodName)

    def matches_test(target: str):
        target_test_parts = re.split(" (?=\\(__main__)", target) if "__main__" in target else target.split()
        if len(target_test_parts) < 2:
            # poorly formed target test name
            return False
        target_testname = target_test_parts[0]
        target_classname = target_test_parts[1][1:-1].split(".")[-1]

        class_device_replace = target_classname
        if "PRIVATEUSE1" in target_classname:
            class_device_replace = target_classname.replace("PRIVATEUSE1", "NPU")
        elif "NPU" in target_classname:
            class_device_replace = target_classname.replace("NPU", "PRIVATEUSE1")

        # if test method name or its sanitized version exactly matches the disabled
        # test method name AND allow non-parametrized suite names to disable
        # parametrized ones (TestSuite disables TestSuiteCPU)
        return (classname.startswith(target_classname) or classname.startswith(class_device_replace)) \
               and (target_testname in (test._testMethodName, sanitized_testname))

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
        if hasattr(test, test._testMethodName) and not getattr(test, test._testMethodName).__dict__.get('slow_test',
                                                                                                        False):
            raise unittest.SkipTest("test is fast; we disabled it with PYTORCH_TEST_SKIP_FAST")


def _supported_dtypes(self, device_type):
    dtypes = self.dtypes
    if device_type in ("privateuse1", "npu"):
        cfg = _load_npu_opinfo_dtypes()
        op_cfg = cfg.get(self.name, {})
        forward_cfg = op_cfg.get("forward", {})
        extra_names = forward_cfg.get("extra", [])
        extra = []
        for name in extra_names:
            dtype = _dtype_from_name(name)
            if dtype is None:
                warnings.warn(f"Unknown dtype '{name}' for op {self.name}")
                continue
            extra.append(dtype)
        dtypes = _merge_dtypes(dtypes, extra)
    return dtypes


def _supported_backward_dtypes(self, device_type):
    if not self.supports_autograd:
        return set()

    backward_dtypes = self.backward_dtypes

    allowed_backward_dtypes = floating_and_complex_types_and(
        torch.bfloat16, torch.float16, torch.complex32
    )
    return set(allowed_backward_dtypes).intersection(backward_dtypes)


def _test_for_npu():
    os.environ['PYTORCH_TESTING_DEVICE_FOR_CUSTOM'] = 'privateuse1'
    os.environ['PYTORCH_TESTING_DEVICE_EXCEPT_FOR'] = 'cuda,cpu'
    common_device_type.onlyCUDA = common_device_type.onlyPRIVATEUSE1
    common_utils.TEST_CUDA = common_utils.TEST_PRIVATEUSE1


def _apply_test_patchs():
    update_skip_list()
    OpInfo.get_decorators = get_decorators
    OpInfo.supported_dtypes = _supported_dtypes
    OpInfo.supported_backward_dtypes = _supported_backward_dtypes
    common_utils.check_if_enable = _check_if_enable_npu


# apply test_ops related patch
_apply_test_patchs()
_test_for_npu()
