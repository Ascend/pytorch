#!/usr/bin/env python

from __future__ import print_function

import argparse
from datetime import datetime
import modulefinder
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import torch
import torch_npu
import torch._six
from torch.utils import cpp_extension
from torch.testing._internal.common_utils import TEST_WITH_ROCM, shell
import torch.distributed as dist
PY2 = sys.version_info <= (3,)
PY33 = sys.version_info >= (3, 3)
PY36 = sys.version_info >= (3, 6)

TESTS = [
    'test_distributed'
]

# skip < 3.3 because mock is added in 3.3 and is used in rpc_spawn
# skip python2 for rpc and dist_autograd tests that do not support python2
if PY33:
    TESTS.extend([
        'distributed/rpc/test_rpc_spawn',
        'distributed/rpc/test_dist_autograd_spawn',
        'distributed/rpc/test_dist_optimizer_spawn',
        'distributed/rpc/jit/test_dist_autograd_spawn',
    ])

# skip < 3.6 b/c fstrings added in 3.6
if PY36:
    TESTS.extend([
        'test_jit_py3',
        'test_determination',
        'distributed/rpc/jit/test_rpc_spawn',
    ])

WINDOWS_BLACKLIST = [
    'distributed/test_distributed',
    'distributed/rpc/test_rpc_spawn',
    'distributed/rpc/test_dist_autograd_spawn',
    'distributed/rpc/test_dist_optimizer_spawn',
    'distributed/rpc/jit/test_rpc_spawn',
    'distributed/rpc/jit/test_dist_autograd_spawn',
]

ROCM_BLACKLIST = [
    'test_cpp_extensions_aot_ninja',
    'test_cpp_extensions_jit',
    'test_multiprocessing',
    'distributed/rpc/test_rpc_spawn',
    'distributed/rpc/test_dist_autograd_spawn',
    'distributed/rpc/test_dist_optimizer_spawn',
    'distributed/rpc/jit/test_rpc_spawn',
    'distributed/rpc/jit/test_dist_autograd_spawn',
    'test_determination',
]

DISTRIBUTED_TESTS_CONFIG = {}


if dist.is_available():
    if dist.is_hccl_available():
        DISTRIBUTED_TESTS_CONFIG['hccl'] = {
            'WORLD_SIZE': '2' if torch.npu.device_count() == 2 else '4',
            'TEST_REPORT_SOURCE_OVERRIDE': 'dist-hccl'
        }
    else:
        if not TEST_WITH_ROCM and dist.is_mpi_available():
            DISTRIBUTED_TESTS_CONFIG['mpi'] = {
                'WORLD_SIZE': '3',
                'TEST_REPORT_SOURCE_OVERRIDE': 'dist-mpi'
            }
        if dist.is_nccl_available():
            DISTRIBUTED_TESTS_CONFIG['nccl'] = {
                'WORLD_SIZE': '2' if torch.cuda.device_count() == 2 else '3',
                'TEST_REPORT_SOURCE_OVERRIDE': 'dist-nccl'
            }
        if not TEST_WITH_ROCM and dist.is_gloo_available():
            DISTRIBUTED_TESTS_CONFIG['gloo'] = {
                'WORLD_SIZE': '2' if torch.cuda.device_count() == 2 else '3',
                'TEST_REPORT_SOURCE_OVERRIDE': 'dist-gloo'
            }

# https://stackoverflow.com/questions/2549939/get-signal-names-from-numbers-in-python
SIGNALS_TO_NAMES_DICT = {getattr(signal, n): n for n in dir(signal)
                         if n.startswith('SIG') and '_' not in n}


def print_to_stderr(message):
    print(message, file=sys.stderr)


def run_test(executable, test_module, test_directory, options, *extra_unittest_args):
    unittest_args = options.additional_unittest_args
    if options.verbose:
        unittest_args.append('--verbose')
    # Can't call `python -m unittest test_*` here because it doesn't run code
    # in `if __name__ == '__main__': `. So call `python test_*.py` instead.
    argv = [test_module + '.py'] + unittest_args + list(extra_unittest_args)

    command = executable + argv
    return shell(command, test_directory)

def test_distributed_npu(executable, test_module, test_directory, options):
    config = DISTRIBUTED_TESTS_CONFIG
    for backend, env_vars in config.items():
        for with_init_file in {True, False}:
            tmp_dir = tempfile.mkdtemp()
            if options.verbose:
                with_init = ' with file init_method' if with_init_file else ''
                print_to_stderr(
                    'Running distributed tests for the {} backend{}'.format(
                        backend, with_init))
            os.environ['TEMP_DIR'] = tmp_dir
            os.environ['BACKEND'] = backend
            os.environ['INIT_METHOD'] = 'env://'
            os.environ.update(env_vars)
            if with_init_file:
                init_method = 'file://{}/shared_init_file'.format(tmp_dir)
                os.environ['INIT_METHOD'] = init_method
            try:
                os.mkdir(os.path.join(tmp_dir, 'barrier'))
                os.mkdir(os.path.join(tmp_dir, 'test_dir'))
                return_code = run_test(executable, test_module, test_directory,
                                       options)
                if return_code != 0:
                    return return_code
            finally:
                shutil.rmtree(tmp_dir)
    return 0

CUSTOM_HANDLERS = {
    'test_distributed': test_distributed_npu
}


def parse_test_module(test):
    return test.split('.')[0]


class TestChoices(list):
    def __init__(self, *args, **kwargs):
        super(TestChoices, self).__init__(args[0])

    def __contains__(self, item):
        return list.__contains__(self, parse_test_module(item))
    
FAILURE_FILE_NAME = 'pytorch_org_failures.txt'
ERROR_FILE_NAME = 'pytorch_org_errors.txt'
def htmlReport_load_failure_error_cases(file_name):
    data = []
    if os.path.isfile(file_name):
        with open(file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                temp = line.strip('\n').strip('\t')
                data.append(temp)
    else:
        print("Invlid filename:",file_name)
    return data

def htmlReport_analyse_failure_error_cases(result):
    new_failures = []
    new_errors = []

    if len(result.failures) > 0:
        print("====================================== failed cases count: ", len(result.failures))
        for failure in result.failures:
            print(failure[0])
        print("============================================================\n")
        orig_failures = htmlReport_load_failure_error_cases(FAILURE_FILE_NAME)
        for failure in result.failures:
            if str(failure[0]) not in orig_failures:
                new_failures.append(str(failure[0]))

    if len(result.errors) > 0:
        print("====================================== error cases count: ", len(result.errors))
        for error_case in result.errors:
            print(error_case[0])
        print("============================================================\n")
        orig_errors = htmlReport_load_failure_error_cases(ERROR_FILE_NAME)
        for error_case in result.errors:
            if str(error_case[0]) not in orig_errors:
                new_errors.append(str(error_case[0]))
    print("====================================== new failed cases count: ", len(new_failures))
    for case in new_failures:
        print(case)
    print("====================================== new error cases count: ", len(new_errors))
    for case in new_errors:
        print(case)
    return new_failures, new_errors

def htmlReport_RunTests(suite):

    ENABLE_HTML = bool(os.environ.get('ENABLE_HTML'))
    ENABLE_HTML_MX = bool(os.environ.get('ENABLE_HTML_MX'))
    ENABLE_CASE_PATH = os.environ.get('ENABLE_CASE_PATH')
    ENABLE_OUTPUT_PATH = os.environ.get('ENABLE_OUTPUT_PATH')
    WHITE_LIST_PATH = os.environ.get('WHITE_LIST_PATH')

    test_case_path = './'
    if ENABLE_CASE_PATH is not None:
        if not os.path.exists(ENABLE_CASE_PATH):
            print('path is not exists: ', ENABLE_CASE_PATH)
        else:
            test_case_path = ENABLE_CASE_PATH

    test_report_path = test_case_path+'ReportResult'

    if ENABLE_OUTPUT_PATH is not None:
        if not os.path.exists(ENABLE_OUTPUT_PATH):
            print('path is not exists: ', ENABLE_OUTPUT_PATH)
        else:
            test_report_path = ENABLE_OUTPUT_PATH

    if not os.path.exists(test_report_path):
        os.mkdir(test_report_path)
        print(test_report_path)

    now = time.strftime("%Y_%m_%d_%H_%M_%S")
    htmlFileName = os.path.join(test_report_path, 'pytorch-unittest-report-'+now+'.html')
    txtFileName = os.path.join(test_report_path, 'pytorch-unittest-report-'+now+'.txt')

    print('start pytorch HTML unittest testset...')
    import HTMLTestRunner
    with os.fdopen(os.open(htmlFileName, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=0o600), "wb") as report_file:
        runner = HTMLTestRunner.HTMLTestRunner(
            stream=report_file, title='AllTest', description='all npu test case', verbosity=2)
        result = runner.run(suite)
        new_failures, new_errors = htmlReport_analyse_failure_error_cases(result)
        if len(new_failures) + len(new_errors) > 0:
            print(" RuntimeError: new error or failed cases found!")
    print('report files path', htmlFileName)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the PyTorch unit test suite',
        epilog='where TESTS is any of: {}'.format(', '.join(TESTS)))
    parser.add_argument(
        '--error-continue',
        action='store_true',
        help='run test continue when error or failure.')
    parser.add_argument(
        '--html-test-runner',
        action='store_true',
        help='run test case by HTML Test Runner.')
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='print verbose information and test-by-test results')
    parser.add_argument(
        '--jit',
        '--jit',
        action='store_true',
        help='run all jit tests')
    parser.add_argument(
        '-pt', '--pytest', action='store_true',
        help='If true, use `pytest` to execute the tests. E.g., this runs '
             'TestTorch with pytest in verbose and coverage mode: '
             'python run_test.py -vci torch -pt')
    parser.add_argument(
        '-c', '--coverage', action='store_true', help='enable coverage')
    parser.add_argument(
        '-i',
        '--include',
        nargs='+',
        choices=TestChoices(TESTS),
        default=TESTS,
        metavar='TESTS',
        help='select a set of tests to include (defaults to ALL tests).'
             ' tests can be specified with module name, module.TestClass'
             ' or module.TestClass.test_method')
    parser.add_argument(
        '-x',
        '--exclude',
        nargs='+',
        choices=TESTS,
        metavar='TESTS',
        default=[],
        help='select a set of tests to exclude')
    parser.add_argument(
        '-f',
        '--first',
        choices=TESTS,
        metavar='TESTS',
        help='select the test to start from (excludes previous tests)')
    parser.add_argument(
        '-l',
        '--last',
        choices=TESTS,
        metavar='TESTS',
        help='select the last test to run (excludes following tests)')
    parser.add_argument(
        '--bring-to-front',
        nargs='+',
        choices=TestChoices(TESTS),
        default=[],
        metavar='TESTS',
        help='select a set of tests to run first. This can be used in situations'
             ' where you want to run all tests, but care more about some set, '
             'e.g. after making a change to a specific component')
    parser.add_argument(
        '--ignore-win-blacklist',
        action='store_true',
        help='always run blacklisted windows tests')
    parser.add_argument(
        '--determine-from',
        help='File of affected source filenames to determine which tests to run.')
    parser.add_argument(
        'additional_unittest_args',
        nargs='*',
        help='additional arguments passed through to unittest, e.g., '
             'python run_test.py -i sparse -- TestSparse.test_factory_size_check')
    return parser.parse_args()


def get_executable_command(options):
    if options.coverage:
        executable = ['coverage', 'run', '--parallel-mode', '--source torch']
    else:
        executable = [sys.executable]
    if options.pytest:
        executable += ['-m', 'pytest']
    return executable


def find_test_index(test, selected_tests, find_last_index=False):
    """Find the index of the first or last occurrence of a given test/test module in the list of selected tests.

    This function is used to determine the indices when slicing the list of selected tests when
    ``options.first``(:attr:`find_last_index`=False) and/or ``options.last``(:attr:`find_last_index`=True) are used.

    :attr:`selected_tests` can be a list that contains multiple consequent occurrences of tests
    as part of the same test module, e.g.:

    ```
    selected_tests = ['autograd', 'cuda', **'torch.TestTorch.test_acos',
                     'torch.TestTorch.test_tan', 'torch.TestTorch.test_add'**, 'utils']
    ```

    If :attr:`test`='torch' and :attr:`find_last_index`=False, result should be **2**.
    If :attr:`test`='torch' and :attr:`find_last_index`=True, result should be **4**.

    Arguments:
        test (str): Name of test to lookup
        selected_tests (list): List of tests
        find_last_index (bool, optional): should we lookup the index of first or last
            occurrence (first is default)

    Returns:
        index of the first or last occurrence of the given test
    """
    idx = 0
    found_idx = -1
    for t in selected_tests:
        if t.startswith(test):
            found_idx = idx
            if not find_last_index:
                break
        idx += 1
    return found_idx


def exclude_tests(exclude_list, selected_tests, exclude_message=None):
    for exclude_test in exclude_list:
        tests_copy = selected_tests[:]
        for test in tests_copy:
            if test.startswith(exclude_test):
                if exclude_message is not None:
                    print_to_stderr('Excluding {} {}'.format(test, exclude_message))
                selected_tests.remove(test)
    return selected_tests


def get_selected_tests(options):
    selected_tests = options.include

    if options.bring_to_front:
        to_front = set(options.bring_to_front)
        selected_tests = options.bring_to_front + list(filter(lambda name: name not in to_front,
                                                              selected_tests))

    if options.first:
        first_index = find_test_index(options.first, selected_tests)
        selected_tests = selected_tests[first_index:]

    if options.last:
        last_index = find_test_index(options.last, selected_tests, find_last_index=True)
        selected_tests = selected_tests[:last_index + 1]

    selected_tests = exclude_tests(options.exclude, selected_tests)

    if sys.platform == 'win32' and not options.ignore_win_blacklist:
        target_arch = os.environ.get('VSCMD_ARG_TGT_ARCH')
        if target_arch != 'x64':
            WINDOWS_BLACKLIST.append('cpp_extensions_aot_no_ninja')
            WINDOWS_BLACKLIST.append('cpp_extensions_aot_ninja')
            WINDOWS_BLACKLIST.append('cpp_extensions_jit')
            WINDOWS_BLACKLIST.append('jit')
            WINDOWS_BLACKLIST.append('jit_fuser')

        selected_tests = exclude_tests(WINDOWS_BLACKLIST, selected_tests, 'on Windows')

    elif TEST_WITH_ROCM:
        selected_tests = exclude_tests(ROCM_BLACKLIST, selected_tests, 'on ROCm')

    return selected_tests

def main():
    options = parse_args()
    executable = get_executable_command(options)  # this is a list
    print_to_stderr('Test executor: {}'.format(executable))
    test_directory = os.path.dirname(os.path.abspath(__file__))
    selected_tests = get_selected_tests(options)

    if options.verbose:
        print_to_stderr('Selected tests: {}'.format(', '.join(selected_tests)))

    if options.coverage:
        shell(['coverage', 'erase'])

    if options.jit:
        selected_tests = filter(lambda test_name: "jit" in test_name, TESTS)

    if options.determine_from is not None and os.path.exists(options.determine_from):
        pass
     
    htmlReport_suite = unittest.TestSuite()
    htmlReport_loader = unittest.TestLoader()

    for test in selected_tests:

        test_module = parse_test_module(test)

        # Printing the date here can help diagnose which tests are slow
        print_to_stderr('Running {} ... [{}]'.format(test, datetime.now()))
        handler = CUSTOM_HANDLERS.get(test, run_test)
        if options.html_test_runner:
            testfileName = test_module + '.py'
            testCase = unittest.defaultTestLoader.discover("./", pattern=testfileName)
            
            rtn = htmlReport_suite.addTest(testCase)
        else:
            return_code = handler(executable, test_module, test_directory, options)
            assert isinstance(return_code, int) and not isinstance(
                return_code, bool), 'Return code should be an integer'
            if return_code != 0:
                message = '{} failed!'.format(test)
                if return_code < 0:
                    # subprocess.Popen returns the child process' exit signal as
                    # return code -N, where N is the signal number.
                    signal_name = SIGNALS_TO_NAMES_DICT[-return_code]
                    message += ' Received signal: {}'.format(signal_name)
                if not options.error_continue:
                    raise RuntimeError(message)
    if options.html_test_runner:
        htmlReport_RunTests(htmlReport_suite)
    if options.coverage:
        shell(['coverage', 'combine'])
        shell(['coverage', 'html'])


if __name__ == '__main__':
    main()
