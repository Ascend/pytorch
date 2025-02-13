import argparse
import pathlib
import os
import sys
import signal
import tempfile
import shutil
import math
from datetime import datetime, timezone
from typing import Optional, List

import torch
from torch.utils import cpp_extension
from torch.testing._internal.common_utils import shell

import torch_npu
from torch_npu.utils._path_manager import PathManager

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

CPP_EXTENSIONS_ERROR = """
Ninja is required for some of the C++ extensions
tests, but it could not be found. Install ninja with `pip install ninja`
or `conda install ninja`.
"""

SIGNALS_TO_NAMES_DICT = {
    getattr(signal, n): n for n in dir(signal) if n.startswith("SIG") and "_" not in n
}


def print_to_stderr(message):
    print(message, file=sys.stderr)


def discover_tests(
        base_dir: Optional[pathlib.Path] = None,
        blocklisted_patterns: Optional[List[str]] = None,
        blocklisted_tests: Optional[List[str]] = None,
        extra_tests: Optional[List[str]] = None) -> List[str]:
    """
    Searches for all python files starting with test_ excluding one specified by patterns
    """
    def skip_test_p(name: str) -> bool:
        rc = False
        if blocklisted_patterns is not None:
            rc |= any(name.startswith(pattern) for pattern in blocklisted_patterns)
        if blocklisted_tests is not None:
            rc |= name in blocklisted_tests
        return rc

    cwd = pathlib.Path(__file__).resolve().parent if base_dir is None else base_dir
    all_py_files = list(cwd.glob('**/test_*.py'))
    rc = [str(fname.relative_to(cwd))[:-3] for fname in all_py_files]
    rc = [test for test in rc if not skip_test_p(test)]
    if extra_tests is not None:
        rc += extra_tests
    return sorted(rc)


def parse_test_module(test):
    return pathlib.Path(test).parts[0]


TESTS = discover_tests(
    blocklisted_patterns=[],
    blocklisted_tests=[],
    extra_tests=[]
)

TESTS_MODULE = list(set([parse_test_module(test) for test in TESTS]))

TEST_CHOICES = TESTS + TESTS_MODULE

CORE_TEST_LIST = [
    "npu",
]


DISTRIBUTED_TESTS_CONFIG = {}


if torch.distributed.is_available():
    if torch.distributed.is_hccl_available():
        DISTRIBUTED_TESTS_CONFIG['hccl'] = {
            'WORLD_SIZE': str(2**math.floor(math.log2(torch.npu.device_count()))),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the PyTorch unit test suite",
        epilog="where TESTS is any of: {}".format(", ".join(TESTS)),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="print verbose information and test-by-test results",
    )
    parser.add_argument(
        "-core",
        "--core",
        action="store_true",
        help="Only run core tests, or tests that validate PyTorch's ops, modules,"
        "and autograd. They are defined by CORE_TEST_LIST."
    )
    parser.add_argument(
        "-i",
        "--include",
        nargs="+",
        choices=TEST_CHOICES,
        default=TESTS,
        metavar="TESTS",
        help="select a set of tests to include (defaults to ALL tests)."
        " tests must be a part of the TESTS list defined in run_test.py",
    )
    parser.add_argument(
        "-e",
        "--exlude",
        nargs="+",
        choices=TEST_CHOICES,
        default=[],
        metavar="TESTS",
        help="select a set of tests to exclude",
    )
    parser.add_argument(
        "-f",
        "--first",
        choices=TEST_CHOICES,
        metavar="TESTS",
        help="select the test to start from (excludes previous tests)",
    )
    parser.add_argument(
        "-l",
        "--last",
        choices=TEST_CHOICES,
        metavar="TESTS",
        help="select the last test to run (excludes following tests)",
    )
    parser.add_argument(
        "additional_unittest_args",
        nargs="*",
        help="additional arguments passed through to unittest, e.g., "
        "python run_test.py -i sparse -- TestSparse.test_factory_size_check",
    )
    parser.add_argument(
        "--init_method",
        default=0,
        type=int,
        help="when specifying the init_method, 1 indicates the use of the \"env\" approach, "
        "2 indicates the use of the \"shared file\" approach, and 0 indicates running both approaches",
    )
    return parser.parse_args()


def find_test_index(test, selected_tests, find_last_index=False):
    """Find the index of the first or last occurrence of a given test/test module in the list of selected tests.
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


def get_selected_tests(options):
    selected_tests = []
    if options.include:
        for item in options.include:
            selected_tests.extend(list(filter(lambda test_name: item == test_name
                                              or (item in TESTS_MODULE and test_name.startswith(item)), TESTS)))
    else:
        selected_tests = TESTS

    if options.core:
        selected_tests = list(filter(lambda test_name: test_name in CORE_TEST_LIST, selected_tests))

    if options.first:
        first_index = find_test_index(options.first, selected_tests)
        selected_tests = selected_tests[first_index:]

    if options.last:
        last_index = find_test_index(options.last, selected_tests, find_last_index=True)
        selected_tests = selected_tests[: last_index + 1]

    for item in options.exlude:
        selected_tests = list(filter(lambda test_name: not test_name.startswith(item), selected_tests))

    return selected_tests


def run_test(test, test_directory, options):
    unittest_args = options.additional_unittest_args.copy()

    if options.verbose:
        unittest_args.append("-v")

    # Enable autoloading
    env = os.environ.copy()
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"

    # get python cmd.
    executable = [sys.executable]

    # Can't call `python -m unittest test_*` here because it doesn't run code
    # in `if __name__ == '__main__': `. So call `python test_*.py` instead.
    argv = [test + ".py"] + unittest_args

    command = executable
    calculate_python_coverage = os.getenv("CALCULATE_PYTHON_COVERAGE")
    if calculate_python_coverage and calculate_python_coverage == "1":
        command = command + ["-m", "coverage", "run", "-p", "--source=torch_npu", "--branch"]
    command = command + argv
    print_to_stderr("Executing {} ... [{}]".format(command, datetime.now(tz=timezone.utc)))
    return shell(command, cwd=test_directory, env=env)


def run_distributed_test(test, test_directory, options):
    config = DISTRIBUTED_TESTS_CONFIG
    for backend, env_vars in config.items():
        methods = {True, False}
        if options.init_method == 1:
            methods = {False}
        elif options.init_method == 2:
            methods = {True}
        for with_init_file in methods:
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
                return_code = run_test(test, test_directory, options)
                if return_code != 0:
                    return return_code
            finally:
                PathManager.remove_path_safety(tmp_dir)
    return 0


def _test_cpp_extensions_aot(test_directory, options, use_ninja):
    if use_ninja:
        try:
            cpp_extension.verify_ninja_availability()
        except RuntimeError:
            print(CPP_EXTENSIONS_ERROR)
            return 1

    # Wipe the build folder, if it exists already
    test_cpp_extensions_directory = os.path.join(test_directory, "cpp_extensions")
    cpp_extensions_test_build_dir = os.path.join(test_cpp_extensions_directory, "build")
    cpp_extensions_test_tests_dir = os.path.join(test_cpp_extensions_directory, "test")
    if os.path.exists(cpp_extensions_test_build_dir):
        PathManager.remove_path_safety(cpp_extensions_test_build_dir)

    # Build the test cpp extensions modules
    shell_env = os.environ.copy()
    shell_env["USE_NINJA"] = str(1 if use_ninja else 0)
    cmd = [sys.executable, "setup.py", "install", "--root", "./install"]
    return_code = shell(cmd, cwd=test_cpp_extensions_directory, env=shell_env)
    return_code = 0
    if return_code != 0:
        return return_code

    python_path = os.environ.get("PYTHONPATH", "")
    from shutil import copyfile

    test_module = "test_cpp_extensions_aot" + ("_ninja" if use_ninja else "_no_ninja")
    copyfile(
        os.path.join(cpp_extensions_test_tests_dir, "test_cpp_extensions_aot.py"),
        os.path.join(cpp_extensions_test_tests_dir, test_module + ".py")
    )
    try:
        install_directory = ""
        # install directory is the one that is named site-packages
        for root, directories, _ in os.walk(os.path.join(test_cpp_extensions_directory, "install")):
            for directory in directories:
                if "-packages" in directory:
                    install_directory = os.path.join(root, directory)

        assert install_directory, "install_directory must not be empty"
        os.environ["PYTHONPATH"] = os.pathsep.join([install_directory, python_path])
        return run_test(test_module, cpp_extensions_test_tests_dir, options)
    finally:
        os.environ["PYTHONPATH"] = python_path
        if os.path.exists(test_cpp_extensions_directory + "/" + test_module + ".py"):
            os.remove(test_cpp_extensions_directory + "/" + test_module + ".py")


def run_cpp_extensions(test, test_directory, options):
    if "test_cpp_extensions_aot" not in test:
        return run_test(test, test_directory, options)

    for use_ninja in [True, False]:
        return_code = _test_cpp_extensions_aot(test_directory, options, use_ninja)
        assert isinstance(return_code, int) and not isinstance(return_code, bool), "Return code should be an integer"
        if return_code != 0:
            return return_code

    return 0


CUSTOM_HANDLERS = {
    "distributed": run_distributed_test,
    "cpp_extensions": run_cpp_extensions,
}


def run_test_module(test: str, test_directory: str, options) -> Optional[str]:
    test_module = parse_test_module(test)

    print_to_stderr("Running {} ... [{}]".format(test, datetime.now(tz=timezone.utc)))
    handler = CUSTOM_HANDLERS.get(test_module, run_test)

    return_code = handler(test, test_directory, options)
    assert isinstance(return_code, int) and not isinstance(return_code, bool), "Return code should be an integer"
    if return_code == 0:
        return None

    message = f"exec ut {test} failed!"
    if return_code < 0:
        # subprocess.Popen returns the child process' exit signal as
        # return code -N, where N is the signal number.
        signal_name = SIGNALS_TO_NAMES_DICT[-return_code]
        message += f" Received signal: {signal_name}"
    return message


def main():
    options = parse_args()
    test_directory = os.path.join(REPO_ROOT, "test")
    selected_tests = get_selected_tests(options)

    if options.verbose:
        print_to_stderr("Selected tests: {}".format(", ".join(selected_tests)))

    has_failed = False
    failure_msgs = []

    for test in selected_tests:
        err_msg = run_test_module(test, test_directory, options)

        if err_msg is None:
            continue
        has_failed = True
        failure_msgs.append(err_msg)
        print_to_stderr(err_msg)

    if has_failed:
        for err in failure_msgs:
            print_to_stderr(err)
        sys.exit(1)


if __name__ == "__main__":
    main()
