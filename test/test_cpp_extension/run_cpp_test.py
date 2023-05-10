# Copyright (c) 2023, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain data copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import copy
import signal
import shutil
import pathlib
import argparse
from datetime import datetime
from distutils.util import strtobool

import torch
from torch.utils import cpp_extension
from torch.testing._internal.common_utils import shell

import torch_npu

REPO_ROOT = pathlib.Path(__file__).resolve().parent

CPP_EXTENSIONS_ERROR = """
Ninja (https://ninja-build.org) is required for some of the C++ extensions
tests, but it could not be found. Install ninja with `pip install ninja`
or `conda install ninja`.
"""

def print_to_stderr(message):
    print(message, file=sys.stderr)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the PyTorch unit test suite",
        formatter_class=argparse.RawTextHelpFormatter,)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="print verbose information and test-by-test results",)
    parser.add_argument(
        "--continue-through-error",
        action="store_true",
        help="Runs the full test suite despite one of the tests failing",
        default=strtobool(os.environ.get("CONTINUE_THROUGH_ERROR", "False")),)

    return parser.parse_args()

def run_test(test_module, test_directory, options, launcher_cmd=None, extra_unittest_args=None):

    executable = [sys.executable]

    argv = [test_module + ".py"] 
    command = (launcher_cmd or []) + executable + argv
    print_to_stderr("Executing {} ... [{}]".format(command, datetime.now()))
    return shell(command, test_directory)

def _test_cpp_extensions_aot(test_directory, options, use_ninja):
    if use_ninja:
        try:
            cpp_extension.verify_ninja_availability()
        except RuntimeError:
            print(CPP_EXTENSIONS_ERROR)
            return 1

    # Wipe the build folder, if it exists already
    cpp_extensions_test_dir = os.path.join(test_directory, "cpp_extensions")
    cpp_extensions_test_build_dir = os.path.join(cpp_extensions_test_dir, "build")
    if os.path.exists(cpp_extensions_test_build_dir):
        shutil.rmtree(cpp_extensions_test_build_dir)

    # Build the test cpp extensions modules
    shell_env = os.environ.copy()
    shell_env["USE_NINJA"] = str(1 if use_ninja else 0)
    cmd = [sys.executable, "setup.py", "install", "--root", "./install"]
    return_code = shell(cmd, cwd=cpp_extensions_test_dir, env=shell_env)
    return_code = 0
    if return_code != 0:
        return return_code

    python_path = os.environ.get("PYTHONPATH", "")
    from shutil import copyfile

    test_module = "test_cpp_extensions_aot" + ("_ninja" if use_ninja else "_no_ninja")
    copyfile(
        test_directory + "/test_cpp_extensions_aot.py",
        test_directory + "/" + test_module + ".py",
    )
    try:
        cpp_extensions = os.path.join(test_directory, "cpp_extensions")
        install_directory = ""
        # install directory is the one that is named site-packages
        for root, directories, _ in os.walk(os.path.join(cpp_extensions, "install")):
            for directory in directories:
                if "-packages" in directory:
                    install_directory = os.path.join(root, directory)

        assert install_directory, "install_directory must not be empty"
        os.environ["PYTHONPATH"] = os.pathsep.join([install_directory, python_path])
        return run_test(test_module, test_directory, options)
    finally:
        os.environ["PYTHONPATH"] = python_path
        if os.path.exists(test_directory + "/" + test_module + ".py"):
            os.remove(test_directory + "/" + test_module + ".py")

def test_cpp_extensions_aot_ninja(test_directory, options, use_ninja=True):
    return _test_cpp_extensions_aot(test_directory, options, use_ninja)

def test_cpp_extensions_aot_no_ninja(test_directory, options, use_ninja=False):
    return _test_cpp_extensions_aot(test_directory, options, use_ninja)

# https://stackoverflow.com/questions/2549939/get-signal-names-from-numbers-in-python
SIGNALS_TO_NAMES_DICT = {
    getattr(signal, n): n for n in dir(signal) if n.startswith("SIG") and "_" not in n
}

CUSTOM_HANDLERS = {
    "test_cpp_extensions_aot_no_ninja": test_cpp_extensions_aot_no_ninja,
    "test_cpp_extensions_aot_ninja": test_cpp_extensions_aot_ninja,
}

def run_test_module(test, test_directory, options):
    handler = CUSTOM_HANDLERS.get(test)
    return_code = handler(test_directory, options)

    assert isinstance(return_code, int) and not isinstance(return_code, bool), \
        "Return code should be an integer"
    if return_code == 0:
        return None

    message = f"{test} failed!"
    if return_code < 0:
        # subprocess.Popen returns the child process' exit signal as
        # return code -N, where N is the signal number.
        signal_name = SIGNALS_TO_NAMES_DICT[-return_code]
        message += f" Received signal: {signal_name}"
    return message

def main():
    options = parse_args()
    test_directory = str(REPO_ROOT)
    selected_tests = ["test_cpp_extensions_aot_no_ninja", "test_cpp_extensions_aot_no_ninja"]

    if options.verbose:
        print_to_stderr("Selected tests: {}".format(", ".join(selected_tests)))
        
    failure_messages = []
    for test in selected_tests:
        options_clone = copy.deepcopy(options)
        err_message = run_test_module(test, test_directory, options_clone)
        if err_message is None:
            continue
        has_failed = True
        failure_messages.append(err_message)
        print(err_message)
        if not options_clone.continue_through_error:
            raise RuntimeError(err_message)
        print_to_stderr(err_message)

    if options.continue_through_error and has_failed:
        for err in failure_messages:
            print_to_stderr(err)
        sys.exit(1)


if __name__ == "__main__":
    main()
