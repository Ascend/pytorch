# -*- coding: UTF-8 -*-

import os
import re
import sys
import subprocess
import threading
import queue
import argparse
from abc import ABCMeta, abstractmethod
from pathlib import Path
import psutil
from torch_npu.utils.path_manager import PathManager
import torch_npu

BASE_DIR = Path(__file__).absolute().parent.parent
TEST_DIR = BASE_DIR / 'test'

# Add slow test cases here (first element must be test_ops)
SLOW_TEST_BLOCKLIST = [
    'test_ops',
    'test_modules',
    'test_binary_ufuncs',
    'test_ops_fwd_gradients',
    'test_ops_gradients',
    'test_reductions',
    'test_unary_ufuncs',
    'test_ops_jit',
    'onnx/test_fx_op_consistency',
    "onnx/test_op_consistency"
]

# exclude some not run directly test files
NOT_RUN_DIRECTLY = {
    "jit": "test_jit.py",
}

# include some files
INCLUDE_FILES = [
    'jit/test_complexity.py',
]

# default ut cmd execution timeout is 2000s
EXEC_TIMEOUT = os.getenv("PTA_UT_EXEC_TIMEOUT", 2000)
try:
    EXEC_TIMEOUT = int(EXEC_TIMEOUT)
except ValueError:
    EXEC_TIMEOUT = 2000


class AccurateTest(metaclass=ABCMeta):
    @abstractmethod
    def identify(self, modify_file):
        """
        该接口提供代码对应的UT的路径信息
        """
        raise Exception("abstract method. Subclasses should implement it.")

    @staticmethod
    def find_ut_by_regex(regex):
        ut_files = []
        cmd = "find {} -name {}".format(str(TEST_DIR), regex)
        status, output = subprocess.getstatusoutput(cmd)
        if status or not output:
            pass  # 对于找不到的暂时不作处理
        else:
            files = output.split('\n')
            for ut_file in files:
                ut_file_basename = os.path.basename(ut_file)
                if ut_file_basename.startswith("test") and ut_file.endswith(".py"):
                    ut_files.append(ut_file)
        return ut_files


class OpStrategy(AccurateTest):
    """
    通过对适配层的代码的识别
    """

    def identify(self, modify_file):
        """
        通过对于算子实现文件的文件名解析获取其单元测试的名字，比如：
        BinaryCrossEntropyWithLogitsBackwardKernelNpu.cpp
        针对这个文件，先识别关键字BinaryCrossEntropyWithLogitsBackward
        然后，获取其正则表达式*binary*cross*entropy*with*logits*backward*识别到符合要求的测试用例
        具体方法：通过大写字母切分关键字，再识别包含所有这些关键字的测试文件名。
        """
        filename = Path(modify_file).name
        if filename.find('KernelNpu') >= 0:
            feature_line = filename.split('KernelNpu')[0]
            features = re.findall('[A-Z][^A-Z]*', feature_line)
            regex = '*' + '*'.join([f"{feature.lower()}" for feature in features]) + '*'
            return AccurateTest.find_ut_by_regex(regex)
        return []


class DirectoryStrategy(AccurateTest):
    """
    Determine whether the modified files are test cases
    """

    def identify(self, modify_file):
        is_test_file = str(Path(modify_file).parts[0]) == "test" \
                       and re.match("test_(.+).py", Path(modify_file).name)
        return [(str(BASE_DIR / modify_file))] if is_test_file else []


class CoreTestStrategy(AccurateTest):
    """
    Determine whether the core tests should be runned
    """
    block_list = ['test', 'docs']
    core_test_cases = [str(i) for i in (BASE_DIR / 'test/npu').rglob('test_*.py')]

    def identify(self, modify_file):
        modified_module = str(Path(modify_file).parts[0])
        if modified_module not in self.block_list:
            return self.core_test_cases
        return []


class CopyOptStrategy(AccurateTest):
    """
    通过识别非连续转连续的测试用例
    """

    def identify(self, modify_file):
        if modify_file.find('contiguous') > 0:
            regex = '*contiguous*'
            return AccurateTest.find_ut_by_regex(regex)
        return []


class DirectoryMappingStrategy(AccurateTest):
    """
    Map the modified files to the corresponding test cases
    """
    mapping_list = {
        'contrib': 'test/contrib',
        'cpp_extension': 'test/cpp_extensions',
        'distributed': 'test/distributed',
        'fx': 'test/test_fx.py',
        'optim': 'test/optim',
        'profiler': 'test/profiler',
        'onnx': 'test/onnx',
        'utils': 'test/test_utils.py',
        'testing': 'test/test_testing.py',
        'jit': 'test/test_jit.py',
        'rpc': 'test/distributed/rpc',
    }

    def get_module_name(self, modify_file):
        module_name = str(Path(modify_file).parts[1])
        if module_name == 'csrc':
            module_name = str(Path(modify_file).parts[2])
        for part in Path(modify_file).parts:
            if part == 'rpc':
                module_name = 'rpc'
        if module_name == 'utils' and Path(modify_file).parts[2] == 'cpp_extension.py':
            module_name = 'cpp_extension'
        return module_name

    def identify(self, modify_file):
        current_all_ut_path = []
        if str(Path(modify_file).parts[0]) == 'torch_npu':
            mapped_ut_path = []
            module_name = self.get_module_name(modify_file)
            if module_name in self.mapping_list:
                mapped_ut_path.append(self.mapping_list[module_name])
            file_name = str(Path(modify_file).stem)
            if file_name in self.mapping_list:
                mapped_ut_path.append(self.mapping_list[file_name])

            for mapped_path in mapped_ut_path:
                if Path.is_file(BASE_DIR / mapped_path):
                    current_all_ut_path.append(str(BASE_DIR / mapped_path))
                else:
                    current_all_ut_path += [str(i) for i in (BASE_DIR / mapped_path).glob('test_*.py')]
        return current_all_ut_path


class TestMgr:
    def __init__(self):
        self.modify_files = []
        self.test_files = {
            'ut_files': [],
            'op_ut_files': []
        }

    def load(self, modify_files):
        PathManager.check_directory_path_readable(modify_files)
        with open(modify_files) as f:
            for line in f:
                line = line.strip()
                self.modify_files.append(line)

    def analyze(self):
        for modify_file in self.modify_files:
            self.test_files['ut_files'] += DirectoryStrategy().identify(modify_file)
            self.test_files['ut_files'] += CopyOptStrategy().identify(modify_file)
            self.test_files['ut_files'] += OpStrategy().identify(modify_file)
            self.test_files['ut_files'] += DirectoryMappingStrategy().identify(modify_file)
            self.test_files['op_ut_files'] += OpStrategy().identify(modify_file)
            self.test_files['ut_files'] += CoreTestStrategy().identify(modify_file)
        unique_files = sorted(set(self.test_files['ut_files']))

        exist_ut_file = [
            changed_file
            for changed_file in unique_files
            if Path(changed_file).exists()
        ]
        self.test_files['ut_files'] = exist_ut_file
        self.exclude_test_files(slow_files=SLOW_TEST_BLOCKLIST)

    def load_core_ut(self):
        self.test_files['ut_files'] += [str(i) for i in (BASE_DIR / 'test/npu').rglob('test_*.py')]

    def load_distributed_ut(self):
        self.test_files['ut_files'] += [str(i) for i in (BASE_DIR / 'test/distributed').rglob('test_*.py')]
        
    def load_op_plugin_ut(self):
        version_path = get_test_torch_version_path()
        file_hash = {}
        for file_path in (BASE_DIR / 'third_party/op-plugin/test').rglob('test_*.py'):
            if str(file_path.parts[-2]) in [version_path, "test_custom_ops", "test_base_ops"]:
                file_name = str(file_path.name)
                if file_name in file_hash:
                    if str(file_path.parts[-2]) == version_path:
                        self.test_files['ut_files'].remove(file_hash[file_name])
                        file_hash[file_name] = str(file_path)
                        self.test_files['ut_files'].append(str(file_path))
                else:
                    file_hash[file_name] = str(file_path)
                    self.test_files['ut_files'].append(str(file_path))

    def load_all_ut(self, include_distributed_case=False, include_op_plugin_case=False):
        all_files = [str(i) for i in (BASE_DIR / 'test').rglob('test_*.py') if 'distributed' not in str(i)]
        self.test_files['ut_files'] = all_files
        if include_distributed_case:
            self.load_distributed_ut()
        if include_op_plugin_case:
            if os.path.exists(BASE_DIR / 'third_party/op-plugin/test'):
                self.load_op_plugin_ut()
            else:
                raise Exception("The path of op-plugin did not exist, check whether it had been pulled.")

    def split_test_files(self, rank, world_size):
        if rank > world_size:
            raise Exception(f'rank {rank} is greater than world_size {world_size}')

        def ordered_split(files, start, step):
            return sorted(files)[start::step] if files else []

        # node rank starts from 1
        self.test_files['ut_files'] = ordered_split(self.test_files['ut_files'], rank - 1, world_size)
        self.test_files['op_ut_files'] = ordered_split(self.test_files['op_ut_files'], rank - 1, world_size)

    def exclude_test_files(self, slow_files=None, not_run_files=None, mode="slow_test"):
        """
        Args:
            slow_files: slow test files.
            not_run_files: not_run_directly test files.
            mode: "slow_test" or "not_run_directly". Default: "slow_test".

        Returns:
        """

        def remove_test_files(key):
            block_list = not_run_files.keys() if mode == "not_run_directly" else slow_files
            for test_name in block_list:
                test_files_copy = self.test_files[key][:]
                for test_file in test_files_copy:
                    is_remove = False
                    if mode == "slow_test" and test_name + ".py" in test_file:
                        print(f'Excluding slow test: {test_name}')
                        is_remove = True
                    if mode == "not_run_directly":
                        # remove not run directly files
                        is_folder_in = not test_name.endswith(".py") and "/" + test_name + "/" in test_file
                        is_file_in = test_name.endswith(".py") and test_name in test_file
                        if is_folder_in or is_file_in:
                            if test_name in ["jit"] and re.search("|".join(INCLUDE_FILES), test_file):
                                continue
                            is_remove = True
                            # add instead file
                            instead_files.add(str(TEST_DIR / not_run_files[test_name]))
                    if is_remove:
                        self.test_files[key].remove(test_file)

        for test_files_key in self.test_files.keys():
            instead_files = set()
            remove_test_files(test_files_key)
            for instead_file in instead_files:
                if instead_file not in self.test_files[test_files_key]:
                    self.test_files[test_files_key].append(instead_file)

    def get_test_files(self):
        return self.test_files

    def print_modify_files(self):
        print("modify files:")
        for modify_file in self.modify_files:
            print(modify_file)

    def print_ut_files(self):
        print("ut files:")
        for ut_file in self.test_files['ut_files']:
            print(ut_file)

    def print_op_ut_files(self):
        print("op ut files:")
        for op_ut_file in self.test_files['op_ut_files']:
            print(op_ut_file)


def get_test_torch_version_path():
    torch_npu_version = torch_npu.__version__
    version_list = torch_npu_version.split('.')
    if len(version_list) > 2:
        return f'test_v{version_list[0]}r{version_list[1]}_ops'
    else:
        raise RuntimeError("Invalid torch_npu version.")


def exec_ut(files):
    """
    执行单元测试文件，其中存在失败，则标识异常并打印相关信息
    """

    def get_op_name(ut_file):
        op_name = str(ut_file.split('/')[-1].split('.')[0])
        return op_name[5:] if op_name.startswith("test_") else op_name

    def get_ut_name(ut_file):
        return str(Path(ut_file).relative_to(TEST_DIR))[:-3]

    def get_ut_cmd(ut_type, ut_file):
        cmd = [sys.executable, "run_test.py", "-v"]
        if ut_type == "op_ut_files":
            # do not skip ops related test entries
            return cmd + ["-e"] + SLOW_TEST_BLOCKLIST[1:] + ["-i", "test_ops", "--", "-k", "_" + get_op_name(ut_file)]
        return cmd + ["-i", get_ut_name(ut_file)]

    def wait_thread(process, event_timer):
        process.wait()
        event_timer.set()

    def enqueue_output(out, log_queue):
        for line in iter(out.readline, b''):
            log_queue.put(line.decode('utf-8'))
        out.close()
        return

    def start_thread(fn, *args):
        stdout_t = threading.Thread(target=fn, args=args)
        stdout_t.daemon = True
        stdout_t.start()

    def print_subprocess_log(log_queue):
        while not log_queue.empty():
            print((log_queue.get()).strip())

    def run_cmd_with_timeout(cmd):
        os.chdir(str(TEST_DIR))
        stdout_queue = queue.Queue()
        event_timer = threading.Event()

        p = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        start_thread(wait_thread, p, event_timer)
        start_thread(enqueue_output, p.stdout, stdout_queue)

        try:
            event_timer.wait(EXEC_TIMEOUT)
            ret = p.poll()
            if ret:
                print_subprocess_log(stdout_queue)
            if not event_timer.is_set():
                ret = 1
                parent_process = psutil.Process(p.pid)
                for children_process in parent_process.children(recursive=True):
                    children_process.kill()
                p.kill()
                p.terminate()
                print("Timeout: Command '{}' timed out after {} seconds".format(" ".join(cmd), EXEC_TIMEOUT))
                print_subprocess_log(stdout_queue)
        except Exception as err:
            ret = 1
            print(err)
        return ret

    def run_tests(test_files):
        test_infos = []
        has_failed = 0
        for ut_type, ut_files in test_files.items():
            for ut_file in ut_files:
                cmd = get_ut_cmd(ut_type, ut_file)
                ut_info = str(cmd[-1])
                if ut_type == "op_ut_files":
                    ut_info = "test_ops " + ut_info
                ret = run_cmd_with_timeout(cmd)
                if ret:
                    has_failed = ret
                    test_infos.append("exec ut {} failed.".format(ut_info))
                else:
                    test_infos.append("exec ut {} success.".format(ut_info))
        return has_failed, test_infos

    ret_status, exec_infos = run_tests(files)

    print("***** Total result:")
    for exec_info in exec_infos:
        print(exec_info)
    return ret_status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Control needed ut cases')
    parser.add_argument('--all', action="store_true", help='Run all testcases')
    parser.add_argument('--distributed', action="store_true", help='Run distributed testcases')
    parser.add_argument('--rank', default=0, type=int, help='Index of current ut nodes')
    parser.add_argument('--world_size', default=0, type=int, help='Number of ut nodes')
    parser.add_argument('--network_ops', action="store_true", help='Run network_ops testcases in the op-plugin repo')
    options = parser.parse_args()
    cur_modify_files = str(BASE_DIR / 'modify_files.txt')
    test_mgr = TestMgr()

    if options.all:
        test_mgr.load_all_ut(options.distributed, options.network_ops)
    elif options.distributed:
        test_mgr.load_distributed_ut()
    elif os.path.exists(cur_modify_files):
        test_mgr.load(cur_modify_files)
        test_mgr.analyze()
    else:
        test_mgr.load_core_ut()
    test_mgr.exclude_test_files(not_run_files=NOT_RUN_DIRECTLY, mode="not_run_directly")

    if options.rank > 0 and options.world_size > 0:
        test_mgr.split_test_files(options.rank, options.world_size)
    cur_test_files = test_mgr.get_test_files()

    test_mgr.print_modify_files()
    test_mgr.print_ut_files()
    test_mgr.print_op_ut_files()

    ret_ut = exec_ut(cur_test_files)
    sys.exit(ret_ut)
