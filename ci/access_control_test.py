# -*- coding: UTF-8 -*-

import os
import re
import sys
import subprocess
import threading
import queue
from abc import ABCMeta, abstractmethod
from pathlib import Path


BASE_DIR = Path(__file__).absolute().parent.parent
TEST_DIR = BASE_DIR / 'test'


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
            pass # 对于找不到的暂时不作处理
        else:
            files = output.split('\n')
            for ut_file in files:
                if ut_file.endswith(".py"):
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
    core_test_cases = [str(i) for i in (BASE_DIR / 'test/test_npu').rglob('test_*.py')]
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
    'contrib': 'test/test_contrib',
    'cpp_extension': 'test/test_cpp_extension', 
    'distributed': 'test/test_distributed', 
    'fx': 'test/test_fx',
    'hooks': 'test/test_hooks', 
    'optim': 'test/test_optim',
    'profiler': 'test/test_profiler',
    'onnx': 'test/test_onnx',
    'utils': 'test/test_utils',
    'testing': 'test/test_testing.py',
    }

    def identify(self, modify_file):
        current_all_ut_path = []
        if str(Path(modify_file).parts[0]) == 'torch_npu':
            mapped_ut_path = []
            module_name = str(Path(modify_file).parts[1])
            if module_name == 'csrc':
                module_name = str(Path(modify_file).parts[2])
            if module_name in self.mapping_list:
                mapped_ut_path.append(self.mapping_list[module_name])
            file_name = str(Path(modify_file).stem)
            if file_name in self.mapping_list:
                mapped_ut_path.append(self.mapping_list[file_name])
            
            for mapped_path in mapped_ut_path:
                if Path.is_file(BASE_DIR / mapped_path):
                    current_all_ut_path.append(str(BASE_DIR / mapped_path))
                else:
                    current_all_ut_path += [str(i) for i in (BASE_DIR / mapped_path).rglob('test_*.py')]
        return current_all_ut_path

class TestMgr():
    def __init__(self):
        self.modify_files = []
        self.test_files = {
            'ut_files': [],
            'op_ut_files': []
        }

    def load(self, modify_files):
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


def exec_ut(files):
    """
    执行单元测试文件，其中存在失败，则标识异常并打印相关信息
    """
    def get_op_name(ut_file):
        return ut_file.split('/')[-1].split('.')[0].lstrip('test_')
    
    def get_ut_name(ut_file):
        return str(Path(ut_file).relative_to(TEST_DIR))[:-3]

    def get_ut_cmd(ut_type, ut_file):
        cmd = [sys.executable, "run_test.py", "-v", "-i"]
        if ut_type == "op_ut_files":
            return cmd + ["test_ops", "--", "-k", get_op_name(ut_file)]
        return cmd + [get_ut_name(ut_file)]

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
        while (not log_queue.empty()):
            print((log_queue.get()).strip())

    def run_cmd_with_timeout(cmd):
        os.chdir(str(TEST_DIR))
        stdout_queue = queue.Queue()
        event_timer = threading.Event()

        p = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        start_thread(wait_thread, p, event_timer)
        start_thread(enqueue_output, p.stdout, stdout_queue)

        try:
            event_timer.wait(2000)
            ret = p.poll()
            if ret:
                print_subprocess_log(stdout_queue)
            if not event_timer.is_set():
                ret = 1
                p.kill()
                p.terminate()
                print("Timeout: Command '{}' timed out after 2000 seconds".format(" ".join(cmd)))
                print_subprocess_log(stdout_queue)
        except Exception as err:
            ret = 1
            print(err)
        return ret

    def run_tests(files):
        exec_infos = []
        has_failed = 0
        for ut_type, ut_files in files.items():
            for ut_file in ut_files:
                cmd = get_ut_cmd(ut_type, ut_file)
                ut_info = " ".join(cmd[4:]).replace(" -- -k", "")
                ret = run_cmd_with_timeout(cmd)
                if ret:
                    has_failed = ret
                    exec_infos.append("exec ut {} failed.".format(ut_info))
                else:
                    exec_infos.append("exec ut {} success.".format(ut_info))
        return has_failed, exec_infos

    ret_status, exec_infos = run_tests(files)

    print("***** Total result:")
    for exec_info in exec_infos:
        print(exec_info)
    return ret_status


if __name__ == "__main__":
    cur_modify_files = str(BASE_DIR / 'modify_files.txt')
    test_mgr = TestMgr()
    test_mgr.load(cur_modify_files)
    test_mgr.analyze()
    cur_test_files = test_mgr.get_test_files()

    test_mgr.print_modify_files()
    test_mgr.print_ut_files()
    test_mgr.print_op_ut_files()

    ret_ut = exec_ut(cur_test_files)
    sys.exit(ret_ut)
