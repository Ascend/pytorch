# Copyright (c) 2020 Huawei Technologies Co., Ltd
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
# -*- coding: UTF-8 -*-

import os
import re
import sys
import subprocess
from abc import ABCMeta, abstractmethod
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_UT_FILE = os.path.join(BASE_DIR, 'ci/pytorch_resnet.py')


class AccurateTest(metaclass=ABCMeta):
    @abstractmethod
    def identify(self, modify_files):
        """
        该接口提供代码对应的UT的路径信息
        """
        raise Exception("abstract method. Subclasses should implement it.")

    @staticmethod
    def find_ut_by_regex(regex):
        ut_files = []
        cmd = "find {} -name {}".format(os.path.join(BASE_DIR, 'test'), regex)
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
        filename = os.path.basename(modify_file)
        if filename.find('KernelNpu') >= 0: 
            feature_line = filename.split('KernelNpu')[0]
            features = re.findall('[A-Z][^A-Z]*', feature_line)
            regex = '*' + '*'.join(["%s" % feature.lower() for feature in features]) + '*'
            return AccurateTest.find_ut_by_regex(regex)
        return []


class DirectoryStrategy(AccurateTest):
    """
    通过识别测试文件的目录确认需要进行UT的文件
    """
    def identify(self, modify_file):
        second_dir = modify_file.split("/")[0]
        if second_dir == 'test':
            return [os.path.join(BASE_DIR, modify_file)]
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
            CORE_TEST_LIST = [str(i) for i in (Path(BASE_DIR) / 'test/test_npu').rglob('test_*.py')]
            current_all_ut_path = CORE_TEST_LIST

            mapped_ut_path = []
            module_name = str(Path(modify_file).parts[1])
            if module_name in self.mapping_list:
                mapped_ut_path.append(self.mapping_list[module_name])
            file_name = str(Path(modify_file).stem)
            if file_name in self.mapping_list:
                mapped_ut_path.append(self.mapping_list[file_name])
            
            for mapped_path in mapped_ut_path:
                if Path.is_file(Path(BASE_DIR) / mapped_path):
                    current_all_ut_path.append(str(Path(BASE_DIR) / mapped_path))
                else:
                    current_all_ut_path += [str(i) for i in (Path(BASE_DIR) / mapped_path).rglob('test_*.py')]
        return current_all_ut_path

class TestMgr():
    def __init__(self):
        self.modify_files = []
        self.ut_files = []

    def load(self, modify_files):
        with open(modify_files) as f:
            for line in f:
                line = line.strip()
                self.modify_files.append(line)

    def analyze(self):
        for modify_file in self.modify_files:
            self.ut_files += DirectoryStrategy().identify(modify_file)
            self.ut_files += OpStrategy().identify(modify_file)
            self.ut_files += CopyOptStrategy().identify(modify_file)
            self.ut_files += DirectoryMappingStrategy().identify(modify_file)
        unique_files = set(self.ut_files)

        exist_ut_file = []
        for changed_file in unique_files:
            if os.path.exists(changed_file):
                exist_ut_file.append(changed_file)
        self.ut_files = exist_ut_file
        self.ut_files.append(DEFAULT_UT_FILE)

    def get_ut_files(self):
        return self.ut_files

    def print_modify_files(self):
        print("modify files:")
        for modify_file in self.modify_files:
            print(modify_file)

    def print_ut_files(self):
        print("ut files:")
        for ut_file in self.ut_files:
            print(ut_file)


def exec_ut(ut_files):
    """
    执行单元测试文件，其中存在失败，则标识异常并打印相关信息
    """
    def change_dir_and_exec(ut_path):
        ut_dir = os.path.dirname(ut_path)
        ut_file = os.path.basename(ut_path)
        os.chdir(ut_dir)
        cmd = "python3 {}".format(ut_file)
        p = subprocess.Popen(['python3', ut_file], stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        try:
            msg = p.communicate(timeout=300)
            ret = p.poll()
            if ret:
                print(msg)

        except subprocess.TimeoutExpired:
            p.kill()
            p.terminate()
            ret = 1
            print("Timeout: Command '" + cmd + "' timed out after 300 seconds")
        except Exception as err:
            ret = 1
            print(err)
        return ret

    ret_status = 0
    exec_infos = []
    for ut_file in ut_files:
        temp_ret = change_dir_and_exec(ut_file)
        if temp_ret:
            ret_status = temp_ret
            exec_infos.append("exec ut {} failed.".format(ut_file))
        else:
            exec_infos.append("exec ut {} success.".format(ut_file))
    
    print("***** Total result:")
    for exec_info in exec_infos:
        print(exec_info)
    return ret_status


if __name__ == "__main__":
    cur_modify_files = os.path.join(BASE_DIR, 'modify_files.txt')
    test_mgr = TestMgr()
    test_mgr.load(cur_modify_files)
    test_mgr.analyze()
    cur_ut_files = test_mgr.get_ut_files()

    test_mgr.print_modify_files()
    test_mgr.print_ut_files()

    ret_ut = exec_ut(cur_ut_files)
    sys.exit(ret_ut)