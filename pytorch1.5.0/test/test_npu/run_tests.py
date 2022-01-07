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
import time
import unittest
import os

FAILURE_FILE_NAME = 'failures.txt'
ERROR_FILE_NAME = 'errors.txt'

def load_failure_error_cases(file_name):
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

def analyse_failure_error_cases(result):
    new_failures = []
    new_errors = []

    if len(result.failures) > 0:
        print("====================================== failed cases count: ", len(result.failures))
        for failure in result.failures:
            print(failure[0])
        print("============================================================\n")
        orig_failures = load_failure_error_cases(FAILURE_FILE_NAME)
        for failure in result.failures:
            if str(failure[0]) not in orig_failures:
                new_failures.append(str(failure[0]))

    if len(result.errors) > 0:
        print("====================================== error cases count: ", len(result.errors))
        for error_case in result.errors:
            print(error_case[0])
        print("============================================================\n")
        orig_errors = load_failure_error_cases(ERROR_FILE_NAME)
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

def load_local_case(test_case_path):
    discover = unittest.defaultTestLoader.discover(test_case_path, "test_*.py")
    return discover

def run_tests():
    test_case_path = os.path.dirname(os.path.realpath(__file__))
    test_report_path = os.path.join(test_case_path, 'ReportResult')
    ENABLE_HTML = bool(os.environ.get('ENABLE_HTML'))
    ENABLE_HTML_MX = bool(os.environ.get('ENABLE_HTML_MX'))
    ENABLE_CASE_PATH = os.environ.get('ENABLE_CASE_PATH')
    ENABLE_OUTPUT_PATH = os.environ.get('ENABLE_OUTPUT_PATH')
    WHITE_LIST_PATH = os.environ.get('WHITE_LIST_PATH')
    if WHITE_LIST_PATH and os.path.exists(WHITE_LIST_PATH):
        global FAILURE_FILE_NAME
        global ERROR_FILE_NAME
        FAILURE_FILE_NAME = os.path.join(WHITE_LIST_PATH, 'failures.txt')
        ERROR_FILE_NAME = os.path.join(WHITE_LIST_PATH, 'errors.txt')

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
    htmlFileName = os.path.join(test_report_path, 'pytorch-unittest-report-' + now + '.html')
    txtFileName = os.path.join(test_report_path, 'pytorch-unittest-report-' + now + '.txt')

    if ENABLE_HTML:
        print('start pytorch HTML unittest testset...')
        import HTMLTestRunner
        with open(htmlFileName, "wb") as report_file:
            runner = HTMLTestRunner.HTMLTestRunner(stream=report_file, title='AllTest', description='all npu test case', verbosity=2)
            result = runner.run(load_local_case(test_case_path))
            new_failures, new_errors = analyse_failure_error_cases(result)
            if len(new_failures) + len(new_errors) > 0:
                print(" RuntimeError: new error or failed cases found!")
        print('report files path', htmlFileName)
    elif ENABLE_HTML_MX:
        print('start pytorch Multi HTML unittest testset...')
        import HtmlTestRunner
        runner = HtmlTestRunner.HTMLTESTRunner(output=test_report_path, verbosity=2)
        result = runner.run(load_local_case(test_case_path))
        if not result.wasSuccessful():
            raise RuntimeError("Some cases of Multi HTML unittest testset failed")
    else:
        print('start pytorch TEXT unittest testset...')
        with open(txtFileName, "a") as report_file:
            runner = unittest.TextTestRunner(stream=report_file, verbosity=2)
            result = runner.run(load_local_case(test_case_path))
            if not result.wasSuccessful():
                raise RuntimeError("Some cases TEXT unittest failed")
        print('report files path', txtFileName)

if __name__ == "__main__":
    run_tests()


