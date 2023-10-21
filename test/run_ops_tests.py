"""
This file provides a tool to pull all ops related tests running on multiple NPUs.
Parameters need to specify:
all_test_names_path: path of test names file.
MAX_TIME_PER_CASE: max running time for each process, default is 5 min
MAX_PROC: max num of processes, default is 10.
result_file: file name to store fail tests.

***** steps *****:
1. create a directory called logs to store the temporary result.
2. designate --file and --result as described below.
3. run '>>python run_test_ops.py' to pull up the full ops tests!
"""

import subprocess
import time
from collections import defaultdict
import os
import shutil
import stat
import argparse
import torch
import torch_npu

parser = argparse.ArgumentParser(description="options of run all ops tests")
parser.add_argument("--file", help="file of all test names")
parser.add_argument("--result", help="file name to store the result.")
args = parser.parse_args()

MAX_TIME_PER_CASE = 3000
MAX_PROC = 10
flags = os.O_WRONLY | os.O_RDONLY | os.O_CREAT
modes = stat.S_IWUSR | stat.S_IRUSR
proc_pool = []
block_list = []
all_test_names_path = os.path.realpath(args.file)
result_file = os.path.realpath(args.result)
num_npus = torch.npu.device_count()

if not all_test_names_path:
    raise RuntimeError("Please provide a file of test names need to run.")
if not result_file:
    raise RuntimeError("Please provide a file to store the filed names. e.g: result.log")


def get_all_test_names(file_name):
    with open(file_name, "r") as f:
        tests = f.read().splitlines()
    return tests


tests_ = [n for n in get_all_test_names(all_test_names_path) if n.startswith("test_")]


def write_result(name):
    with os.fdopen(os.open(result_file, flags, modes), "a") as f:
        f.write(name + '\n')


def check_proc(proc):
    name, proc, prev_time = proc
    flag = proc.poll()
    if flag is None:
        if time.time() - prev_time > MAX_TIME_PER_CASE:
            proc.kill()
            proc.terminate()
            write_result(name)
            return True
        return False
    elif flag == 0:
        return True
    else:
        write_result(name)
        return True


def congestion_control():
    while len(proc_pool) > MAX_PROC:
        q_length = len(proc_pool)
        for _ in range(q_length):
            check_end = check_proc(proc_pool[0])
            if check_end:
                proc_pool.pop(0)
            else:
                proc_pool.append(proc_pool.pop(0))
        time.sleep(2)


def run_tests():
    curr_id = 0
    if not os.path.exists("logs"):
        os.mkdir("logs")
    for test in tests_:
        congestion_control()
        if test.startswith("test_"):
            try:
                pth = os.path.realpath(os.path.join("logs", "{}.log".format(test)))
                with os.fdopen(os.open(pth, flags, modes), "w") as f:
                    p = subprocess.Popen([shutil.which('python3'), os.path.realpath('test_ops.py'), '-v', '-k', test, str(curr_id)],
                                         stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
                    proc_pool.append((test, p, time.time()))
            except subprocess.CalledProcessError:
                block_list.append(test)
                with os.fdopen(os.open(result_file, flags, modes), "a") as f:
                    f.write(test + '\n')
                proc_pool.pop(0)

        curr_id = (curr_id + 1) % num_npus


if __name__ == "__main__":
    run_tests()
