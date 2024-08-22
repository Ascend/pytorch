# -*- coding: UTF-8 -*-

import os
import sys
import subprocess
import threading
import queue
import argparse
from pathlib import Path
import random
import psutil
from access_control import (
    TestMgr,
    BASE_DIR, TEST_DIR, SLOW_TEST_BLOCKLIST, NOT_RUN_DIRECTLY, EXEC_TIMEOUT, NETWORK_OPS_DIR
)


def exec_ut(files):
    """
    执行单元测试文件，其中存在失败，则标识异常并打印相关信息
    """

    def get_op_name(ut_file):
        op_name = str(ut_file.split('/')[-1].split('.')[0])
        return op_name[5:] if op_name.startswith("test_") else op_name

    def get_ut_name(ut_file):
        if 'op-plugin' in str(Path(ut_file)):
            return str(Path(ut_file).relative_to(NETWORK_OPS_DIR))[:-3]
        return str(Path(ut_file).relative_to(TEST_DIR))[:-3]

    def get_ut_cmd(ut_type, ut_file):
        cmd = [sys.executable, "run_test.py", "-v"]
        if ut_type == "op_ut_files":
            # do not skip ops related test entries
            return cmd + ["-e"] + SLOW_TEST_BLOCKLIST[1:] + ["-i", "test_ops", "--", "-k", "_" + get_op_name(ut_file)]
        if 'op-plugin' in str(Path(ut_file)):
            cmd = [sys.executable, NETWORK_OPS_DIR / "run_test.py", "-v"]
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
        init_method = random.randint(1, 2)
        for ut_type, ut_files in test_files.items():
            for ut_file in ut_files:
                cmd = get_ut_cmd(ut_type, ut_file)
                ut_info = str(cmd[-1])
                if ut_type == "op_ut_files":
                    ut_info = "test_ops " + ut_info
                else:
                    cmd = cmd if 'op-plugin' in str(Path(ut_file)) else cmd + ["--init_method={}".format(init_method)]
                ret = run_cmd_with_timeout(cmd)
                if ret:
                    has_failed = ret
                    test_infos.append("exec ut {} failed.".format(ut_info))
                else:
                    test_infos.append("exec ut {} success.".format(ut_info))
                init_method = 2 if init_method == 1 else 1
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
    print(f"options: {options}")
    cur_modify_files = str(BASE_DIR / 'modify_files.txt')
    test_mgr = TestMgr()

    if options.all:
        test_mgr.load_all_ut(options.distributed, options.network_ops)
    elif options.distributed:
        test_mgr.load_distributed_ut()
    elif os.path.exists(cur_modify_files):
        test_mgr.load(cur_modify_files, world_size=options.world_size)
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
