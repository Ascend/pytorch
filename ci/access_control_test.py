# -*- coding: UTF-8 -*-

import os
import sys
import subprocess
import threading
import queue
import argparse
import shutil
import json
from pathlib import Path
import random
import time
import psutil
from access_control import (
    TestMgr,
    BASE_DIR, TEST_DIR, SLOW_TEST_BLOCKLIST, NOT_RUN_DIRECTLY, EXEC_TIMEOUT, NETWORK_OPS_DIR
)
from split_by_time import load_and_validate_time_data, split_by_time, get_test_key


def fetch_acl_headers():
    acl_dest = BASE_DIR / 'third_party' / 'acl' / 'inc' / 'acl'
    acl_src = BASE_DIR / 'third_party' / 'acl_src'

    print(" --- Fetching ACL headers...")

    copied_from_submodule = False

    # 1. Try submodule source
    runtime_acl = acl_src / 'runtime' / 'include' / 'external' / 'acl'
    if runtime_acl.is_dir():
        acl_dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(runtime_acl), str(acl_dest), dirs_exist_ok=True)
        print(" --- Copied runtime acl headers")
        copied_from_submodule = True

    ge_acl = acl_src / 'ge' / 'inc' / 'external' / 'acl'
    if ge_acl.is_dir():
        acl_dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(ge_acl), str(acl_dest), dirs_exist_ok=True)
        print(" --- Copied ge acl headers")
        copied_from_submodule = True

    super_kernel_src = acl_src / 'graph-autofusion' / 'super_kernel' / 'include' / 'super_kernel' / 'super_kernel.h'
    if super_kernel_src.is_file():
        acl_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(super_kernel_src), str(acl_dest / 'super_kernel.h'))
        print(" --- Copied super_kernel.h")
        copied_from_submodule = True

    if copied_from_submodule:
        if acl_src.is_dir():
            shutil.rmtree(str(acl_src))
            print(" --- Cleaned up acl_src submodule directories")
    else:
        # 2. Fallback: copy from installed torch_npu
        try:
            import torch_npu
            installed_acl = Path(
                torch_npu.__file__).resolve().parent / 'include' / 'third_party' / 'acl' / 'inc' / 'acl'
            if installed_acl.is_dir():
                acl_dest.mkdir(parents=True, exist_ok=True)
                shutil.copytree(str(installed_acl), str(acl_dest), dirs_exist_ok=True)
                print(" --- Fallback: copied acl headers from installed torch_npu")
        except Exception as e:
            print(f" --- Fallback failed: {e}")

    print(" --- ACL headers fetched successfully")


def exec_ut(files, test_classes=None):
    """
    执行单元测试文件，其中存在失败，则标识异常并打印相关信息。

    Args:
        files: {'ut_files': [...], 'op_ut_files': [...]}
        test_classes: {'ut_files': {file_path: [class_name, ...]}, 'op_ut_files': {}}
                      当文件有类级拆分时，逐个类执行并记录类级耗时。
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
            log_queue.put(line.decode('utf-8', errors='ignore'))
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

    def run_tests(test_files, test_classes_map=None):
        test_infos = []
        success_durations = {}
        has_failed = 0
        init_method = random.randint(1, 2)
        for ut_type, ut_files in test_files.items():
            classes_map = (test_classes_map or {}).get(ut_type, {})
            for ut_file in ut_files:
                test_key = get_test_key(ut_file, ut_type)
                classes_to_run = classes_map.get(ut_file)

                if classes_to_run:
                    # 类级执行：逐个类执行并记录耗时
                    class_durations = {}
                    for class_name in classes_to_run:
                        cmd = get_ut_cmd(ut_type, ut_file)
                        if ut_type == "op_ut_files":
                            # op_ut_files 不进行类级拆分，理论上不会走到这里
                            ut_info = "test_ops _" + get_op_name(ut_file)
                        elif 'op-plugin' in str(Path(ut_file)):
                            cmd = cmd + ["--", "-k", class_name]
                            ut_info = f"{test_key}.{class_name}"
                        else:
                            cmd = cmd + ["--init_method={}".format(init_method), "--", "-k", class_name]
                            ut_info = f"{test_key}.{class_name}"
                        t_start = time.time()
                        ret = run_cmd_with_timeout(cmd)
                        elapsed = time.time() - t_start
                        duration = "{:.1f}s".format(elapsed)
                        if ret:
                            has_failed = ret
                            test_infos.append("exec ut {} failed. [{}]".format(ut_info, duration))
                        else:
                            test_infos.append("exec ut {} success. [{}]".format(ut_info, duration))
                            class_durations[class_name] = round(elapsed, 1)
                        init_method = 2 if init_method == 1 else 1
                    # 记录类级耗时数据：总耗时为当前机器上执行的各类耗时之和
                    if class_durations:
                        total_time = round(sum(class_durations.values()), 1)
                        success_durations[test_key] = {
                            'total_time': total_time,
                            'classes': class_durations
                        }
                else:
                    # 文件级执行
                    cmd = get_ut_cmd(ut_type, ut_file)
                    ut_info = str(cmd[-1])
                    if ut_type == "op_ut_files":
                        ut_info = "test_ops " + ut_info
                    else:
                        cmd = cmd if 'op-plugin' in str(Path(ut_file)) else cmd + ["--init_method={}".format(init_method)]
                    t_start = time.time()
                    ret = run_cmd_with_timeout(cmd)
                    elapsed = time.time() - t_start
                    duration = "{:.1f}s".format(elapsed)
                    if ret:
                        has_failed = ret
                        test_infos.append("exec ut {} failed. [{}]".format(ut_info, duration))
                    else:
                        test_infos.append("exec ut {} success. [{}]".format(ut_info, duration))
                        success_durations[test_key] = {
                            'total_time': round(elapsed, 1)
                        }
                    init_method = 2 if init_method == 1 else 1
        return has_failed, test_infos, success_durations

    ret_status, exec_infos, success_durations = run_tests(files, test_classes)

    print("***** Total result:")
    for exec_info in exec_infos:
        print(exec_info)

    json_file = str(BASE_DIR / 'temp_time_data.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(success_durations, f, indent=2, ensure_ascii=False)
    print(f"***** Duration data saved to: {json_file}")

    return ret_status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Control needed ut cases')
    parser.add_argument('--all', action="store_true", help='Run all testcases')
    parser.add_argument('--distributed', action="store_true", help='Run distributed testcases')
    parser.add_argument('--inductor', action="store_true", help='Run inductor testcases')
    parser.add_argument('--inductor_a5', action="store_true", help='Run inductor A5 testcases')
    parser.add_argument('--rank', default=0, type=int, help='Index of current ut nodes')
    parser.add_argument('--world_size', default=0, type=int, help='Number of ut nodes')
    parser.add_argument('--npu_core', help='Run core testcases in npu')
    parser.add_argument('--network_ops', action="store_true", help='Run network_ops testcases in the op-plugin repo')
    options = parser.parse_args()
    print(f"options: {options}")
    fetch_acl_headers()
    cur_modify_files = str(BASE_DIR / 'modify_files.txt')
    test_mgr = TestMgr()

    if options.all:
        test_mgr.load_all_ut(options.distributed, options.network_ops)
    elif options.distributed:
        test_mgr.load_distributed_ut()
    elif options.network_ops:
        test_mgr.load_op_plugin_ut()
    elif options.inductor:
        test_mgr.load_inductor_ut()
    elif options.inductor_a5:
        test_mgr.load_inductor_a5_ut()
    elif os.path.exists(cur_modify_files):
        test_mgr.load(cur_modify_files, world_size=options.world_size)
        test_mgr.analyze()
    else:
        test_mgr.load_core_ut()
    test_mgr.exclude_test_files(not_run_files=NOT_RUN_DIRECTLY, mode="not_run_directly")
    common_files = str(BASE_DIR / 'common_files.txt')
    if os.path.exists(common_files):
        test_mgr.exclude_files_from_list(common_files)

    if options.rank > 0 and options.world_size > 0:
        time_data_file = str(BASE_DIR / 'time_data.json')
        timedata = load_and_validate_time_data(time_data_file)
        if timedata is not None:
            print("time_data.json loaded successfully, splitting by time data")
            split_result = split_by_time(
                test_mgr.test_files, timedata, options.rank, options.world_size
            )
            test_mgr.test_files = {
                'ut_files': split_result['ut_files'],
                'op_ut_files': split_result['op_ut_files']
            }
            test_mgr.test_classes = {
                'ut_files': split_result['ut_classes'],
                'op_ut_files': split_result['op_ut_classes']
            }
        else:
            print("time_data.json not valid or not exist, falling back to round-robin split")
            test_mgr.split_test_files(options.rank, options.world_size)
    cur_test_files = test_mgr.get_test_files()

    if options.npu_core in ("yes", "no"):
        npu_dir = str(TEST_DIR / "npu")
        for ut_type in list(cur_test_files.keys()):
            if options.npu_core == "yes":
                cur_test_files[ut_type] = [f for f in cur_test_files[ut_type]
                                           if str(Path(f)).startswith(npu_dir)]
            else:
                cur_test_files[ut_type] = [f for f in cur_test_files[ut_type]
                                           if not str(Path(f)).startswith(npu_dir)]

    test_mgr.print_modify_files()
    test_mgr.print_ut_files()
    test_mgr.print_op_ut_files()

    ret_ut = exec_ut(cur_test_files, test_mgr.test_classes)
    sys.exit(ret_ut)
