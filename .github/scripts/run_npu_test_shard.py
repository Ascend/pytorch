#!/usr/bin/env python3
"""
NPU 测试分片执行脚本
功能：
1. 加载 disabled_testcases.json 禁用测试列表
2. 发现测试文件并分片
3. 执行 pytest 并生成 JUnit XML 报告
4. 输出测试统计信息
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import List, Set, Dict, Tuple


EXCLUDED_TEST_DIRS = ('custom_ops', 'distributions', 'jit')


def parse_args():
    parser = argparse.ArgumentParser(description='Run PyTorch NPU tests for a shard')
    parser.add_argument('--shard', type=int, required=True,
                        help='Shard number (1-indexed)')
    parser.add_argument('--num-shards', type=int, required=True,
                        help='Total number of shards')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Path to PyTorch test directory')
    parser.add_argument('--disabled-testcases', type=str,
                        help='Path to disabled_testcases.json')
    parser.add_argument('--report-dir', type=str, default='test-reports',
                        help='Directory for test reports')
    parser.add_argument('--timeout', type=int, default=600,
                        help='Timeout per test in seconds')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    return parser.parse_args()


def load_disabled_testcases(json_file: str) -> Set[str]:
    """加载 disabled_testcases.json 中的禁用测试用例"""
    disabled = set()
    if json_file and os.path.exists(json_file):
        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)
            # JSON 格式: {"test_name": ["reason", ["issues"]], ...}
            disabled = set(data.keys())
            print(f"Loaded {len(disabled)} disabled test cases from {json_file}")
    return disabled


def discover_test_files(test_dir: str) -> Tuple[List[str], List[str]]:
    """发现所有 test_*.py 文件，并记录因目录规则被排除的文件"""
    test_path = Path(test_dir)
    test_files = []
    excluded_test_files = []

    # 查找所有 test_*.py 文件（排除某些特殊目录）
    for test_file in test_path.rglob('test_*.py'):
        rel_path = test_file.relative_to(test_path)
        # 检查是否在排除目录中
        if any(part in EXCLUDED_TEST_DIRS for part in rel_path.parts):
            excluded_test_files.append(str(rel_path))
            continue
        test_files.append(str(test_file))

    return sorted(test_files), sorted(excluded_test_files)


def shard_tests(tests: List[str], shard: int, num_shards: int) -> List[str]:
    """将测试文件均匀分片"""
    if num_shards <= 1:
        return tests

    total = len(tests)
    base_size = total // num_shards
    remainder = total % num_shards

    start = 0
    for i in range(1, shard):
        start += base_size + (1 if i <= remainder else 0)

    current_size = base_size + (1 if shard <= remainder else 0)
    return tests[start:start + current_size]


def parse_junit_xml(xml_file: str) -> Dict:
    """解析 JUnit XML 报告，返回测试统计"""
    stats = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'errors': 0,
        'duration': 0.0
    }

    if not os.path.exists(xml_file):
        return stats

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 查找所有 testsuite
        for testsuite in root.iter('testsuite'):
            stats['total'] += int(testsuite.get('tests', 0))
            stats['failed'] += int(testsuite.get('failures', 0))
            stats['skipped'] += int(testsuite.get('skipped', 0))
            stats['errors'] += int(testsuite.get('errors', 0))
            stats['duration'] += float(testsuite.get('time', 0))

        # passed = total - failed - skipped - errors
        stats['passed'] = stats['total'] - stats['failed'] - stats['skipped'] - stats['errors']

    except Exception as e:
        print(f"Warning: Failed to parse XML report: {e}")

    return stats


def create_empty_stats() -> Dict:
    """创建空的测试统计信息"""
    return {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'errors': 0,
        'duration': 0.0
    }


def create_shard_info(shard: int, num_shards: int, timestamp: str) -> Dict:
    """创建分片信息的默认结构"""
    return {
        'shard': shard,
        'num_shards': num_shards,
        'total_files': 0,
        'shard_files': 0,
        'excluded_dirs': list(EXCLUDED_TEST_DIRS),
        'excluded_test_files': 0,
        'disabled_count': 0,
        'disabled_count_matched': 0,
        'disabled_count_deselected': 0,
        'timestamp': timestamp,
    }


def finalize_stats(
    xml_report: str,
    returncode: int,
    fallback_duration: float = 0.0,
    error_message: str = '',
    timed_out: bool = False,
) -> Dict:
    """解析 XML 并补齐兜底统计字段"""
    has_xml_report = os.path.exists(xml_report)
    stats = parse_junit_xml(xml_report)
    if not stats:
        stats = create_empty_stats()

    if timed_out and fallback_duration > 0:
        stats['duration'] = max(stats.get('duration', 0.0), fallback_duration)
    elif stats.get('duration', 0.0) <= 0 and fallback_duration > 0:
        stats['duration'] = fallback_duration

    if timed_out:
        stats['errors'] = max(stats.get('errors', 0), 1)
        stats['timed_out'] = True

    if returncode < 0:
        signal_num = abs(returncode)
        try:
            signal_name = signal.Signals(signal_num).name
        except ValueError:
            signal_name = f"SIG{signal_num}"
        stats['crashed'] = True
        stats['crash_signal'] = signal_name

        if not error_message:
            error_message = f"Pytest process crashed with signal {signal_name}"

    if returncode != 0 and not has_xml_report:
        stats['errors'] = max(stats.get('errors', 0), 1)
        stats['incomplete'] = True
        if not error_message:
            error_message = (
                "Pytest exited before producing a JUnit XML report"
            )

    if error_message:
        stats['error_message'] = error_message

    stats['returncode'] = returncode
    return stats


def print_stats_summary(shard: int, stats: Dict) -> None:
    """输出测试统计信息"""
    print(f"\n{'='*60}")
    print(f"Test Results for Shard {shard}")
    print(f"{'='*60}")
    print(f"Total:  {stats['total']}")
    print(f"Passed: {stats['passed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print(f"Duration: {stats['duration']:.2f}s")
    print(f"{'='*60}")


def save_stats_file(report_dir: str, shard: int, stats: Dict) -> str:
    """保存测试统计到 JSON 文件"""
    os.makedirs(report_dir, exist_ok=True)
    stats_file = os.path.join(report_dir, f'shard_{shard}_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    return stats_file


def save_info_file(report_dir: str, shard: int, info: Dict) -> str:
    """保存分片信息到 JSON 文件"""
    os.makedirs(report_dir, exist_ok=True)
    info_file = os.path.join(report_dir, f'shard_{shard}_info.json')
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    return info_file


def get_disabled_testcases_report_file(report_dir: str, shard: int) -> str:
    """获取当前分片的 disabled testcases 命中统计文件路径"""
    return os.path.join(report_dir, f'shard_{shard}_disabled_testcases.json')


def load_disabled_testcases_report(report_dir: str, shard: int) -> Dict:
    """读取当前分片的 disabled testcases 命中统计"""
    report_file = get_disabled_testcases_report_file(report_dir, shard)
    if not os.path.exists(report_file):
        return {
            'disabled_count_matched': 0,
            'disabled_count_deselected': 0,
        }

    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {
            'disabled_count_matched': data.get('disabled_count_matched', 0),
            'disabled_count_deselected': data.get('disabled_count_deselected', 0),
        }
    except Exception as e:
        print(f"Warning: Failed to read disabled testcase report: {e}")
        return {
            'disabled_count_matched': 0,
            'disabled_count_deselected': 0,
        }


def format_subprocess_output(output) -> str:
    """将 subprocess 输出统一转换为字符串"""
    if not output:
        return ''
    if isinstance(output, bytes):
        return output.decode('utf-8', errors='replace')
    return str(output)


def log_print(message: str = '') -> None:
    """统一打印日志并立即刷新，避免长时间无输出"""
    print(message, flush=True)


def build_test_targets(test_files: List[str], test_dir: str) -> List[str]:
    """将测试文件转换为 pytest 目标路径"""
    test_dir_path = Path(test_dir).resolve()
    targets = []

    for tf in test_files:
        tf_path = Path(tf).resolve()
        try:
            rel_path = tf_path.relative_to(test_dir_path)
            targets.append(str(rel_path))
        except ValueError:
            # 如果不在 test_dir 下，使用绝对路径
            targets.append(str(tf_path))

    return targets


def save_test_plan_file(report_dir: str, shard: int, test_targets: List[str]) -> str:
    """保存当前分片的测试计划文件列表"""
    os.makedirs(report_dir, exist_ok=True)
    plan_file = os.path.join(report_dir, f'shard_{shard}_planned_test_files.txt')
    with open(plan_file, 'w', encoding='utf-8') as f:
        for target in test_targets:
            f.write(f"{target}\n")
    return plan_file


def save_excluded_test_files_file(report_dir: str, shard: int, test_targets: List[str]) -> str:
    """保存因目录规则被排除的测试文件列表"""
    os.makedirs(report_dir, exist_ok=True)
    excluded_file = os.path.join(report_dir, f'shard_{shard}_excluded_test_files.txt')
    with open(excluded_file, 'w', encoding='utf-8') as f:
        for target in test_targets:
            f.write(f"{target}\n")
    return excluded_file


def print_test_plan(shard: int, test_targets: List[str], plan_file: str) -> None:
    """在执行前打印当前分片计划执行的测试文件"""
    log_print(f"Planned test files for shard {shard} ({len(test_targets)} files):")
    for index, target in enumerate(test_targets, 1):
        log_print(f"  [{index:03d}] {target}")
    log_print(f"Saved planned test file list to: {plan_file}")
    log_print()


def run_pytest(
    test_files: List[str],
    test_dir: str,
    disabled_testcases_file: str,
    report_dir: str,
    shard: int,
    timeout: int,
    verbose: bool
) -> Dict:
    """执行 pytest 并返回测试统计"""
    # 转换为绝对路径
    report_dir = os.path.abspath(report_dir)
    test_dir = os.path.abspath(test_dir)
    test_dir_path = Path(test_dir).resolve()
    os.makedirs(report_dir, exist_ok=True)

    xml_report = os.path.join(report_dir, f'junit_shard_{shard}.xml')
    log_file = os.path.join(report_dir, f'test_shard_{shard}.log')
    overall_timeout = 7200

    # 构建 pytest 命令
    cmd = [
        sys.executable, '-m', 'pytest',
        '-v' if verbose else '-q',
        f'--junit-xml={xml_report}',
        '--tb=short',
        f'--timeout={timeout}',
        '-p', 'no:xdist',  # 禁用 xdist（单进程）
        '-p', 'pytest_disabled_testcases_plugin',
        '--durations=50',
    ]

    test_targets = build_test_targets(test_files, test_dir)
    cmd.extend(test_targets)
    plan_file = save_test_plan_file(report_dir, shard, test_targets)

    log_print(f"\n{'='*60}")
    log_print(f"Running shard {shard}: {len(test_files)} test files")
    log_print(f"Working directory: {test_dir}")
    log_print(f"{'='*60}\n")
    print_test_plan(shard, test_targets, plan_file)

    # --------------------------------------------------------------------
    # 关键：Python 路径优先级
    # 1. 已安装的 torch/torch_npu（pip site-packages）优先
    # 2. 不把源码根目录加入 PYTHONPATH（避免导入未编译的 torch）
    # 3. 只添加必要的测试辅助路径
    # --------------------------------------------------------------------
    env = os.environ.copy()

    # 获取已安装 torch 的路径，确保优先级最高
    import torch
    torch_path = Path(torch.__file__).parent.parent  # torch 包的父目录 (site-packages)
    script_dir = Path(__file__).resolve().parent

    # 构建 PYTHONPATH：
    # - 已安装的 torch 路径放在最前面
    # - test 目录（用于测试工具模块如 common_utils）
    # - 脚本目录（用于 pytest plugin）
    # - 不包含源码 torch/ 目录
    existing_pythonpath = env.get('PYTHONPATH', '')

    # 优先级：已安装 torch > test 目录 > 脚本目录 > 现有 PYTHONPATH
    pythonpath_parts = [str(torch_path), str(test_dir_path), str(script_dir)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    env['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)
    env['PYTORCH_TEST_NPU'] = '1'
    env['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '1'  # 允许加载 torch_npu backend
    if disabled_testcases_file:
        env['NPU_DISABLED_TESTCASES_JSON'] = os.path.abspath(disabled_testcases_file)
        env['NPU_DISABLED_TESTCASES_REPORT'] = os.path.abspath(
            get_disabled_testcases_report_file(report_dir, shard)
        )

    log_print(f"PYTHONPATH priority: {torch_path} (installed torch)")

    with open(log_file, 'w', encoding='utf-8') as log:
        log.write(f"Test execution started at {datetime.now()}\n")
        log.write(f"Test files: {len(test_files)}\n\n")
        for index, target in enumerate(test_targets, 1):
            log.write(f"[{index:03d}] {target}\n")
        log.write("\n")
        log.flush()

        start_time = monotonic()
        try:
            result = subprocess.run(
                cmd,
                cwd=test_dir,  # 工作目录设为 test 目录
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=overall_timeout
            )
            log.write(result.stdout)
            log_print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)

            stats = finalize_stats(
                xml_report,
                returncode=result.returncode,
                fallback_duration=monotonic() - start_time
            )
        except subprocess.TimeoutExpired as e:
            partial_output = format_subprocess_output(e.stdout)
            if partial_output:
                log.write(partial_output)
                if not partial_output.endswith('\n'):
                    log.write('\n')

            timeout_message = (
                f"Shard {shard} exceeded overall timeout of {overall_timeout} seconds"
            )
            log.write(f"\n{timeout_message}\n")
            log_print(timeout_message)
            if partial_output:
                log_print(partial_output[-5000:] if len(partial_output) > 5000 else partial_output)

            stats = finalize_stats(
                xml_report,
                returncode=124,
                fallback_duration=monotonic() - start_time,
                error_message=timeout_message,
                timed_out=True,
            )
        except Exception as e:
            error_trace = traceback.format_exc()
            log.write("\nUnexpected error while running pytest:\n")
            log.write(error_trace)

            error_message = f"Unexpected error while running pytest: {e}"
            log_print(error_message)
            log_print(error_trace[-5000:] if len(error_trace) > 5000 else error_trace)

            stats = finalize_stats(
                xml_report,
                returncode=1,
                fallback_duration=monotonic() - start_time,
                error_message=error_message,
            )
            stats['errors'] = max(stats.get('errors', 0), 1)

    print_stats_summary(shard, stats)
    return stats


def main():
    args = parse_args()
    timestamp = datetime.now().isoformat()
    info = create_shard_info(args.shard, args.num_shards, timestamp)

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(line_buffering=True)

    log_print(f"\n{'='*60}")
    log_print("PyTorch NPU Test Shard Runner")
    log_print(f"{'='*60}")
    log_print(f"Shard: {args.shard}/{args.num_shards}")
    log_print(f"Test directory: {args.test_dir}")
    log_print(f"Disabled testcases: {args.disabled_testcases}")
    log_print(f"{'='*60}\n")

    try:
        # Step 1: 加载禁用测试用例
        disabled = load_disabled_testcases(args.disabled_testcases)
        info['disabled_count'] = len(disabled)

        # Step 2: 发现测试文件
        all_test_files, excluded_test_files = discover_test_files(args.test_dir)
        log_print(f"Discovered {len(all_test_files)} test files")
        info['total_files'] = len(all_test_files)
        info['excluded_test_files'] = len(excluded_test_files)
        if excluded_test_files:
            excluded_file = save_excluded_test_files_file(
                args.report_dir,
                args.shard,
                excluded_test_files,
            )
            log_print(
                f"Excluded {len(excluded_test_files)} test files by directory rules: {excluded_file}"
            )

        # Step 3: 分片
        sharded_tests = shard_tests(all_test_files, args.shard, args.num_shards)
        log_print(f"Shard {args.shard} contains {len(sharded_tests)} test files")
        info['shard_files'] = len(sharded_tests)

        if not sharded_tests:
            log_print("No tests to run for this shard")
            # 输出空结果
            stats = create_empty_stats()
            stats['returncode'] = 0
            save_info_file(args.report_dir, args.shard, info)
            save_stats_file(args.report_dir, args.shard, stats)
            return 0

        # Step 4: 执行测试
        stats = run_pytest(
            sharded_tests,
            args.test_dir,
            args.disabled_testcases,
            args.report_dir,
            args.shard,
            args.timeout,
            args.verbose
        )
        info.update(load_disabled_testcases_report(args.report_dir, args.shard))
    except Exception as e:
        error_trace = traceback.format_exc()
        log_print(f"Fatal error in shard runner: {e}")
        log_print(error_trace[-5000:] if len(error_trace) > 5000 else error_trace)

        stats = create_empty_stats()
        stats['errors'] = 1
        stats['returncode'] = 1
        stats['error_message'] = f"Fatal error in shard runner: {e}"
        stats['fatal_runner_error'] = True

    # Step 5: 保存分片信息和测试统计（供 workflow 读取）
    save_info_file(args.report_dir, args.shard, info)
    save_stats_file(args.report_dir, args.shard, stats)
    return stats.get('returncode', 1)


if __name__ == '__main__':
    sys.exit(main())
