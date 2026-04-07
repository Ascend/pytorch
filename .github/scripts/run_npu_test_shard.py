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
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Set, Dict


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
        with open(json_file) as f:
            data = json.load(f)
            # JSON 格式: {"test_name": ["reason", ["issues"]], ...}
            disabled = set(data.keys())
            print(f"Loaded {len(disabled)} disabled test cases from {json_file}")
    return disabled


def discover_test_files(test_dir: str) -> List[str]:
    """发现所有 test_*.py 文件"""
    test_path = Path(test_dir)
    test_files = []

    # 查找所有 test_*.py 文件（排除某些特殊目录）
    exclude_dirs = {'distributions', 'custom_ops', 'jit'}

    for test_file in test_path.rglob('test_*.py'):
        rel_path = test_file.relative_to(test_path)
        # 检查是否在排除目录中
        if any(part in exclude_dirs for part in rel_path.parts):
            continue
        test_files.append(str(test_file))

    return sorted(test_files)


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


def run_pytest(
    test_files: List[str],
    test_dir: str,
    report_dir: str,
    shard: int,
    timeout: int,
    verbose: bool
) -> Dict:
    """执行 pytest 并返回测试统计"""
    # 转换为绝对路径
    report_dir = os.path.abspath(report_dir)
    test_dir = os.path.abspath(test_dir)
    os.makedirs(report_dir, exist_ok=True)

    xml_report = os.path.join(report_dir, f'junit_shard_{shard}.xml')
    log_file = os.path.join(report_dir, f'test_shard_{shard}.log')

    # 构建 pytest 命令
    cmd = [
        sys.executable, '-m', 'pytest',
        '-v' if verbose else '-q',
        f'--junit-xml={xml_report}',
        '--tb=short',
        f'--timeout={timeout}',
        '-p', 'no:xdist',  # 禁用 xdist（单进程）
        '--durations=50',
    ]

    # 添加测试文件 - 使用相对于 test_dir 的相对路径
    test_dir_path = Path(test_dir).resolve()
    for tf in test_files:
        tf_path = Path(tf).resolve()
        try:
            rel_path = tf_path.relative_to(test_dir_path)
            cmd.append(str(rel_path))
        except ValueError:
            # 如果不在 test_dir 下，使用绝对路径
            cmd.append(str(tf_path))

    print(f"\n{'='*60}")
    print(f"Running shard {shard}: {len(test_files)} test files")
    print(f"Working directory: {test_dir}")
    print(f"{'='*60}\n")

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

    # 构建 PYTHONPATH：
    # - 已安装的 torch 路径放在最前面
    # - test 目录（用于测试工具模块如 common_utils）
    # - 不包含源码 torch/ 目录
    existing_pythonpath = env.get('PYTHONPATH', '')

    # 优先级：已安装 torch > test 目录 > 现有 PYTHONPATH
    new_pythonpath = str(torch_path)
    new_pythonpath += f":{str(test_dir_path)}"
    if existing_pythonpath:
        new_pythonpath += f":{existing_pythonpath}"

    env['PYTHONPATH'] = new_pythonpath
    env['PYTORCH_TEST_NPU'] = '1'
    env['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '1'  # 允许加载 torch_npu backend

    print(f"PYTHONPATH priority: {torch_path} (installed torch)")

    with open(log_file, 'w') as log:
        log.write(f"Test execution started at {datetime.now()}\n")
        log.write(f"Test files: {len(test_files)}\n\n")
        log.flush()

        result = subprocess.run(
            cmd,
            cwd=test_dir,  # 工作目录设为 test 目录
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=7200  # 整体超时 2 小时
        )

        log.write(result.stdout)

    print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)

    # 解析测试结果
    stats = parse_junit_xml(xml_report)
    stats['returncode'] = result.returncode

    # 输出统计信息
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

    return stats


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("PyTorch NPU Test Shard Runner")
    print(f"{'='*60}")
    print(f"Shard: {args.shard}/{args.num_shards}")
    print(f"Test directory: {args.test_dir}")
    print(f"Disabled testcases: {args.disabled_testcases}")
    print(f"{'='*60}\n")

    # Step 1: 加载禁用测试用例
    disabled = load_disabled_testcases(args.disabled_testcases)

    # Step 2: 发现测试文件
    all_test_files = discover_test_files(args.test_dir)
    print(f"Discovered {len(all_test_files)} test files")

    # Step 3: 分片
    sharded_tests = shard_tests(all_test_files, args.shard, args.num_shards)
    print(f"Shard {args.shard} contains {len(sharded_tests)} test files")

    if not sharded_tests:
        print("No tests to run for this shard")
        # 输出空结果
        stats = {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 0, 'duration': 0.0, 'returncode': 0}
        stats_file = os.path.join(args.report_dir, f'shard_{args.shard}_stats.json')
        os.makedirs(args.report_dir, exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        return 0

    # Step 4: 保存分片信息
    info = {
        'shard': args.shard,
        'num_shards': args.num_shards,
        'total_files': len(all_test_files),
        'shard_files': len(sharded_tests),
        'disabled_count': len(disabled),
        'timestamp': datetime.now().isoformat()
    }
    info_file = os.path.join(args.report_dir, f'shard_{args.shard}_info.json')
    os.makedirs(args.report_dir, exist_ok=True)
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)

    # Step 5: 执行测试
    stats = run_pytest(
        sharded_tests,
        args.test_dir,
        args.report_dir,
        args.shard,
        args.timeout,
        args.verbose
    )

    # Step 6: 保存测试统计到 JSON 文件（供 workflow 读取）
    stats_file = os.path.join(args.report_dir, f'shard_{args.shard}_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    return stats.get('returncode', 1)


if __name__ == '__main__':
    sys.exit(main())