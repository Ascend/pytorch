#!/usr/bin/env python3
"""
性能对比测试：验证文件写入对测试执行速度的影响
"""

import subprocess
import time
import sys
from pathlib import Path

def run_single_case_with_write(nodeid: str, test_dir: Path, report_dir: Path):
    """带文件写入的执行（当前方式）"""
    xml_file = report_dir / "junit_xmls" / f"test_{nodeid.replace('/', '_')}.xml"
    log_file = report_dir / "cases_logs" / f"test_{nodeid.replace('/', '_')}.log"

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "junit_xmls").mkdir(parents=True, exist_ok=True)
    (report_dir / "cases_logs").mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    subprocess.run([
        sys.executable, "-m", "pytest",
        "-v", "--tb=short",
        nodeid,
        f"--junitxml={xml_file}",
    ], cwd=test_dir, capture_output=True)
    duration = time.monotonic() - start

    # 写入log
    log_file.write_text(f"Test: {nodeid}\nDuration: {duration}s\n")

    return duration


def run_single_case_no_write(nodeid: str, test_dir: Path):
    """不写入文件的执行（对比方式）"""
    start = time.monotonic()
    subprocess.run([
        sys.executable, "-m", "pytest",
        "-v", "--tb=short",
        nodeid,
    ], cwd=test_dir, capture_output=True)
    duration = time.monotonic() - start
    return duration


def benchmark(test_dir: Path, nodeid: str, iterations: int = 10):
    """对比测试"""
    report_dir = Path("/tmp/benchmark_write")

    print(f"测试用例: {nodeid}")
    print(f"迭代次数: {iterations}")
    print(f"测试目录: {test_dir}")
    print()

    # 测试1: 带文件写入
    print("=== 带文件写入 ===")
    times_with_write = []
    for i in range(iterations):
        t = run_single_case_with_write(nodeid, test_dir, report_dir)
        times_with_write.append(t)
        print(f"  [{i+1}] {t:.2f}s")
    avg_with_write = sum(times_with_write) / len(times_with_write)

    # 测试2: 不写入文件
    print("\n=== 不写入文件 ===")
    times_no_write = []
    for i in range(iterations):
        t = run_single_case_no_write(nodeid, test_dir)
        times_no_write.append(t)
        print(f"  [{i+1}] {t:.2f}s")
    avg_no_write = sum(times_no_write) / len(times_no_write)

    # 结果对比
    print("\n=== 对比结果 ===")
    print(f"带文件写入平均时间: {avg_with_write:.2f}s")
    print(f"不写入文件平均时间: {avg_no_write:.2f}s")
    print(f"写入开销: {avg_with_write - avg_no_write:.2f}s ({(avg_with_write/avg_no_write-1)*100:.1f}% 增加)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="文件写入性能对比测试")
    parser.add_argument("--test-dir", required=True, help="PyTorch test directory")
    parser.add_argument("--nodeid", default="test/test_torch.py::TestTorch::test_abs", help="Test case nodeid")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    args = parser.parse_args()

    benchmark(Path(args.test_dir), args.nodeid, args.iterations)