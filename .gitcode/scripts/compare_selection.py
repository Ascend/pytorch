#!/usr/bin/env python3
"""对比精准选测清单和原有策略选出的测试文件。"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "ci"))

from access_control import TestMgr  # noqa: E402
from access_control.constants import BASE_DIR, TEST_DIR, SLOW_TEST_BLOCKLIST  # noqa: E402
from access_control.strategy import DirectoryMappingStrategy  # noqa: E402


def normalize(path):
    """把测试路径归一化为仓库内的绝对路径。"""
    p = path.strip()
    if "::" in p:
        p = p.split("::", 1)[0]
    if not p:
        return None
    p = Path(p)
    if p.is_absolute():
        return str(p)
    if p.parts and p.parts[0] == "test":
        return str(BASE_DIR / p)
    return str(TEST_DIR / p)


def load_precise(precise_dir):
    """读取精准选测结果；status 不是 success 时返回 None。"""
    status_file = precise_dir / "status"
    list_file = precise_dir / "test_files.txt"
    if not status_file.is_file():
        return None
    if status_file.read_text(encoding="utf-8").strip() != "success":
        return None
    files = set()
    if list_file.is_file():
        for line in list_file.read_text(encoding="utf-8").splitlines():
            normalized = normalize(line)
            if normalized:
                files.add(normalized)
    return files


def is_core(path):
    return path.startswith(str(BASE_DIR / "test/npu") + "/")


def apply_test_filters(files):
    """应用和 TestMgr.analyze() 相同的存在性、慢测试过滤。"""
    result = set()
    for f in files:
        if not Path(f).exists():
            continue
        if any(name + ".py" in f for name in SLOW_TEST_BLOCKLIST):
            continue
        result.add(f)
    return result


def run_strategy(modify_files, skip_directory_mapping=False):
    """跑一遍原策略；skip_directory_mapping 模拟正式启用时精准替代 DirectoryMapping。"""
    manager = TestMgr()
    if not Path(modify_files).is_file():
        return set()
    manager.load(modify_files, world_size=0)
    if skip_directory_mapping:
        original_identify = DirectoryMappingStrategy.identify
        DirectoryMappingStrategy.identify = lambda self, modify_file: []
        try:
            manager.analyze()
        finally:
            DirectoryMappingStrategy.identify = original_identify
    else:
        manager.analyze()
    return set(manager.get_test_files()["ut_files"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modify-files", default="modify_files.txt")
    parser.add_argument("--precise-dir", default="precise")
    parser.add_argument("--output", default="selection_comparison.txt")
    args = parser.parse_args()

    strategy_files = run_strategy(args.modify_files)

    precise_raw = load_precise(Path(args.precise_dir))
    precise_files = apply_test_filters(precise_raw) if precise_raw is not None else None

    if precise_files is None:
        formal_strategy_files = strategy_files
        formal_union = strategy_files
    else:
        formal_strategy_files = run_strategy(args.modify_files, skip_directory_mapping=True)
        formal_union = precise_files | formal_strategy_files
    formal_added = formal_union - strategy_files
    formal_removed = strategy_files - formal_union

    lines = []
    if precise_files is None:
        lines.append("precise status: failed or missing")
        lines.append(f"strategy total: {len(strategy_files)}")
        lines.append(f"strategy core files: {len([f for f in strategy_files if is_core(f)])}")
        lines.append(f"formal strategy total: {len(formal_strategy_files)}")
        lines.append(f"formal union total: {len(formal_union)}")
        lines.append("")
        lines.append("=== strategy files ===")
        lines.extend(sorted(strategy_files))
    else:
        both = precise_files & strategy_files
        precise_only = precise_files - strategy_files
        strategy_only = strategy_files - precise_files
        union = precise_files | strategy_files
        lines.append("precise status: success")
        lines.append(f"precise total (raw): {len(precise_raw)}")
        lines.append(f"precise total (filtered): {len(precise_files)}")
        lines.append(f"strategy total: {len(strategy_files)}")
        lines.append(f"precise core files: {len([f for f in precise_files if is_core(f)])}")
        lines.append(f"strategy core files: {len([f for f in strategy_files if is_core(f)])}")
        lines.append(f"both: {len(both)}")
        lines.append(f"precise only: {len(precise_only)}")
        lines.append(f"strategy only: {len(strategy_only)}")
        lines.append(f"union total: {len(union)}")
        lines.append(f"precise only (non-core): {len([f for f in precise_only if not is_core(f)])}")
        lines.append(f"strategy only (non-core): {len([f for f in strategy_only if not is_core(f)])}")
        lines.append(f"formal strategy total: {len(formal_strategy_files)}")
        lines.append(f"formal union total: {len(formal_union)}")
        lines.append(f"formal added: {len(formal_added)}")
        lines.append(f"formal removed: {len(formal_removed)}")
        lines.append("")
        lines.append("=== precise (filtered) ===")
        lines.extend(sorted(precise_files))
        lines.append("")
        lines.append("=== strategy (filtered) ===")
        lines.extend(sorted(strategy_files))
        lines.append("")
        lines.append("=== both ===")
        lines.extend(sorted(both))
        lines.append("")
        lines.append("=== precise only ===")
        lines.extend(sorted(precise_only))
        lines.append("")
        lines.append("=== strategy only ===")
        lines.extend(sorted(strategy_only))
        lines.append("")
        lines.append("=== union ===")
        lines.extend(sorted(union))
        lines.append("")
        lines.append("=== formal strategy (filtered) ===")
        lines.extend(sorted(formal_strategy_files))
        lines.append("")
        lines.append("=== formal union ===")
        lines.extend(sorted(formal_union))
        lines.append("")
        lines.append("=== formal added (precise + retained vs original) ===")
        lines.extend(sorted(formal_added))
        lines.append("")
        lines.append("=== formal removed (original only) ===")
        lines.extend(sorted(formal_removed))

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
