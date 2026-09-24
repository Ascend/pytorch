#!/usr/bin/env python3
"""
扫描分布式 nodeid，按卡数归类并输出为 shard JSON 或返回分组 dict。

原理:
  解析 @skipIfUnsupportMultiNPU(N) 和 world_size 属性，取最大值作为卡数要求。

两种使用方式:
  1. 作为库 import (供 collect_all_cases.py 调用):
       from classify_distributed_cards import group_cases_by_card_requirement
       groups = group_cases_by_card_requirement(dist_cases, test_dir)
       # groups = {
       #     "2card":        [{"nodeid": ..., "file": ...}, ...],
       #     "4card":        [...],
       #     "8card":        [...],
       #     "unclassified": [...],
       # }

  2. CLI 独立运行 (输出 shard JSON):
       cat nodeids.txt | python classify_distributed_cards.py \
           --test-dir test --output-dir cases-shards --shard-count 5

      每组内均分到 min(shard-count, case-count) 个 shard，
      输出格式与 collect_all_cases.py 的 save_shards 一致。

分类:
  2card   → distributed_2card_cases_shard_*.json
  4card   → distributed_4card_cases_shard_*.json
  8card   → distributed_8card_cases_shard_*.json
  unclassified → distributed_cases_shard_*.json (兜底 16 卡)
"""

import argparse
import ast
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# ==============================================================================
# 正则
# ==============================================================================

DECORATOR_PATTERN = re.compile(r"skipIfUnsupportMultiNPU\((\d+)\)")
WORLD_SIZE_PATTERN = re.compile(r"return\s+(?:min\(\s*)?(\d+)")

# ==============================================================================
# 文件级缓存
# ==============================================================================

_cache_decorator: Dict[str, Dict[str, int]] = {}
_cache_worldsize: Dict[str, Dict[str, int]] = {}


def parse_decorator_requirements(file_path: Path) -> Dict[str, int]:
    if not file_path.exists():
        return {}
    try:
        source = file_path.read_text(encoding="utf-8")
    except Exception:
        return {}
    method_map: Dict[str, int] = {}
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        for decorator in node.decorator_list:
            seg = ast.get_source_segment(source, decorator)
            if seg is None:
                continue
            m = DECORATOR_PATTERN.search(seg)
            if m:
                method_map[node.name] = int(m.group(1))
                break
    return method_map


def parse_worldsize_requirements(file_path: Path) -> Dict[str, int]:
    if not file_path.exists():
        return {}
    try:
        source = file_path.read_text(encoding="utf-8")
    except Exception:
        return {}
    class_map: Dict[str, int] = {}
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not node.name.startswith("Test"):
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name != "world_size":
                continue
            for stmt in item.body:
                seg = ast.get_source_segment(source, stmt)
                if seg is None:
                    continue
                m = WORLD_SIZE_PATTERN.search(seg)
                if m:
                    class_map[node.name] = int(m.group(1))
                    break
            break
    return class_map


# ==============================================================================
# 单条分类
# ==============================================================================

VALID_CARD_COUNTS = {2, 4, 8}

def classify_one(nodeid: str, test_dir: Path) -> str:
    """输入一条 nodeid，返回卡数标签 (如 "2card", "unclassified")。"""
    parts = nodeid.split("::")
    if len(parts) < 2:
        return "unclassified"

    file_part = parts[0]
    class_name = parts[1] if len(parts) >= 2 else ""
    method_name = parts[-1]

    if not file_part.startswith("test/"):
        file_part = "test/" + file_part

    full = test_dir.parent / file_part
    cards = 0

    if file_part not in _cache_decorator:
        _cache_decorator[file_part] = parse_decorator_requirements(full)
    cards = max(cards, _cache_decorator[file_part].get(method_name, 0))

    if file_part not in _cache_worldsize:
        _cache_worldsize[file_part] = parse_worldsize_requirements(full)
    cards = max(cards, _cache_worldsize[file_part].get(class_name, 0))

    if cards not in VALID_CARD_COUNTS:
        return "unclassified"
    return f"{cards}card"


# ==============================================================================
# 库入口 — 供 collect_all_cases.py 直接调用
# ==============================================================================

def group_cases_by_card_requirement(cases: List[Dict], test_dir: Path) -> Dict[str, List[Dict]]:
    """
    输入 dist_cases 格式的用例列表，返回按卡数分组的 dict。

    cases 格式 (与 collect_all_cases.py 的 dist_cases 一致):
      [{"nodeid": "distributed/xxx.py::TestX::test_y", "file": "test/distributed/xxx.py"}, ...]

    返回格式:
      {
          "2card":        [{"nodeid": ..., "file": ...}, ...],
          "4card":        [...],
          "8card":        [...],
          "unclassified": [...],
      }
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for case in cases:
        nodeid = case.get("nodeid", "")
        if not nodeid:
            continue
        label = classify_one(nodeid, test_dir)
        # 保持原有字段不变，兼容 save_shards / save_cases_by_file
        groups[label].append(case)

    return dict(groups)


# ==============================================================================
# 分片 (CLI 用)
# ==============================================================================

def split_into_shards(cases: List[Dict], num_shards: int) -> List[List[Dict]]:
    total = len(cases)
    if total == 0:
        return [[] for _ in range(num_shards)]
    base = total // num_shards
    rem = total % num_shards
    shards = []
    start = 0
    for i in range(num_shards):
        size = base + (1 if i < rem else 0)
        shards.append(cases[start : start + size])
        start += size
    return shards


def write_shard_json(cases: List[Dict], test_type: str, shard_idx: int,
                     num_shards: int, output_dir: Path) -> int:
    if not cases:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_file = output_dir / f"{test_type}_cases_shard_{shard_idx}.json"
    data = {
        "shard": shard_idx,
        "num_shards": num_shards,
        "test_type": "distributed",     # 固定: 执行引擎据此走串行独占模式
        "total_cases": len(cases),
        "cases": cases,
    }
    shard_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(cases)


# ==============================================================================
# CLI 入口
# ==============================================================================

CARD_TO_TYPE = {
    "2card": "distributed_2card",
    "3card": "distributed_3card",
    "4card": "distributed_4card",
    "8card": "distributed_8card",
    "unclassified": "distributed",
}


def main():
    parser = argparse.ArgumentParser(description="Classify distributed nodeids and output shard JSONs")
    parser.add_argument("--test-dir", required=True, help="Path to test/ directory")
    parser.add_argument("--input", "-i", help="Input file of nodeids (one per line). If omitted, read from stdin.")
    parser.add_argument("--output-dir", required=True, help="Output directory for shard JSON files")
    parser.add_argument("--shard-count", type=int, default=5,
                        help="Target machines per card group. Actual shards = min(shard-count, case-count).")
    args = parser.parse_args()

    test_dir = Path(args.test_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.input:
        lines = Path(args.input).read_text(encoding="utf-8").strip().splitlines()
    else:
        lines = [line.strip() for line in sys.stdin if line.strip()]

    # 通过 group_cases_by_card_requirement 复用同一套分类逻辑
    cases_list = []
    for line in lines:
        if not line.strip():
            continue
        nodeid = line.strip()
        file_part = nodeid.split("::")[0]
        if not file_part.startswith("test/"):
            file_part = "test/" + file_part
        cases_list.append({"nodeid": nodeid, "file": file_part})

    groups = group_cases_by_card_requirement(cases_list, test_dir)

    # 分片输出
    machine_count = args.shard_count
    summary = {}
    for label, cases in sorted(groups.items()):
        test_type = CARD_TO_TYPE.get(label, f"distributed_{label}")
        case_count = len(cases)
        num_shards = min(machine_count, case_count) if case_count > 0 else 1
        shards = split_into_shards(cases, num_shards)
        total = 0
        for i, shard_cases in enumerate(shards, 1):
            n = write_shard_json(shard_cases, test_type, i, len(shards), output_dir)
            total += n
        summary[test_type] = {"total_cases": total, "num_shards": len(shards)}
        print(f"[{test_type}] {total} cases → {len(shards)} shard(s) (target machines: {machine_count})")

    # 汇总
    summary_file = output_dir / "distributed_cases_collection_summary.json"
    summary_file.write_text(json.dumps({
        "card_groups": summary,
        "total_cases": sum(s["total_cases"] for s in summary.values()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary → {summary_file}")

    print("\nGenerated shard files:")
    for f in sorted(output_dir.glob("distributed*_cases_shard_*.json")):
        print(f"  {f.name} ({os.path.getsize(f)} bytes)")


if __name__ == "__main__":
    main()
