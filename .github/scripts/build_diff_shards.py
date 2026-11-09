#!/usr/bin/env python3
"""Build cross-platform blacklist diff shards.

Sets (nodeid level, deduplicated through the same loader the pipelines use):

    a3only = a3 blacklist - a5 blacklist
    a5only = a5 blacklist - a3 blacklist

Every set is emitted for BOTH platforms (``diff-<set>-<platform>``), because the
sync check needs the result of the same case on the platform that does not
blacklist it yet.  Both sides are re-run so a single workflow run contains all
the evidence; no result from an earlier run is depended on.

Output layout (one directory per category, uploaded as one artifact each):

    <output-dir>/diff-a3only-a3/diff-a3only-a3_cases_shard_<n>.json
    <output-dir>/diff-a3only-a5/...
    <output-dir>/diff-a5only-a3/...
    <output-dir>/diff-a5only-a5/...
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

BLACKLIST_FILES = [
    "disabled_testcases_{platform}_all.json",
    "unsupported_aclnn_testcases.json",
    "unsupport_dtype_testcases.json",
]

SETS = ("a3only", "a5only")
PLATFORMS = ("a3", "a5")


def load_collect_module(scripts_dir: Path):
    path = scripts_dir / "collect_all_cases.py"
    spec = importlib.util.spec_from_file_location("collect_all_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_blacklist(module, config_dir: Path, platform: str):
    files = [str(config_dir / name.format(platform=platform)) for name in BLACKLIST_FILES]
    return module.load_skip_list(files, platform=platform) or {}


def write_shards(nodeids, category, output_dir: Path, num_shards: int):
    out_dir = output_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, num_shards + 1):
        cases = [
            {"nodeid": nodeid, "file": nodeid.split("::", 1)[0]}
            for nodeid in nodeids[i - 1 :: num_shards]
        ]
        data = {
            "shard": i,
            "num_shards": num_shards,
            "test_type": "regular",
            "total_cases": len(cases),
            "cases": cases,
        }
        (out_dir / f"{category}_cases_shard_{i}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=".github/config")
    parser.add_argument("--scripts-dir", default=".github/scripts")
    parser.add_argument("--output-dir", default="cases-shards")
    parser.add_argument("--num-shards", type=int, default=4)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.num_shards < 1:
        raise SystemExit("num_shards must be >= 1")

    module = load_collect_module(Path(args.scripts_dir))
    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)

    by_platform = {p: set(load_blacklist(module, config_dir, p)) for p in PLATFORMS}
    diff = {
        "a3only": by_platform["a3"] - by_platform["a5"],
        "a5only": by_platform["a5"] - by_platform["a3"],
    }

    lines = [
        "### Cross-platform diff shards",
        "",
        f"- a3 blacklist: {len(by_platform['a3'])}, a5 blacklist: {len(by_platform['a5'])}",
        f"- a3only (missing on a5): {len(diff['a3only'])}",
        f"- a5only (missing on a3): {len(diff['a5only'])}",
        "",
        "| category | cases | shards |",
        "|---|---|---|",
    ]
    for set_name in SETS:
        nodeids = sorted(diff[set_name])
        for platform in PLATFORMS:
            category = f"diff-{set_name}-{platform}"
            write_shards(nodeids, category, output_dir, args.num_shards)
            lines.append(f"| {category} | {len(nodeids)} | {args.num_shards} |")

    summary = "\n".join(lines)
    print(summary)

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as f:
            f.write(summary + "\n")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"shards_json={json.dumps(list(range(1, args.num_shards + 1)))}\n")
            f.write(f"num_shards={args.num_shards}\n")


if __name__ == "__main__":
    main()
