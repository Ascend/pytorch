#!/usr/bin/env python3
"""
Flatten the grouped blacklist (sample-D) back into a plain nodeid list.

Reads ``disabled_testcases_grouped.json`` and writes every ``nodes`` entry from
every group, one nodeid per line, to the output file. The grouped source file
is left untouched.

This is a read-only "flattened for reference" companion used by the
blacklist-test workflow; it does NOT feed any shard or test job (those still
read the flat per-platform blacklist files).
"""

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten a grouped blacklist JSON into a plain nodeid list."
    )
    parser.add_argument("--input", required=True, help="path to the grouped JSON file")
    parser.add_argument("--output", required=True, help="path to write the nodeid list to")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    groups = data.get("groups", [])
    nodeids = []
    for group in groups:
        nodeids.extend(group.get("nodes", []))

    with open(args.output, "w", encoding="utf-8") as f:
        for nodeid in nodeids:
            f.write(f"{nodeid}\n")

    print(f"flattened {len(nodeids)} nodeids from {len(groups)} groups -> {args.output}")


if __name__ == "__main__":
    main()