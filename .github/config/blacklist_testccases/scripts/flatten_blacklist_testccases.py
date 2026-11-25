#!/usr/bin/env python3
"""Flatten all blacklist files under ``blacklist_testccases/conf/`` into a JSON array.

The files in that directory use two different shapes:

  * ``*_unsupported_testcases.json`` files are a top-level list of
    ``{log, category, nodeids: {nodeid: {cann, issue}}}`` entries; the nodeids
    are the keys of each ``nodeids`` object;
  * grouped ``*_grouped.json`` files carry ``nodes`` lists under ``groups[]``
    (and, for dtype, one level deeper under ``groups[].canns[].nodes``).

This script walks every ``*.json`` file, collects every nodeid string (from a
``nodes`` list or a ``nodeids`` object key), de-duplicates and sorts the result
into a single JSON nodeid array.

Usage:

    python3 flatten_blacklist_testccases.py \
        --config-dir .github/config/blacklist_testccases \
        --platform a3 \
        --output grouped-nodeids.json
"""

import argparse
import json
from pathlib import Path


def _add(entries, nodeids):
    """Add nodeids from a ``nodes`` list (str or {"nodeid": str} entries)."""
    for entry in entries:
        if isinstance(entry, str) and entry:
            nodeids.add(entry)
        elif isinstance(entry, dict):
            nodeid = entry.get("nodeid")
            if isinstance(nodeid, str) and nodeid:
                nodeids.add(nodeid)


def _extract(obj, nodeids):
    """Recursively collect nodeids from ``nodes`` lists and ``nodeids`` dicts."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "nodes" and isinstance(value, list):
                _add(value, nodeids)
            elif key == "nodeids" and isinstance(value, dict):
                for nodeid in value:
                    if isinstance(nodeid, str) and nodeid:
                        nodeids.add(nodeid)
            else:
                _extract(value, nodeids)
    elif isinstance(obj, list):
        for value in obj:
            _extract(value, nodeids)


def collect_nodeids(config_dir):
    """Return (sorted unique nodeids, per-file stats).

    ``stats`` is a list of (filename, kind, count) where kind is "nodes"
    (contributing file) or "parse-error".
    """
    config_dir = Path(config_dir)
    if config_dir.is_dir():
        paths = sorted(config_dir.glob("**/*.json"))
    else:
        paths = []

    nodeids = set()
    stats = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            stats.append((path.name, "parse-error", str(exc)))
            continue
        before = len(nodeids)
        _extract(data, nodeids)
        stats.append((path.name, "nodes", len(nodeids) - before))

    return sorted(nodeids), stats


def _nodeid_strings(entries):
    """Yield nodeid strings from a ``nodes`` list (str or {"nodeid": ...})."""
    for entry in entries:
        if isinstance(entry, str) and entry:
            yield entry
        elif isinstance(entry, dict):
            nodeid = entry.get("nodeid")
            if isinstance(nodeid, str) and nodeid:
                yield nodeid


def normalize_logs(value):
    """Return a group's ``log`` field as a list of log strings.

    ``log`` is a JSON array with one element per distinct observed error; an
    element may bind two fragments with ``|`` (both must appear in the same
    error message). A plain string is still tolerated for older configs and
    returned as a single-element list.
    """
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    return []


def collect_log_map(config_dir):
    """Return a {nodeid: [log, ...]} mapping from the blacklist files.

    Both file shapes are supported:

      * ``*_unsupported_testcases.json``: a top-level list of
        ``{log, nodeids: {...}}`` entries, where every nodeid key maps to its
        group's ``log`` list;
      * grouped ``*_grouped.json``: nodeids under ``groups[].nodes`` and/or
        ``groups[].canns[].nodes`` with a group-level ``log``.

    Empty logs yield an empty list so callers can skip them.
    """
    config_dir = Path(config_dir)
    if config_dir.is_dir():
        paths = sorted(config_dir.glob("**/*.json"))
    else:
        paths = []

    log_map = {}
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                log = normalize_logs(entry.get("log"))
                nodeids = entry.get("nodeids")
                if isinstance(nodeids, dict):
                    for nodeid in nodeids:
                        if isinstance(nodeid, str) and nodeid not in log_map:
                            log_map[nodeid] = log
        else:
            for group in data.get("groups", []) or []:
                if not isinstance(group, dict):
                    continue
                log = normalize_logs(group.get("log"))
                nodeid_lists = []
                nodes = group.get("nodes")
                if isinstance(nodes, list):
                    nodeid_lists.append(nodes)
                for cann in group.get("canns", []) or []:
                    if isinstance(cann, dict) and isinstance(cann.get("nodes"), list):
                        nodeid_lists.append(cann["nodes"])
                for entries in nodeid_lists:
                    for nodeid in _nodeid_strings(entries):
                        if nodeid not in log_map:
                            log_map[nodeid] = log
    return log_map


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        default=".github/config/blacklist_testccases",
        help="Directory containing the blacklist files (default: "
        ".github/config/blacklist_testccases)",
    )
    parser.add_argument(
        "--platform",
        choices=("a3", "a5"),
        help="Optional platform; remove nodeids recorded as passed for it",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the JSON nodeid array",
    )
    args = parser.parse_args(argv)

    nodeids, stats = collect_nodeids(args.config_dir)
    parse_errors = []
    for name, kind, count in stats:
        if kind == "nodes":
            print(f"  {name}: +{count} nodeids")
        else:
            print(f"  {name}: {kind}: {count}")
            parse_errors.append(f"{name}: {count}")
    if parse_errors:
        raise SystemExit("Blacklist JSON parse errors: " + "; ".join(parse_errors))
    if not nodeids:
        raise SystemExit(f"Flattened blacklist is empty: {args.config_dir}")

    if args.platform:
        diff_path = Path(args.config_dir) / "conf" / "platform_passed_diff.json"
        if diff_path.is_file():
            diff_data = json.loads(diff_path.read_text(encoding="utf-8"))
            passed = set(diff_data.get(f"passed_{args.platform}", []) or [])
            before = len(nodeids)
            nodeids = [nodeid for nodeid in nodeids if nodeid not in passed]
            print(f"  removed {before - len(nodeids)} passed nodeids for {args.platform}")

    print(f"total unique nodeids: {len(nodeids)}")
    if args.output:
        Path(args.output).write_text(
            json.dumps(nodeids, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"written: {args.output}")


if __name__ == "__main__":
    main()