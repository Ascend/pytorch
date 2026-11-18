#!/usr/bin/env python3
"""Flatten all blacklist files under ``blacklist_testccases/`` into pure nodeids.

The files in that directory use several different shapes:

  * grouped files with top-level ``groups``, where each group carries
    ``nodes`` as a list of plain nodeid strings
    (cann/device/temporary ``*_grouped.json``);
  * the dtype file, whose nodeids live one level deeper under
    ``groups[].canns[].nodes``;
  * ``blacklist_logs.json``, a ``log -> count`` map that contains no nodeids
    and is therefore skipped.

This script walks every ``*.json`` file, collects every string found under a
``nodes`` key (accepting either a bare string or a ``{"nodeid": ...}`` object),
de-duplicates and sorts the result into a single pure nodeid list.

Usage (in the shard workflow it is imported, not run as a CLI):

    python3 flatten_blacklist_testccases.py \
        --config-dir .github/config/blacklist_testccases \
        --output grouped-nodeids.txt
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
    """Recursively collect nodeids from every ``nodes`` key."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "nodes" and isinstance(value, list):
                _add(value, nodeids)
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
        paths = sorted(config_dir.glob("*.json"))
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


def collect_log_map(config_dir):
    """Return a {nodeid: log} mapping from the grouped files.

    Each group carries a ``log`` (raw error message, possibly ';'-joined) and
    its nodeids live under ``nodes`` and/or ``canns[].nodes``. Empty logs are
    kept as-is so callers can skip them.
    """
    config_dir = Path(config_dir)
    if config_dir.is_dir():
        paths = sorted(config_dir.glob("*.json"))
    else:
        paths = []

    log_map = {}
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for group in data.get("groups", []) or []:
            if not isinstance(group, dict):
                continue
            log = group.get("log") or ""
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
        "--output",
        help="Optional path to write the pure nodeid list to (one per line)",
    )
    args = parser.parse_args(argv)

    nodeids, stats = collect_nodeids(args.config_dir)
    for name, kind, count in stats:
        if kind == "nodes":
            print(f"  {name}: +{count} nodeids")
        else:
            print(f"  {name}: {kind}: {count}")
    print(f"total unique nodeids: {len(nodeids)}")

    if args.output:
        Path(args.output).write_text(
            "\n".join(nodeids) + ("\n" if nodeids else ""), encoding="utf-8"
        )
        print(f"written: {args.output}")


if __name__ == "__main__":
    main()