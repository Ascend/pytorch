#!/usr/bin/env python3
"""
Discover test files for PyTorch NPU testing.

This script integrates 3 steps:
    Step 1: Test file discovery (scan all test_*.py)
    Step 2: Shard type filtering (distributed/regular)
    Step 3: Whitelist/blacklist filtering (case_paths_ci.yml)

Output: Sorted list of test file paths (with 'test/' prefix)

Usage:
    python discover_test_files.py \
        --test-dir /path/to/pytorch/test \
        --test-type distributed \
        --case-paths-config /path/to/case_paths_ci.yml \
        --output /path/to/output_file.txt

    # Or output to stdout:
    python discover_test_files.py \
        --test-dir /path/to/pytorch/test \
        --test-type regular \
        --case-paths-config /path/to/case_paths_ci.yml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore


# ==============================================================================
# Path Normalization Functions
# ==============================================================================


def normalize_path(value: str) -> str:
    """Normalize path: convert backslashes, remove ./ prefix."""
    normalized = value.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def normalize_rule_path(rule: str) -> str:
    """Normalize rule path: ensure it has 'test/' prefix."""
    normalized = normalize_path(rule)
    if not normalized:
        return ""
    if normalized == "test" or normalized.startswith("test/"):
        return normalized.rstrip("/")
    return f"test/{normalized}".rstrip("/")


# ==============================================================================
# YAML Parsing Functions
# ==============================================================================


def parse_simple_yaml_lists(raw_text: str) -> Dict[str, List[str]]:
    """Parse YAML file for whitelist/blacklist without yaml library."""
    parsed = {"whitelist": [], "blacklist": []}
    current_key = None

    for raw_line in raw_text.splitlines():
        without_comment = raw_line.split("#", 1)[0].rstrip()
        if not without_comment.strip():
            continue

        stripped = without_comment.lstrip()
        if not raw_line.startswith((" ", "\t")) and stripped.endswith(":"):
            key = stripped[:-1].strip()
            current_key = key if key in parsed else None
            continue

        if current_key and stripped.startswith("- "):
            value = stripped[2:].strip().strip("\"'")
            if value:
                parsed[current_key].append(value)

    return parsed


def coerce_rule_list(value, key: str) -> List[str]:
    """Validate and normalize rule list."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Expected '{key}' to be a list, got {type(value).__name__}")

    normalized_values = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Expected every '{key}' entry to be a string, got {type(item).__name__}")
        normalized = normalize_rule_path(item)
        if normalized:
            normalized_values.append(normalized)
    return normalized_values


def load_case_path_rules(config_file: Optional[str]) -> Tuple[str, List[str], List[str]]:
    """Load whitelist/blacklist rules from case_paths_ci.yml."""
    if not config_file:
        return "", [], []

    config_path = Path(config_file).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"case_paths_ci config not found: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8")

    if yaml is not None:
        payload = yaml.safe_load(raw_text) or {}
    else:
        payload = parse_simple_yaml_lists(raw_text)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML object in {config_path}, got {type(payload).__name__}")

    whitelist = coerce_rule_list(payload.get("whitelist"), "whitelist")
    blacklist = coerce_rule_list(payload.get("blacklist"), "blacklist")
    return str(config_path), whitelist, blacklist


# ==============================================================================
# Test File Discovery (Step 1)
# ==============================================================================


def discover_raw_test_files(test_dir: Path) -> List[str]:
    """Scan all test_*.py files in test directory."""
    files = []
    for test_file in test_dir.rglob("test_*.py"):
        rel_path = test_file.relative_to(test_dir).as_posix()
        files.append(f"test/{rel_path}")
    return sorted(files)


# ==============================================================================
# Type Filtering (Step 2)
# ==============================================================================


def filter_tests_by_type(test_files: List[str], test_type: str) -> Tuple[List[str], List[str]]:
    """Filter test files by test type (distributed/regular)."""
    if test_type == "distributed":
        selected = [f for f in test_files if f.startswith("test/distributed/")]
        excluded = [f for f in test_files if not f.startswith("test/distributed/")]
    else:
        selected = [f for f in test_files if not f.startswith("test/distributed/")]
        excluded = [f for f in test_files if f.startswith("test/distributed/")]
    return selected, excluded


# ==============================================================================
# Path Rules Filtering (Step 3)
# ==============================================================================


def path_matches_rule(test_path: str, rule: str) -> bool:
    """Check if test path matches a rule (supports glob patterns)."""
    import fnmatch

    normalized_path = normalize_path(test_path)
    normalized_rule = normalize_rule_path(rule)
    if not normalized_rule:
        return False

    if any(char in normalized_rule for char in "*?[]"):
        return fnmatch.fnmatch(normalized_path, normalized_rule)

    return normalized_path == normalized_rule or normalized_path.startswith(f"{normalized_rule}/")


def apply_case_path_rules(
    test_files: List[str], whitelist: List[str], blacklist: List[str]
) -> Tuple[List[str], List[str]]:
    """Apply whitelist and blacklist rules to filter test files."""
    # Apply whitelist (if empty, select all)
    if whitelist:
        selected = [path for path in test_files if any(path_matches_rule(path, rule) for rule in whitelist)]
    else:
        selected = list(test_files)

    # Apply blacklist
    if blacklist:
        selected = [path for path in selected if not any(path_matches_rule(path, rule) for rule in blacklist)]

    selected_set = set(selected)
    excluded = [path for path in test_files if path not in selected_set]
    return selected, excluded


# ==============================================================================
# Main Discovery Function
# ==============================================================================


def discover_test_files(
    test_dir: Path,
    test_type: str,
    case_paths_config: Optional[str],
) -> Tuple[List[str], Dict]:
    """
    Execute all 3 steps to discover test files.

    Returns:
        Tuple of (selected_files, metadata_dict)
    """
    # Step 1: Discover all test files
    all_test_files = discover_raw_test_files(test_dir)
    total_count = len(all_test_files)

    # Step 2: Filter by test type
    type_selected, type_excluded = filter_tests_by_type(all_test_files, test_type)

    # Step 3: Apply whitelist/blacklist rules
    config_path, whitelist, blacklist = load_case_path_rules(case_paths_config)
    rules_selected, rules_excluded = apply_case_path_rules(type_selected, whitelist, blacklist)

    # Metadata for reporting
    metadata = {
        "test_dir": str(test_dir),
        "test_type": test_type,
        "total_files": total_count,
        "type_selected": len(type_selected),
        "type_excluded": len(type_excluded),
        "whitelist_entries": len(whitelist),
        "blacklist_entries": len(blacklist),
        "rules_selected": len(rules_selected),
        "rules_excluded": len(rules_excluded),
        "case_paths_config": config_path,
    }

    return rules_selected, metadata


# ==============================================================================
# CLI Interface
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Discover test files for PyTorch NPU testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Path to the PyTorch test directory",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["distributed", "regular"],
        default="regular",
        help="Test type: 'distributed' for distributed tests, 'regular' for other tests",
    )
    parser.add_argument(
        "--case-paths-config",
        type=str,
        help="Path to case_paths_ci.yml for file-level whitelist/blacklist control",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for test file list (default: stdout)",
    )
    parser.add_argument(
        "--metadata-output",
        type=str,
        help="Output file path for metadata JSON (optional)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output including metadata",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    test_dir = Path(args.test_dir).resolve()
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Execute discovery
    selected_files, metadata = discover_test_files(
        test_dir=test_dir,
        test_type=args.test_type,
        case_paths_config=args.case_paths_config,
    )

    # Output test file list
    output_content = "\n".join(selected_files) + ("\n" if selected_files else "")

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_content, encoding="utf-8")
        if args.verbose:
            print(f"Written {len(selected_files)} test files to: {output_path}")
    else:
        sys.stdout.write(output_content)

    # Output metadata
    if args.metadata_output:
        metadata_path = Path(args.metadata_output).resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if args.verbose:
            print(f"Written metadata to: {metadata_path}")

    # Verbose summary
    if args.verbose:
        print(f"\nDiscovery Summary:")
        print(f"  Test directory: {test_dir}")
        print(f"  Test type: {args.test_type}")
        print(f"  Total files scanned: {metadata['total_files']}")
        print(f"  After type filter: {metadata['type_selected']} selected, {metadata['type_excluded']} excluded")
        if args.case_paths_config:
            print(f"  Whitelist entries: {metadata['whitelist_entries']}")
            print(f"  Blacklist entries: {metadata['blacklist_entries']}")
            print(f"  After rules filter: {metadata['rules_selected']} selected, {metadata['rules_excluded']} excluded")
        print(f"  Final selected files: {len(selected_files)}")


if __name__ == "__main__":
    main()