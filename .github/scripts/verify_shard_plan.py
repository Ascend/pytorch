#!/usr/bin/env python3
"""
Test script for the 4-step test planning flow.

Usage:
    python test_4step_flow.py --test-dir /path/to/pytorch/test --shard 1 --num-shards 10 --test-type regular

This script tests:
    Step 1: Test file discovery
    Step 2: Shard type filtering
    Step 3: Whitelist/blacklist filtering
    Step 4: Shard assignment
"""

import sys
from pathlib import Path

# Add scripts directory to path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from run_npu_test_shard import (
    step1_discover_test_files,
    step2_filter_by_type,
    step3_apply_path_rules,
    step4_assign_shard,
    plan_shard_tests,
    create_test_plan_summary,
    load_case_path_rules,
    DiscoveryResult,
    TypeFilterResult,
    PathRulesFilterResult,
    ShardAssignmentResult,
    ShardPlanResult,
)


def test_step1_discovery(test_dir: Path) -> DiscoveryResult:
    """Test Step 1: Test file discovery"""
    print("\n" + "=" * 80)
    print("Step 1: Test File Discovery")
    print("=" * 80)

    result = step1_discover_test_files(test_dir)

    print(f"  Test directory: {result.test_dir}")
    print(f"  Total test files discovered: {result.total_count}")
    print(f"  Sample files (first 5):")
    for f in result.all_test_files[:5]:
        print(f"    - {f}")
    if result.total_count > 5:
        print(f"    ... and {result.total_count - 5} more files")

    return result


def test_step2_type_filter(discovery_result: DiscoveryResult, shard_type: str) -> TypeFilterResult:
    """Test Step 2: Shard type filtering"""
    print("\n" + "=" * 80)
    print(f"Step 2: Shard Type Filtering (type={shard_type})")
    print("=" * 80)

    result = step2_filter_by_type(discovery_result, shard_type)

    print(f"  Shard type: {result.shard_type}")
    print(f"  Selected files: {result.selected_count}")
    print(f"  Excluded files: {result.excluded_count}")
    print(f"  Selected sample (first 5):")
    for f in result.selected_files[:5]:
        print(f"    - {f}")
    if result.selected_count > 5:
        print(f"    ... and {result.selected_count - 5} more files")

    return result


def test_step3_path_rules(
    type_filter_result: TypeFilterResult,
    whitelist: list,
    blacklist: list
) -> PathRulesFilterResult:
    """Test Step 3: Whitelist/blacklist filtering"""
    print("\n" + "=" * 80)
    print("Step 3: Whitelist/Blacklist Filtering")
    print("=" * 80)

    result = step3_apply_path_rules(type_filter_result, whitelist, blacklist)

    print(f"  Whitelist entries: {len(result.whitelist)}")
    print(f"  Blacklist entries: {len(result.blacklist)}")
    print(f"  Selected files: {result.selected_count}")
    print(f"  Excluded files: {result.excluded_count}")
    print(f"  Whitelist sample (first 3):")
    for r in result.whitelist[:3]:
        print(f"    - {r}")
    print(f"  Blacklist sample (first 3):")
    for r in result.blacklist[:3]:
        print(f"    - {r}")
    print(f"  Selected sample (first 5):")
    for f in result.selected_files[:5]:
        print(f"    - {f}")
    if result.selected_count > 5:
        print(f"    ... and {result.selected_count - 5} more files")

    return result


def test_step4_shard_assignment(
    path_rules_result: PathRulesFilterResult,
    shard: int,
    num_shards: int
) -> ShardAssignmentResult:
    """Test Step 4: Shard assignment"""
    print("\n" + "=" * 80)
    print(f"Step 4: Shard Assignment (shard={shard}/{num_shards})")
    print("=" * 80)

    result = step4_assign_shard(path_rules_result, shard, num_shards)

    print(f"  Shard: {result.shard}/{result.num_shards}")
    print(f"  Planned test files: {result.planned_count}")
    print(f"  Planned tests:")
    for f in result.planned_tests:
        print(f"    - {f}")

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test 4-step test planning flow")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to PyTorch test directory")
    parser.add_argument("--shard", type=int, default=1, help="Shard number (1-indexed)")
    parser.add_argument("--num-shards", type=int, default=10, help="Total number of shards")
    parser.add_argument("--test-type", type=str, choices=["distributed", "regular"], default="regular", help="Test type")
    parser.add_argument("--case-paths-config", type=str, help="Path to case_paths_ci.yml")
    parser.add_argument("--output-plan", type=str, help="Output planned tests to file")
    args = parser.parse_args()

    test_dir = Path(args.test_dir).resolve()
    if not test_dir.is_dir():
        print(f"ERROR: Test directory not found: {test_dir}")
        sys.exit(1)

    # Load whitelist/blacklist
    whitelist = []
    blacklist = []
    if args.case_paths_config:
        config_path = Path(args.case_paths_config).resolve()
        if config_path.exists():
            _, whitelist, blacklist = load_case_path_rules(str(config_path))
            print(f"\nLoaded case_paths_ci.yml from: {config_path}")
            print(f"  Whitelist entries: {len(whitelist)}")
            print(f"  Blacklist entries: {len(blacklist)}")

    # Execute 4-step flow
    print("\n" + "=" * 80)
    print("Executing 4-Step Test Planning Flow")
    print("=" * 80)

    # Step 1
    discovery_result = test_step1_discovery(test_dir)

    # Step 2
    type_filter_result = test_step2_type_filter(discovery_result, args.test_type)

    # Step 3
    path_rules_result = test_step3_path_rules(type_filter_result, whitelist, blacklist)

    # Step 4
    shard_assignment_result = test_step4_shard_assignment(
        path_rules_result,
        args.shard,
        args.num_shards
    )

    # Create complete result
    plan_result = ShardPlanResult(
        discovery=discovery_result,
        type_filter=type_filter_result,
        path_rules_filter=path_rules_result,
        shard_assignment=shard_assignment_result,
    )

    # Print final summary
    print("\n" + "=" * 80)
    print("Final Test Planning Summary")
    print("=" * 80)
    print(create_test_plan_summary(plan_result))

    # Output planned tests to file
    if args.output_plan:
        output_path = Path(args.output_plan)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            for test in plan_result.get_planned_tests():
                f.write(f"{test}\n")
        print(f"\nPlanned tests saved to: {output_path}")

    # Print all planned tests
    print("\n" + "=" * 80)
    print(f"Final Planned Tests List ({len(plan_result.get_planned_tests())} files)")
    print("=" * 80)
    for i, test in enumerate(plan_result.get_planned_tests(), 1):
        print(f"  [{i:03d}] {test}")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()