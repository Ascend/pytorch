#!/usr/bin/env python3
"""
Test suite for discover_test_files.py

This test script covers:
    - Path normalization functions
    - YAML parsing functions
    - Test file discovery (Step 1)
    - Type filtering (Step 2)
    - Path rules filtering (Step 3)
    - Integration tests (all 3 steps)
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# Add script directory to path for import
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

# Import the module to test
import discover_test_files as dtf


# ==============================================================================
# Test Infrastructure
# ==============================================================================


class TestResult:
    """Collect test results."""
    passed: int = 0
    failed: int = 0
    errors: List[str] = []

    def add_pass(self, name: str):
        self.passed += 1
        print(f"  ✓ {name}")

    def add_fail(self, name: str, reason: str):
        self.failed += 1
        self.errors.append(f"{name}: {reason}")
        print(f"  ✗ {name}")
        print(f"    Reason: {reason}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"Test Results: {self.passed} passed, {self.failed} failed ({total} total)")
        print(f"{'=' * 60}")
        if self.errors:
            print("\nFailed tests:")
            for error in self.errors:
                print(f"  - {error}")
        return self.failed == 0


def assert_equal(actual, expected, name: str, result: TestResult):
    """Assert equality and record result."""
    if actual == expected:
        result.add_pass(name)
    else:
        result.add_fail(name, f"Expected: {expected}, Actual: {actual}")


def assert_true(condition: bool, name: str, result: TestResult, reason: str = ""):
    """Assert true and record result."""
    if condition:
        result.add_pass(name)
    else:
        result.add_fail(name, reason or "Condition was False")


def assert_in(item, container, name: str, result: TestResult):
    """Assert item in container."""
    if item in container:
        result.add_pass(name)
    else:
        result.add_fail(name, f"{item} not in {container}")


def assert_not_in(item, container, name: str, result: TestResult):
    """Assert item not in container."""
    if item not in container:
        result.add_pass(name)
    else:
        result.add_fail(name, f"{item} should not be in {container}")


def assert_list_equal(actual: List, expected: List, name: str, result: TestResult):
    """Assert lists are equal."""
    if sorted(actual) == sorted(expected):
        result.add_pass(name)
    else:
        result.add_fail(name, f"Expected sorted: {sorted(expected)}, Actual sorted: {sorted(actual)}")


# ==============================================================================
# Test Fixtures (Mock Test Directory)
# ==============================================================================


def create_mock_test_directory() -> Path:
    """Create a mock test directory structure for testing."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_discover_"))
    test_dir = temp_dir / "test"
    test_dir.mkdir()

    # Create test files in various directories
    files_to_create = [
        # Root level tests
        "test_autograd.py",
        "test_cuda.py",
        "test_distributed.py",  # This should NOT be in distributed/ subdir
        "test_nn.py",
        "test_ops.py",

        # Distributed tests
        "distributed/test_c10d.py",
        "distributed/test_ddp.py",
        "distributed/algorithms/test_algo.py",

        # NN tests
        "nn/test_convolution.py",
        "nn/test_linear.py",
        "nn/test_pooling.py",

        # Export tests
        "export/test_export.py",
        "export/test_export_legacy.py",

        # FX tests
        "fx/test_fx.py",
        "fx/test_shape_inference.py",

        # Quantization tests
        "quantization/test_quantize.py",

        # Functorch tests
        "functorch/test_functorch.py",

        # Mobile tests
        "mobile/test_mobile.py",

        # Dynamo tests
        "dynamo/test_dynamo.py",

        # Custom backend tests
        "custom_backend/test_backend.py",

        # Non-test files (should be ignored)
        "conftest.py",
        "common_utils.py",
        "distributed/__init__.py",
    ]

    for file_path in files_to_create:
        full_path = test_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f"# Mock test file: {file_path}\n", encoding="utf-8")

    return test_dir


def create_mock_case_paths_yml(test_dir: Path) -> Path:
    """Create a mock case_paths_ci.yml file."""
    yml_content = """
# Mock whitelist and blacklist configuration
whitelist:
  - test/test_autograd.py
  - test/test_nn.py
  - test/test_ops.py
  - test/nn
  - test/export
  - test/functorch
  - test/distributed

blacklist:
  - test/export/test_export_legacy.py
  - test/nn/test_convolution.py
  - test/test_cuda.py
"""
    yml_path = test_dir.parent / "case_paths_ci.yml"
    yml_path.write_text(yml_content, encoding="utf-8")
    return yml_path


def create_mock_simple_yml(test_dir: Path) -> Path:
    """Create a simple YAML file (for testing parse_simple_yaml_lists)."""
    yml_content = """
whitelist:
  - "test/test_autograd.py"
  - 'test/test_nn.py'
  - test/nn
  # This is a comment
blacklist:
  - test/test_cuda.py
"""
    yml_path = test_dir.parent / "simple_yml.yml"
    yml_path.write_text(yml_content, encoding="utf-8")
    return yml_path


def cleanup_mock_directory(test_dir: Path):
    """Clean up mock test directory."""
    import shutil
    try:
        shutil.rmtree(test_dir.parent)
    except Exception:
        pass


# ==============================================================================
# Unit Tests: Path Normalization
# ==============================================================================


def test_normalize_path(result: TestResult):
    """Test normalize_path function."""
    print("\n[Path Normalization Tests]")

    # Test 1: Backslash conversion
    assert_equal(
        dtf.normalize_path("test\\distributed\\test_c10d.py"),
        "test/distributed/test_c10d.py",
        "normalize_path: backslash conversion",
        result
    )

    # Test 2: Remove ./ prefix
    assert_equal(
        dtf.normalize_path("./test/test_autograd.py"),
        "test/test_autograd.py",
        "normalize_path: remove ./ prefix",
        result
    )

    # Test 3: Remove multiple ./ prefixes
    assert_equal(
        dtf.normalize_path("././test/test.py"),
        "test/test.py",
        "normalize_path: multiple ./ prefixes",
        result
    )

    # Test 4: Strip trailing slashes
    assert_equal(
        dtf.normalize_path("test/distributed/"),
        "test/distributed",
        "normalize_path: strip trailing slash",
        result
    )

    # Test 5: Strip leading/trailing whitespace
    assert_equal(
        dtf.normalize_path("  test/test.py  "),
        "test/test.py",
        "normalize_path: strip whitespace",
        result
    )


def test_normalize_rule_path(result: TestResult):
    """Test normalize_rule_path function."""
    print("\n[Rule Path Normalization Tests]")

    # Test 1: Add test/ prefix to simple path
    assert_equal(
        dtf.normalize_rule_path("test_autograd.py"),
        "test/test_autograd.py",
        "normalize_rule_path: add test/ prefix",
        result
    )

    # Test 2: Path already has test/ prefix
    assert_equal(
        dtf.normalize_rule_path("test/distributed"),
        "test/distributed",
        "normalize_rule_path: keep test/ prefix",
        result
    )

    # Test 3: Directory path (no .py suffix)
    assert_equal(
        dtf.normalize_rule_path("nn"),
        "test/nn",
        "normalize_rule_path: directory path",
        result
    )

    # Test 4: Empty string
    assert_equal(
        dtf.normalize_rule_path(""),
        "",
        "normalize_rule_path: empty string",
        result
    )

    # Test 5: Just 'test'
    assert_equal(
        dtf.normalize_rule_path("test"),
        "test",
        "normalize_rule_path: just 'test'",
        result
    )


# ==============================================================================
# Unit Tests: YAML Parsing
# ==============================================================================


def test_parse_simple_yaml_lists(result: TestResult):
    """Test parse_simple_yaml_lists function."""
    print("\n[YAML Parsing Tests]")

    # Test 1: Basic parsing
    yml_text = """
whitelist:
  - test/test_autograd.py
  - test/nn
blacklist:
  - test/test_cuda.py
"""
    parsed = dtf.parse_simple_yaml_lists(yml_text)
    assert_list_equal(
        parsed["whitelist"],
        ["test/test_autograd.py", "test/nn"],
        "parse_simple_yaml_lists: whitelist",
        result
    )
    assert_list_equal(
        parsed["blacklist"],
        ["test/test_cuda.py"],
        "parse_simple_yaml_lists: blacklist",
        result
    )

    # Test 2: With quotes and comments
    yml_text2 = """
whitelist:
  - "test/test_autograd.py"  # quoted value
  - 'test/test_nn.py'        # single quotes
  # This is a comment line
blacklist:
  - test/test_cuda.py
"""
    parsed2 = dtf.parse_simple_yaml_lists(yml_text2)
    assert_list_equal(
        parsed2["whitelist"],
        ["test/test_autograd.py", "test/test_nn.py"],
        "parse_simple_yaml_lists: quoted values",
        result
    )

    # Test 3: Empty file
    parsed3 = dtf.parse_simple_yaml_lists("")
    assert_list_equal(
        parsed3["whitelist"],
        [],
        "parse_simple_yaml_lists: empty whitelist",
        result
    )
    assert_list_equal(
        parsed3["blacklist"],
        [],
        "parse_simple_yaml_lists: empty blacklist",
        result
    )


def test_coerce_rule_list(result: TestResult):
    """Test coerce_rule_list function."""
    print("\n[Rule List Coercion Tests]")

    # Test 1: List of strings
    input_list = ["test_autograd.py", "nn", "test/distributed"]
    output = dtf.coerce_rule_list(input_list, "whitelist")
    assert_list_equal(
        output,
        ["test/test_autograd.py", "test/nn", "test/distributed"],
        "coerce_rule_list: string list",
        result
    )

    # Test 2: None input
    output2 = dtf.coerce_rule_list(None, "whitelist")
    assert_equal(output2, [], "coerce_rule_list: None input", result)

    # Test 3: Invalid type (should raise)
    try:
        dtf.coerce_rule_list("not_a_list", "whitelist")
        result.add_fail("coerce_rule_list: invalid type", "Should have raised ValueError")
    except ValueError:
        result.add_pass("coerce_rule_list: invalid type raises ValueError")

    # Test 4: Empty strings filtered
    input_list4 = ["test_autograd.py", "", "  ", "nn"]
    output4 = dtf.coerce_rule_list(input_list4, "whitelist")
    assert_list_equal(
        output4,
        ["test/test_autograd.py", "test/nn"],
        "coerce_rule_list: filter empty strings",
        result
    )


# ==============================================================================
# Unit Tests: Step 1 - Test File Discovery
# ==============================================================================


def test_discover_raw_test_files(result: TestResult):
    """Test discover_raw_test_files function."""
    print("\n[Step 1: Test File Discovery Tests]")

    test_dir = create_mock_test_directory()

    try:
        files = dtf.discover_raw_test_files(test_dir)

        # Test 1: All files have test/ prefix
        for f in files:
            assert_true(
                f.startswith("test/"),
                f"discover_raw_test_files: prefix check for {f}",
                result
            )

        # Test 2: Files are sorted
        assert_true(
            files == sorted(files),
            "discover_raw_test_files: files are sorted",
            result
        )

        # Test 3: Expected files found
        expected_files = [
            "test/test_autograd.py",
            "test/test_cuda.py",
            "test/test_nn.py",
            "test/test_ops.py",
            "test/distributed/test_c10d.py",
            "test/distributed/test_ddp.py",
            "test/nn/test_convolution.py",
            "test/nn/test_linear.py",
        ]
        for expected in expected_files:
            assert_in(expected, files, f"discover_raw_test_files: found {expected}", result)

        # Test 4: Non-test files NOT found
        assert_not_in("test/conftest.py", files, "discover_raw_test_files: skip conftest.py", result)
        assert_not_in("test/common_utils.py", files, "discover_raw_test_files: skip common_utils.py", result)
        assert_not_in("test/distributed/__init__.py", files, "discover_raw_test_files: skip __init__.py", result)

        # Test 5: Count check
        # We created 20 test_*.py files
        assert_true(
            len(files) >= 18,
            "discover_raw_test_files: count check",
            result,
            f"Expected at least 18 files, got {len(files)}"
        )

    finally:
        cleanup_mock_directory(test_dir)


# ==============================================================================
# Unit Tests: Step 2 - Type Filtering
# ==============================================================================


def test_filter_tests_by_type(result: TestResult):
    """Test filter_tests_by_type function."""
    print("\n[Step 2: Type Filtering Tests]")

    # Mock input files
    test_files = [
        "test/test_autograd.py",
        "test/test_cuda.py",
        "test/test_nn.py",
        "test/distributed/test_c10d.py",
        "test/distributed/test_ddp.py",
        "test/distributed/algorithms/test_algo.py",
        "test/nn/test_convolution.py",
    ]

    # Test 1: Distributed type
    selected, excluded = dtf.filter_tests_by_type(test_files, "distributed")
    assert_list_equal(
        selected,
        ["test/distributed/test_c10d.py", "test/distributed/test_ddp.py", "test/distributed/algorithms/test_algo.py"],
        "filter_tests_by_type: distributed selected",
        result
    )
    assert_list_equal(
        excluded,
        ["test/test_autograd.py", "test/test_cuda.py", "test/test_nn.py", "test/nn/test_convolution.py"],
        "filter_tests_by_type: distributed excluded",
        result
    )

    # Test 2: Regular type
    selected2, excluded2 = dtf.filter_tests_by_type(test_files, "regular")
    assert_list_equal(
        selected2,
        ["test/test_autograd.py", "test/test_cuda.py", "test/test_nn.py", "test/nn/test_convolution.py"],
        "filter_tests_by_type: regular selected",
        result
    )
    assert_list_equal(
        excluded2,
        ["test/distributed/test_c10d.py", "test/distributed/test_ddp.py", "test/distributed/algorithms/test_algo.py"],
        "filter_tests_by_type: regular excluded",
        result
    )

    # Test 3: Empty input
    selected3, excluded3 = dtf.filter_tests_by_type([], "distributed")
    assert_equal(selected3, [], "filter_tests_by_type: empty distributed", result)
    assert_equal(excluded3, [], "filter_tests_by_type: empty excluded", result)


# ==============================================================================
# Unit Tests: Step 3 - Path Rules Filtering
# ==============================================================================


def test_path_matches_rule(result: TestResult):
    """Test path_matches_rule function."""
    print("\n[Path Rule Matching Tests]")

    # Test 1: Exact match
    assert_true(
        dtf.path_matches_rule("test/test_autograd.py", "test/test_autograd.py"),
        "path_matches_rule: exact match",
        result
    )

    # Test 2: Directory prefix match
    assert_true(
        dtf.path_matches_rule("test/nn/test_convolution.py", "nn"),
        "path_matches_rule: directory prefix",
        result
    )

    # Test 3: Directory prefix match (with test/)
    assert_true(
        dtf.path_matches_rule("test/nn/test_linear.py", "test/nn"),
        "path_matches_rule: directory prefix with test/",
        result
    )

    # Test 4: No match
    assert_true(
        not dtf.path_matches_rule("test/test_autograd.py", "test/nn"),
        "path_matches_rule: no match",
        result
    )

    # Test 5: Glob pattern match
    assert_true(
        dtf.path_matches_rule("test/test_autograd.py", "test_*.py"),
        "path_matches_rule: glob pattern",
        result
    )

    # Test 6: Glob pattern no match
    assert_true(
        not dtf.path_matches_rule("test/nn/test_convolution.py", "test_*.py"),
        "path_matches_rule: glob pattern no match",
        result
    )


def test_apply_case_path_rules(result: TestResult):
    """Test apply_case_path_rules function."""
    print("\n[Step 3: Apply Path Rules Tests]")

    # Mock input files
    test_files = [
        "test/test_autograd.py",
        "test/test_cuda.py",
        "test/test_nn.py",
        "test/test_ops.py",
        "test/nn/test_convolution.py",
        "test/nn/test_linear.py",
        "test/export/test_export.py",
        "test/export/test_export_legacy.py",
    ]

    # Test 1: Whitelist only
    whitelist = ["test/test_autograd.py", "test/nn"]
    blacklist = []
    selected, excluded = dtf.apply_case_path_rules(test_files, whitelist, blacklist)
    assert_list_equal(
        selected,
        ["test/test_autograd.py", "test/nn/test_convolution.py", "test/nn/test_linear.py"],
        "apply_case_path_rules: whitelist only",
        result
    )

    # Test 2: Blacklist only (whitelist empty = select all)
    whitelist2 = []
    blacklist2 = ["test/test_cuda.py", "test/nn/test_convolution.py"]
    selected2, excluded2 = dtf.apply_case_path_rules(test_files, whitelist2, blacklist2)
    expected_selected2 = [f for f in test_files if f not in ["test/test_cuda.py", "test/nn/test_convolution.py"]]
    assert_list_equal(selected2, expected_selected2, "apply_case_path_rules: blacklist only", result)

    # Test 3: Both whitelist and blacklist
    whitelist3 = ["test/test_autograd.py", "test/nn", "test/export"]
    blacklist3 = ["test/nn/test_convolution.py", "test/export/test_export_legacy.py"]
    selected3, excluded3 = dtf.apply_case_path_rules(test_files, whitelist3, blacklist3)
    assert_list_equal(
        selected3,
        ["test/test_autograd.py", "test/nn/test_linear.py", "test/export/test_export.py"],
        "apply_case_path_rules: whitelist + blacklist",
        result
    )

    # Test 4: Empty whitelist and blacklist
    selected4, excluded4 = dtf.apply_case_path_rules(test_files, [], [])
    assert_equal(selected4, test_files, "apply_case_path_rules: empty rules", result)
    assert_equal(excluded4, [], "apply_case_path_rules: empty rules excluded", result)


# ==============================================================================
# Integration Tests: All 3 Steps
# ==============================================================================


def test_discover_test_files_integration(result: TestResult):
    """Test full discovery integration."""
    print("\n[Integration Tests: All 3 Steps]")

    test_dir = create_mock_test_directory()
    yml_path = create_mock_case_paths_yml(test_dir)

    try:
        # Test 1: Distributed tests with rules
        files_distributed, metadata_distributed = dtf.discover_test_files(
            test_dir=test_dir,
            test_type="distributed",
            case_paths_config=str(yml_path),
        )

        # All distributed files should be under test/distributed/
        for f in files_distributed:
            assert_true(
                f.startswith("test/distributed/"),
                f"integration: distributed prefix check for {f}",
                result
            )

        # Metadata should have correct counts
        assert_true(
            metadata_distributed["test_type"] == "distributed",
            "integration: distributed metadata type",
            result
        )

        # Test 2: Regular tests with rules
        files_regular, metadata_regular = dtf.discover_test_files(
            test_dir=test_dir,
            test_type="regular",
            case_paths_config=str(yml_path),
        )

        # No distributed files should be in regular
        for f in files_regular:
            assert_true(
                not f.startswith("test/distributed/"),
                f"integration: regular no distributed for {f}",
                result
            )

        # Files should be sorted
        assert_true(
            files_regular == sorted(files_regular),
            "integration: regular files sorted",
            result
        )

        # Test 3: Verify whitelist/blacklist effects
        # Based on yml: whitelist includes test/test_autograd.py, test/test_nn.py, test/nn, etc.
        # blacklist includes test/export/test_export_legacy.py, test/nn/test_convolution.py, test/test_cuda.py

        # test_cuda.py should NOT be in regular (blacklisted)
        assert_not_in("test/test_cuda.py", files_regular, "integration: blacklist excludes test_cuda.py", result)

        # test_nn/test_convolution.py should NOT be in regular (blacklisted)
        assert_not_in("test/nn/test_convolution.py", files_regular, "integration: blacklist excludes test_convolution.py", result)

        # Test 4: Without case_paths_config
        files_no_config, metadata_no_config = dtf.discover_test_files(
            test_dir=test_dir,
            test_type="regular",
            case_paths_config=None,
        )

        # Should include all regular tests (no whitelist/blacklist)
        assert_true(
            len(files_no_config) >= len(files_regular),
            "integration: no config has more files",
            result,
            f"Expected no_config >= with_config, got {len(files_no_config)} vs {len(files_regular)}"
        )

        # test_cuda.py should be present (no blacklist)
        assert_in("test/test_cuda.py", files_no_config, "integration: no blacklist includes test_cuda.py", result)

        # Test 5: Metadata structure
        assert_true(
            "total_files" in metadata_regular,
            "integration: metadata has total_files",
            result
        )
        assert_true(
            "whitelist_entries" in metadata_regular,
            "integration: metadata has whitelist_entries",
            result
        )
        assert_true(
            "blacklist_entries" in metadata_regular,
            "integration: metadata has blacklist_entries",
            result
        )

        print(f"\n  Distributed files count: {len(files_distributed)}")
        print(f"  Regular files count (with config): {len(files_regular)}")
        print(f"  Regular files count (no config): {len(files_no_config)}")
        print(f"  Metadata: {json.dumps(metadata_regular, indent=2)}")

    finally:
        cleanup_mock_directory(test_dir)


# ==============================================================================
# Edge Cases and Error Handling Tests
# ==============================================================================


def test_edge_cases(result: TestResult):
    """Test edge cases and error handling."""
    print("\n[Edge Cases Tests]")

    # Test 1: Empty test directory
    temp_dir = Path(tempfile.mkdtemp(prefix="test_empty_"))
    empty_test_dir = temp_dir / "test"
    empty_test_dir.mkdir()

    try:
        files = dtf.discover_raw_test_files(empty_test_dir)
        assert_equal(files, [], "edge case: empty test directory", result)
    finally:
        cleanup_mock_directory(empty_test_dir)

    # Test 2: Non-existent config file
    try:
        dtf.load_case_path_rules("/nonexistent/path.yml")
        result.add_fail("edge case: nonexistent config", "Should have raised FileNotFoundError")
    except FileNotFoundError:
        result.add_pass("edge case: nonexistent config raises FileNotFoundError")

    # Test 3: Malformed YAML (handled gracefully)
    temp_dir2 = Path(tempfile.mkdtemp(prefix="test_malformed_"))
    malformed_yml = temp_dir2 / "malformed.yml"
    malformed_yml.write_text("not a yaml at all!!!", encoding="utf-8")

    try:
        # parse_simple_yaml_lists should handle this gracefully
        parsed = dtf.parse_simple_yaml_lists("not a yaml at all!!!")
        # Should return empty lists
        assert_equal(parsed["whitelist"], [], "edge case: malformed yaml whitelist", result)
        assert_equal(parsed["blacklist"], [], "edge case: malformed yaml blacklist", result)
    finally:
        cleanup_mock_directory(temp_dir2)

    # Test 4: Glob pattern edge cases
    assert_true(
        dtf.path_matches_rule("test/test_a.py", "*"),
        "edge case: glob * matches everything",
        result
    )

    # test_?.py matches test_a.py, test_b.py, etc. (single char after test_)
    assert_true(
        dtf.path_matches_rule("test/test_a.py", "test_?.py"),
        "edge case: glob ? single char",
        result
    )


# ==============================================================================
# Main Test Runner
# ==============================================================================


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("discover_test_files.py Test Suite")
    print("=" * 60)

    result = TestResult()

    # Run all test functions
    test_normalize_path(result)
    test_normalize_rule_path(result)
    test_parse_simple_yaml_lists(result)
    test_coerce_rule_list(result)
    test_discover_raw_test_files(result)
    test_filter_tests_by_type(result)
    test_path_matches_rule(result)
    test_apply_case_path_rules(result)
    test_discover_test_files_integration(result)
    test_edge_cases(result)

    # Print final summary
    success = result.summary()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)