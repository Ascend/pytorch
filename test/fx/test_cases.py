"""
Test Cases for ShapeEnv APIs Compatibility on Ascend NPU

This module contains individual test cases for each ShapeEnv API,
designed for independent testing and validation following PyTorch test conventions.

Usage:
    python test_cases.py [test_name]
    
    Without arguments, runs all test cases.
    With test_name, runs only that specific test case.

Author: NPU Compatibility Testing
Date: 2026-05-19
"""

import torch
import torch_npu
import sympy
import sys
from torch.fx.experimental.symbolic_shapes import ShapeEnv, Source
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor


def test_produce_guards_expression():
    """Test produce_guards_expression API on NPU."""
    env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=env)
    fake_tensor = fake_mode.from_tensor(torch.randn(3, 4))
    placeholders = [fake_tensor]
    guards = env.produce_guards_expression(placeholders)
    assert isinstance(guards, str), f"Expected str, got {type(guards)}"


def test_produce_guards_verbose():
    """Test produce_guards_verbose API on NPU."""
    env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=env)
    fake_tensor = fake_mode.from_tensor(torch.randn(3, 4))
    placeholders = [fake_tensor]
    source = Source()
    sources = [source] * len(placeholders)
    guards = env.produce_guards_verbose(placeholders, sources)
    assert guards is not None


def test_replace():
    """Test replace API on NPU."""
    env = ShapeEnv()
    a, b = sympy.symbols('a b')
    original_expr = a + b
    new_expr = env.replace(original_expr)
    assert new_expr == original_expr


def test_add_backed_var_to_val():
    """Test add_backed_var_to_val API on NPU."""
    env = ShapeEnv()
    backed_sym = sympy.Symbol('test_backed_1')
    env.add_backed_var_to_val(backed_sym, 10)


def test_simplify():
    """Test simplify API on NPU."""
    env = ShapeEnv()
    a, b = sympy.symbols('a b')
    expr = (a + b) - b
    simplified = env.simplify(expr)
    assert simplified == a, f"Expected {a}, got {simplified}"


def run_all_tests():
    """Run all test cases and generate summary report."""
    print("=" * 70)
    print("RUNNING ALL SHAPEENV API TESTS")
    print("=" * 70)
    print()
    
    if not torch.npu.is_available():
        print("ERROR: NPU is not available!")
        print("Please ensure CANN and torch-npu are properly installed.")
        sys.exit(1)
    
    print(f"Environment:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  torch-npu: {torch_npu.__version__}")
    print(f"  NPU Device: {torch.npu.get_device_name(0)}")
    print()
    
    test_funcs = [
        ("produce_guards_expression", test_produce_guards_expression),
        ("produce_guards_verbose", test_produce_guards_verbose),
        ("replace", test_replace),
        ("add_backed_var_to_val", test_add_backed_var_to_val),
        ("simplify", test_simplify),
    ]
    
    passed = 0
    failed = 0
    results = []
    
    for name, func in test_funcs:
        print(f"Testing: {name}")
        try:
            func()
            print(f"  [PASSED]")
            passed += 1
            results.append((name, True, None))
        except Exception as e:
            print(f"  [FAILED]: {type(e).__name__}: {str(e)[:80]}")
            failed += 1
            results.append((name, False, str(e)))
        print()
    
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total: {len(test_funcs)} | Passed: {passed} | Failed: {failed}")
    print(f"Success Rate: {passed/len(test_funcs)*100:.1f}%")
    print()
    
    for name, success, error in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    print("=" * 70)
    
    return passed == len(test_funcs)


def run_specific_test(test_name):
    """Run a specific test case by name."""
    test_map = {
        "produce_guards_expression": test_produce_guards_expression,
        "produce_guards_verbose": test_produce_guards_verbose,
        "replace": test_replace,
        "add_backed_var_to_val": test_add_backed_var_to_val,
        "simplify": test_simplify,
    }
    
    if test_name not in test_map:
        print(f"ERROR: Unknown test '{test_name}'")
        print(f"Available tests: {', '.join(test_map.keys())}")
        return False
    
    if not torch.npu.is_available():
        print("ERROR: NPU is not available!")
        return False
    
    print(f"Running test: {test_name}")
    test_map[test_name]()
    print("[PASSED]")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
        sys.exit(0 if success else 1)
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)