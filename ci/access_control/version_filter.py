"""AST-based version filter for test files.

Scans test files for @runIfVersion decorators and filters files based on
whether --between_version is specified.

When --between_version IS specified (versioned job):
- No @runIfVersion decorators found  -> drop (only version-decorated cases run)
- Any @runIfVersion intersects with CI range -> keep
- All @runIfVersion decorators outside CI range -> drop

When --between_version is NOT specified (non-versioned job):
- No @runIfVersion decorators found  -> keep (universal cases run)
- Any @runIfVersion decorators found -> drop (versioned cases skip)
"""
import ast
from pathlib import Path
from typing import Optional, Tuple

VersionTuple = Tuple[int, ...]


def _parse_version(s: str) -> VersionTuple:
    s = s.split("+")[0]
    return tuple(int(x) for x in s.split("."))


def _extract_args(call_node: ast.Call) -> dict:
    """Extract min/max from a Call node (positional + keyword).

    Decorator signature: runIfVersion(min=None, max=None), so positional
    order is (min, max). Keyword args override positional.
    Only string values are kept; None and non-string constants are ignored
    (treated as "not specified").
    """
    result = {}
    # Keyword args first.
    for kw in call_node.keywords:
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            result[kw.arg] = kw.value.value
    # Fill positional args for params not already set by kwargs.
    pos_params = ["min", "max"]
    for i, a in enumerate(call_node.args):
        if i < len(pos_params) and pos_params[i] not in result:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                result[pos_params[i]] = a.value
    return result


def _decorator_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _ranges_intersect(
    a_min: Optional[VersionTuple],
    a_max: Optional[VersionTuple],
    b_min: Optional[VersionTuple],
    b_max: Optional[VersionTuple],
) -> bool:
    """Check if range A intersects range B (both inclusive).

    None means unbounded on that side.
    """
    if (a_min is None and a_max is None) or (b_min is None and b_max is None):
        return True
    if a_max is not None and b_min is not None and a_max < b_min:
        return False
    if a_min is not None and b_max is not None and a_min > b_max:
        return False
    return True


def _file_has_version_decorators(file_path: str) -> bool:
    """Return True if the file has any @runIfVersion decorators."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=file_path)
    except (SyntaxError, OSError, UnicodeDecodeError):
        return False

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for dec in node.decorator_list:
            call = dec if isinstance(dec, ast.Call) else None
            if call is None and isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Call):
                call = dec.value
            if call is not None and _decorator_name(call) == "runIfVersion":
                return True
    return False


def _file_should_run(
    file_path: str,
    ci_min: Optional[VersionTuple],
    ci_max: Optional[VersionTuple],
) -> bool:
    """Return True if the file should be kept for the CI version range."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=file_path)
    except (SyntaxError, OSError, UnicodeDecodeError):
        # Cannot parse -> keep the file, let the runner decide
        return True

    found_any = False
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for dec in node.decorator_list:
            call = dec if isinstance(dec, ast.Call) else None
            if call is None and isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Call):
                call = dec.value
            if call is None or _decorator_name(call) != "runIfVersion":
                continue

            found_any = True
            args = _extract_args(call)
            try:
                min_ver = _parse_version(args["min"]) if "min" in args else None
                max_ver = _parse_version(args["max"]) if "max" in args else None
            except ValueError:
                # Invalid version string in decorator -> treat as no bound -> keep
                return True

            if _ranges_intersect(min_ver, max_ver, ci_min, ci_max):
                return True

    # No decorators -> drop; decorators but none intersected -> drop
    return False


def filter_test_files(
    test_files: dict,
    ci_min: Optional[VersionTuple],
    ci_max: Optional[VersionTuple],
) -> dict:
    """Filter a {ut_type: [file_paths]} dict by the CI version range.

    When ci_min/ci_max are None (no --between_version):
      - Keep files WITHOUT @runIfVersion decorators
      - Drop files WITH @runIfVersion decorators

    When ci_min/ci_max are set (--between_version specified):
      - Keep files with @runIfVersion that intersects the CI range
      - Drop files without @runIfVersion or with non-intersecting ranges
    """
    if ci_min is None and ci_max is None:
        # No version filter: keep only files without version decorators
        filtered = {}
        for ut_type, files in test_files.items():
            kept = [f for f in files if not _file_has_version_decorators(str(Path(f)))]
            filtered[ut_type] = kept
        return filtered

    filtered = {}
    for ut_type, files in test_files.items():
        kept = [f for f in files if _file_should_run(str(Path(f)), ci_min, ci_max)]
        filtered[ut_type] = kept
    return filtered
