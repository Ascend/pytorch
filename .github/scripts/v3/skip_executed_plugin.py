"""
pytest plugin: skip tests that were already executed in a previous attempt.

Used by run_npu_test_file.py to continue execution after NPU poisoning.
Instead of re-running ALL tests from the previous attempt (including failed
ones), this plugin skips only the tests that already ran, allowing the
remaining tests (after the poisoned case) to execute.

The skip list is a text file with one entry per line:
    classname\tname

where classname and name match the JUnit XML's <testcase> attributes.

The file path is passed via the SKIP_EXECUTED_FILE environment variable.
"""

import os

_skip_pairs = set()


def pytest_configure(config):
    skip_file = os.environ.get("SKIP_EXECUTED_FILE")
    if skip_file and os.path.exists(skip_file):
        with open(skip_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "\t" in line:
                    classname, name = line.split("\t", 1)
                    _skip_pairs.add((classname, name))


def pytest_collection_modifyitems(config, items):
    if not _skip_pairs:
        return
    selected = []
    deselected = []
    for item in items:
        nodeid = item.nodeid
        parts = nodeid.split("::")
        file_part = parts[0]
        if file_part.endswith(".py"):
            module = file_part[:-3].replace("/", ".")
        else:
            module = file_part.replace("/", ".")
        if len(parts) >= 3:
            classname = f"{module}.{parts[1]}"
        else:
            classname = module
        name = item.name
        if (classname, name) in _skip_pairs:
            deselected.append(item)
        else:
            selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
