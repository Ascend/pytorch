#!/usr/bin/env python3
"""
Pytest plugin that deselects tests listed in disabled_testcases.json.

The disabled list format follows the existing convention used by Ascend
tooling, for example:
    test_example (__main__.TestClass)
"""

import json
import os
from typing import Iterable, Optional, Set


DISABLED_TESTCASES_ENV = "NPU_DISABLED_TESTCASES_JSON"


def _load_disabled_testcases() -> Set[str]:
    json_file = os.environ.get(DISABLED_TESTCASES_ENV)
    if not json_file:
        return set()

    try:
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[npu-disabled] Disabled testcases file not found: {json_file}")
        return set()
    except Exception as exc:
        print(f"[npu-disabled] Failed to load disabled testcases from {json_file}: {exc}")
        return set()

    if not isinstance(data, dict):
        print(f"[npu-disabled] Expected a JSON object in {json_file}, got {type(data).__name__}")
        return set()

    return set(data.keys())


def _iter_test_names(item) -> Iterable[str]:
    raw_name = getattr(item, "name", "")
    if raw_name:
        yield raw_name

    original_name = getattr(item, "originalname", None)
    if original_name and original_name != raw_name:
        yield original_name

    if raw_name and "[" in raw_name:
        yield raw_name.split("[", 1)[0]


def _get_class_name(item) -> Optional[str]:
    cls = getattr(item, "cls", None)
    if cls is not None:
        return cls.__name__

    parent = getattr(item, "parent", None)
    while parent is not None:
        name = getattr(parent, "name", None)
        if name and name.startswith("Test"):
            return name
        parent = getattr(parent, "parent", None)

    return None


def _build_match_keys(item) -> Set[str]:
    keys = set()

    nodeid = getattr(item, "nodeid", "")
    if nodeid:
        keys.add(nodeid)

    module = getattr(item, "module", None)
    module_name = getattr(module, "__name__", None)
    class_name = _get_class_name(item)

    for test_name in _iter_test_names(item):
        keys.add(test_name)
        if class_name:
            keys.add(f"{test_name} (__main__.{class_name})")
            if module_name:
                keys.add(f"{test_name} ({module_name}.{class_name})")

    return keys


_DISABLED_TESTCASES = _load_disabled_testcases()


def pytest_collection_modifyitems(config, items):
    if not _DISABLED_TESTCASES:
        return

    deselected = []
    kept = []
    matched = set()

    for item in items:
        match_keys = _build_match_keys(item)
        if _DISABLED_TESTCASES.intersection(match_keys):
            deselected.append(item)
            matched.update(_DISABLED_TESTCASES.intersection(match_keys))
        else:
            kept.append(item)

    if deselected:
        items[:] = kept
        config.hook.pytest_deselected(items=deselected)
        print(
            "[npu-disabled] Deselected "
            f"{len(deselected)} collected tests from disabled_testcases.json "
            f"({len(matched)} unique entries matched in this shard)"
        )
