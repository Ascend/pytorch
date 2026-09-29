#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


PUBLIC_ENV_PREFIX = "torch_npu_public_env: "


def normalize_whitespace(value):
    return re.sub(r"\s+", " ", value).strip()


def check_public_environments(schema_path, source_path):
    with schema_path.open(encoding="utf-8") as schema_file:
        schema = json.load(schema_file)

    source = normalize_whitespace(source_path.read_text(encoding="utf-8"))
    failures = []
    for key, value in schema.items():
        if not key.startswith(PUBLIC_ENV_PREFIX):
            continue
        if normalize_whitespace(value["mode"]) not in source:
            env_name = key.removeprefix(PUBLIC_ENV_PREFIX)
            failures.append(f"the mode of the environment variable {env_name} has been changed")

    return failures


def main():
    parser = argparse.ArgumentParser(description="Check public environment variable modes.")
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()

    failures = check_public_environments(args.schema, args.source)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
