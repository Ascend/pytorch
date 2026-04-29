import argparse
import json
import os
import re


def _get_disabled_tests_file():
    try:
        from torch_npu._compat.version import CURRENT_VERSION
        ver = f"{CURRENT_VERSION[0]}.{CURRENT_VERSION[1]}"
        versioned = f"unsupported_test_cases/.pytorch-disabled-tests-{ver}.json"
        if os.path.exists(versioned):
            return versioned
    except Exception:
        pass
    return "unsupported_test_cases/.pytorch-disabled-tests.json"


def get_error_or_fail_ut(file):
    result = set()
    with open(file, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            if re.search(".*ERROR.*: test_.+\\)\n", line) \
                    or re.search(".*FAIL.*: test_.+\\)\n", line) \
                    or re.search(".*XPASS.*: test_.+\\)\n", line):
                if "(opset=" in line:
                    line = line.split("(opset")[0]
                substring = re.findall("test_.+\\)", line)[0].strip()
                result.add(substring)
    return result


def write_to_json(ut_list=None):
    file1 = _get_disabled_tests_file()
    fr = open(file1)
    content = json.load(fr)
    if not ut_list:
        return
    for line in ut_list:
        content[line] = ["", [""]]
    with open("./pytorch-disabled-tests.json", mode="w") as fp:
        # TODO: now only write main file, need rename to different version manually.
        fp.write("{\n")
        length = len(content.keys()) - 1
        for i, (key, (value1, value2)) in enumerate(content.items()):
            value2_str = "\"" + "\",\"".join(value2) + "\""
            if i < length:
                fp.write(f"  \"{key}\": [\"{value1}\", [{value2_str}]]" + ",\n")
            else:
                fp.write(f"  \"{key}\": [\"{value1}\", [{value2_str}]]" + "\n")
        fp.write("}\n")
    fr.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="out.log")
    args = parser.parse_args()
    failed_ut = get_error_or_fail_ut(args.file)
    write_to_json(ut_list=failed_ut)

