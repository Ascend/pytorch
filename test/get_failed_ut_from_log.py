import argparse
import json
import re


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
    file1 = "unsupported_test_cases/.pytorch-disabled-tests.json"
    fr = open(file1)
    content = json.load(fr)
    if not ut_list:
        return
    for line in ut_list:
        content[line] = ["", [""]]
    with open("./pytorch-disabled-tests.json", mode="w") as fp:
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

