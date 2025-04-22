import os.path
import re
import shutil
import warnings


def _adapter(content):
    content = re.sub("import torch", "import torch\nimport torch_npu\nimport torch_npu.testing", content,
                     count=1)
    replace_map = {
        "\\.cuda": ".npu",
        "onlyCUDA": "onlyPRIVATEUSE1",
        "dtypesIfCUDA": "dtypesIfPRIVATEUSE1",
        "(?<!common_)cuda(?=[\": ',])": "npu",
        "(?<!_)CUDA(?=[\": '])": "NPU",
        "from torch.testing._internal.common_cuda import TEST_CUDA":
            "from torch.testing._internal.common_utils import TEST_PRIVATEUSE1",
        "(?<!import )TEST_CUDA": "TEST_PRIVATEUSE1"
    }

    for key, value in replace_map.items():
        content = re.sub(key, value, content)
    return content

def _write(content, file):
    with open(file, mode="w") as fw:
        fw.write(content)
    adapt_files.append(file)

def _read(file):
    with open(file, mode="r") as fr:
        content = fr.read()
    return content

def _directory(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    walk_path = os.path.join(dir_path, "test/temp_tests", "pytorch/test", line)
    if not os.path.exists(walk_path):
        warnings.warn(f"{walk_path} does not exist.")
        return
    for root, sub_path, files in os.walk(walk_path):
        for file in files:
            content = _read(os.path.join(root, file))
            content = _adapter(content)
            _write(content, os.path.join(dir_path, "test", line, file))


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(dir_path, "test/adaptive_tests.txt")
    adapt_files = []
    with open(path, mode="r") as fp:
        for line in fp.readlines():
            line = line.strip()
            if not line:
                continue
            if line.endswith(".py"):
                read_path = os.path.join(dir_path, "test/temp_tests", "pytorch/test", line)
                if not os.path.exists(read_path):
                    warnings.warn(f"{read_path} does not exist.")
                    continue
                text = _read(read_path)
                text = _adapter(text)
                _write(text, os.path.join(dir_path, "test", line))
            else:
                _directory(os.path.join(dir_path, "test", line))

    # print adapt_files
    print("adaptive files:")
    for adapt_file in adapt_files:
        print(f"{adapt_file}")

    # delete temp_tests
    shutil.rmtree(os.path.join(dir_path, "test/temp_tests"))
