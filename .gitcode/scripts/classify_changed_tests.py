#!/usr/bin/env python3
"""
复刻旧线 access_control 的 4 个策略，输入变更文件列表，输出应跑的测试文件。

策略:
  1. TestFileStrategy:   改了 test/**/test_*.py → 返回该文件
  2. CoreTestStrategy:   改了 torch_npu/ 下非 test/非 docs 文件 → 跑 test/npu/ 全量
  3. DirectoryMapping:   改了 torch_npu/<module>/ → 映射到对应 test/<module>/
  4. OpStrategy:         改了 *KernelNpu*.cpp → 从文件名解析算子名查找匹配的测试文件

输出格式:
  所有路径为相对路径，去掉 test/ 前缀和 .py 后缀:
  test/npu/test_add.py → npu/test_add
  test/distributed/test_ddp.py → distributed/test_ddp

  与 detect_changed_tests.sh 的直接检测格式一致，
  可直接传给 run_npu_test_shard.py --test-files。

用法:
  echo "torch_npu/nn/functional.py" | python classify_changed_tests.py
"""

import fnmatch
import os
import re
import sys
from pathlib import Path
from typing import List


# ==============================================================================
# 路径标准化
# ==============================================================================

def normalize_test_path(path: str) -> str:
    """test/npu/test_add.py → npu/test_add"""
    p = path
    if p.startswith("test/"):
        p = p[5:]
    if p.endswith(".py"):
        p = p[:-3]
    return p


# ==============================================================================
# 策略 1: TestFileStrategy — 直接改了测试文件
# ==============================================================================

def strategy_test_file(modify_file: str, repo_root: str) -> List[str]:
    """
    如果变更路径在 test/ 下且文件名匹配 test_*.py，直接返回该文件 (去掉前缀后缀)。
    """
    f = Path(modify_file)
    if f.parts and f.parts[0] == "test" and re.match(r"test_(.+)\.py", f.name):
        full = Path(repo_root) / modify_file
        if full.exists():
            return [normalize_test_path(modify_file)]
    return []


# ==============================================================================
# 策略 2: CoreTestStrategy — 核心代码变更跑全量核心
# ==============================================================================

def strategy_core(modify_file: str, repo_root: str) -> List[str]:
    """
    变更的第一级目录不是 test 也不是 docs → 跑 test/npu/ 下全部 test_*.py。
    """
    parts = Path(modify_file).parts
    if not parts:
        return []
    top = parts[0]
    if top not in ("test", "docs", ".github", ".gitcode"):
        npu_dir = Path(repo_root) / "test" / "npu"
        if npu_dir.is_dir():
            return [normalize_test_path(str(p.relative_to(repo_root))) for p in npu_dir.rglob("test_*.py")]
    return []


# ==============================================================================
# 策略 3: DirectoryMappingStrategy — 源码目录 → 测试目录
# ==============================================================================

DIRECTORY_MAPPING = {
    "contrib": "test/contrib",
    "cpp_extension": "test/cpp_extensions",
    "distributed": "test/distributed",
    "fx": "test/test_fx.py",
    "optim": "test/optim",
    "profiler": "test/profiler",
    "onnx": "test/onnx",
    "utils": "test/test_utils.py",
    "testing": "test/test_testing.py",
    "jit": "test/test_jit.py",
    "rpc": "test/distributed/rpc",
    "meta": "test/test_fake_tensor.py",
}


def _get_module_name(modify_file: str) -> str:
    """从 torch_npu 变更文件中提取二级模块名 (参照 DirectoryMappingStrategy.get_module_name)。"""
    parts = Path(modify_file).parts
    if len(parts) >= 2 and parts[0] == "torch_npu":
        module_name = parts[1]
        if module_name == "csrc" and len(parts) >= 3:
            module_name = parts[2]
        # rpc 特殊处理
        for p in parts:
            if p == "rpc":
                module_name = "rpc"
        if module_name == "utils" and len(parts) >= 3 and parts[2] == "cpp_extension.py":
            module_name = "cpp_extension"
        return module_name
    return ""


def strategy_directory_mapping(modify_file: str, repo_root: str) -> List[str]:
    """
    torch_npu/<module>/ 变更 → 找到映射的测试文件/目录 → 返回里面的 test_*.py。
    """
    if modify_file.split("/")[0] != "torch_npu":
        return []

    module_name = _get_module_name(modify_file)
    file_stem = Path(modify_file).stem

    mapped_paths = []
    for name in (module_name, file_stem):
        if name in DIRECTORY_MAPPING:
            mapped_paths.append(DIRECTORY_MAPPING[name])

    result = []
    for mapped in mapped_paths:
        full = Path(repo_root) / mapped
        if full.is_file():
            result.append(normalize_test_path(mapped))
        elif full.is_dir():
            result.extend(
                normalize_test_path(str(p.relative_to(repo_root)))
                for p in sorted(full.rglob("test_*.py"))
            )
    return sorted(set(result))


# ==============================================================================
# 策略 4: OpStrategy — 算子 KernelNpu 文件名解析
# ==============================================================================

def strategy_op(modify_file: str, repo_root: str) -> List[str]:
    """
    文件名包含 KernelNpu → 从文件名提取关键字 → 用 glob 查找匹配的测试文件。
    例: BinaryCrossEntropyWithLogitsBackwardKernelNpu.cpp
         → Binary, Cross, Entropy, With, Logits, Backward
         → *binary*cross*entropy*with*logits*backward*
    """
    filename = Path(modify_file).name
    if "KernelNpu" not in filename:
        return []

    feature = filename.split("KernelNpu")[0]
    words = re.findall(r"[A-Z][^A-Z]*", feature)
    if not words:
        return []

    pattern = "*" + "*".join(w.lower() for w in words) + "*"

    test_dir = Path(repo_root) / "test"
    matches = []
    for f in test_dir.rglob("test_*.py"):
        rel = str(f.relative_to(repo_root))
        if fnmatch.fnmatch(rel, pattern):
            matches.append(normalize_test_path(rel))
    return sorted(set(matches))


# ==============================================================================
# 主流程
# ==============================================================================

def is_source_file(path: str) -> bool:
    """判断是否需要重新编译 torch_npu。只有 C++/构建脚本/CMake 变更才需要。"""
    return (
        path.endswith((".cpp", ".h", ".cu"))
        or "CMakeLists.txt" in path
        or path.startswith("ci/build.sh")
        or path in ("setup.py", "setup.cfg", "pyproject.toml")
    )


def main():
    # 从 stdin 读取全量变更文件 (由 detect_changed_tests.sh 传入)
    changed_files = [line.strip() for line in sys.stdin if line.strip()]

    repo_root = os.getcwd()
    all_test_files = set()
    need_rebuild = False

    strategies = [
        ("test_file", strategy_test_file),
        ("core", strategy_core),
        ("directory_mapping", strategy_directory_mapping),
        ("op", strategy_op),
    ]

    for modify_file in changed_files:
        for strategy_name, func in strategies:
            try:
                matched = func(modify_file, repo_root)
                for f in matched:
                    all_test_files.add(f)
                    print(f"  [{strategy_name}] {modify_file} → {f}", file=sys.stderr)
            except Exception:
                pass

        if is_source_file(modify_file):
            need_rebuild = True

    # 输出结果 (stdout, 供 shell 读取)
    has_test_changes = len(all_test_files) > 0
    test_files_str = ",".join(sorted(all_test_files))

    print(f"has_test_changes={'true' if has_test_changes else 'false'}")
    print(f"test_files={test_files_str}")
    print(f"need_rebuild={'true' if need_rebuild else 'false'}")
    print(f"strategy_count={len(all_test_files)}")


if __name__ == "__main__":
    main()
