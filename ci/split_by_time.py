# -*- coding: UTF-8 -*-

"""
基于耗时数据的用例拆分模块。

根据 time_data.json 中的用例文件执行耗时数据和类级执行耗时数据，
将用例拆分成 world_size 份，目标是使拆分后的每一个机器上的总耗时尽量相等。
"""

import ast
import json
import os
from pathlib import Path

from access_control import TEST_DIR, NETWORK_OPS_DIR

# 无耗时数据时文件级默认耗时（秒）
DEFAULT_FILE_TIME = 10.0
# 慢文件无类级数据时每个类的默认耗时（秒）
DEFAULT_CLASS_TIME = 60.0
# 慢文件阈值（秒），总耗时 >= 此值则进行类级拆分
SLOW_FILE_THRESHOLD = 300.0


def get_op_name(ut_file):
    """从用例文件路径获取操作名（与 exec_ut 中逻辑一致）"""
    op_name = str(Path(ut_file).name).split('.')[0]
    return op_name[5:] if op_name.startswith("test_") else op_name


def get_test_key(ut_file, ut_type):
    """
    生成与 time_data.json 中 timedata 键匹配的标识。

    与 exec_ut 中 ut_info 的生成逻辑保持一致：
    - op_ut_files: "test_ops _" + op_name
    - ut_files (op-plugin): 相对 NETWORK_OPS_DIR 的路径（去 .py）
    - ut_files (其他): 相对 TEST_DIR 的路径（去 .py）
    """
    if ut_type == "op_ut_files":
        return "test_ops _" + get_op_name(ut_file)
    if 'op-plugin' in str(Path(ut_file)):
        return Path(ut_file).relative_to(NETWORK_OPS_DIR).as_posix()[:-3]
    return Path(ut_file).relative_to(TEST_DIR).as_posix()[:-3]


def load_and_validate_time_data(file_path):
    """
    加载并验证 time_data.json 文件。

    验证条件：
    1. 文件存在
    2. 内容为有效 JSON 且为字典
    3. 包含 timedata 字段且为字典
    4. timedata 中每项为字典且含 total_time 数值
    5. 若存在 classes 字段，则必须为字典且每个 class_time 为数值

    Returns:
        timedata 字典（成功）或 None（失败，触发回退）
    """
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.loads(f.read())
        if not isinstance(content, dict):
            return None
        timedata = content.get('timedata')
        if not isinstance(timedata, dict):
            return None
        # 验证每项格式
        for key, value in timedata.items():
            if not isinstance(value, dict):
                return None
            total_time = value.get('total_time')
            if total_time is None or not isinstance(total_time, (int, float)):
                return None
            # 验证 classes 字段（若存在）结构
            classes = value.get('classes')
            if classes is not None:
                if not isinstance(classes, dict):
                    return None
                for class_time in classes.values():
                    if not isinstance(class_time, (int, float)):
                        return None
        return timedata
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def discover_test_classes(file_path):
    """
    用 AST 解析发现测试文件中以 Test 开头的类名列表。

    用于慢文件（>= 300秒）无类级耗时数据时，发现类名进行类级拆分。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                classes.append(node.name)
        return classes
    except Exception:
        return []


def build_split_units(test_files, timedata):
    """
    构建拆分单元列表。

    拆分规则：
    - op_ut_files 始终按文件级拆分（通过 -k 过滤运行，类级拆分不适用）
    - 无耗时数据 → 文件级单元，时间=10秒
    - total_time < 300 → 文件级单元，时间=total_time
    - total_time >= 300 且有 classes → 类级单元，每个类单独一个单元
    - total_time >= 300 但无 classes → AST发现类，类级单元，每个类时间=60秒

    Returns:
        list of (ut_type, ut_file, class_name_or_None, estimated_time)
    """
    units = []
    for ut_type, ut_files in test_files.items():
        for ut_file in ut_files:
            test_key = get_test_key(ut_file, ut_type)
            time_data = timedata.get(test_key)

            # op_ut_files 始终按文件级拆分
            if ut_type == "op_ut_files":
                if time_data:
                    est_time = time_data.get('total_time', DEFAULT_FILE_TIME)
                else:
                    est_time = DEFAULT_FILE_TIME
                units.append((ut_type, ut_file, None, est_time))
                continue

            # ut_files 的处理
            if time_data is None:
                # 无耗时数据，文件级，默认10秒
                units.append((ut_type, ut_file, None, DEFAULT_FILE_TIME))
            else:
                total_time = time_data.get('total_time', DEFAULT_FILE_TIME)
                classes = time_data.get('classes')

                if total_time < SLOW_FILE_THRESHOLD:
                    # 非慢文件，文件级
                    units.append((ut_type, ut_file, None, total_time))
                elif classes and isinstance(classes, dict):
                    # 慢文件且有类级数据，按类级拆分
                    for class_name, class_time in classes.items():
                        if not isinstance(class_time, (int, float)):
                            continue
                        units.append((ut_type, ut_file, class_name, class_time))
                else:
                    # 慢文件但无类级数据，AST发现类，每个类默认60秒
                    discovered = discover_test_classes(ut_file)
                    if discovered:
                        for class_name in discovered:
                            units.append((ut_type, ut_file, class_name, DEFAULT_CLASS_TIME))
                    else:
                        # AST没找到类，按文件级
                        units.append((ut_type, ut_file, None, total_time))

    return units


def split_by_time(test_files, timedata, rank, world_size):
    """
    基于耗时数据拆分用例，使每个机器上的总耗时尽量相等。

    使用 LPT（Longest Processing Time first）贪心算法：
    1. 将所有拆分单元按预估时间降序排序
    2. 每个单元分配给当前总时间最小的机器

    Args:
        test_files: {'ut_files': [...], 'op_ut_files': [...]}
        timedata: time_data.json 中的 timedata 字典
        rank: 当前机器的序号（从1开始）
        world_size: 机器总数

    Returns:
        {
            'ut_files': [file_path, ...],
            'op_ut_files': [file_path, ...],
            'ut_classes': {file_path: [class_name, ...]},
            'op_ut_classes': {file_path: [class_name, ...]}
        }
    """
    if rank > world_size:
        raise Exception(f'rank {rank} is greater than world_size {world_size}')

    units = build_split_units(test_files, timedata)

    # 按预估时间降序排序（LPT算法）
    units.sort(key=lambda x: x[3], reverse=True)

    # LPT贪心：每个单元分配给当前总时间最小的机器
    machine_loads = [0.0] * world_size
    machine_units = [[] for _ in range(world_size)]

    for unit in units:
        min_machine = machine_loads.index(min(machine_loads))
        machine_units[min_machine].append(unit)
        machine_loads[min_machine] += unit[3]

    # 打印各机器负载情况
    print("***** Time-based split result:")
    for i, (load, mus) in enumerate(zip(machine_loads, machine_units)):
        print(f"  Machine {i + 1}: estimated_time={load:.1f}s, units={len(mus)}")

    # 获取当前 rank 的分配（rank从1开始）
    my_units = machine_units[rank - 1]

    # 构建结果
    result = {
        'ut_files': [],
        'op_ut_files': [],
        'ut_classes': {},
        'op_ut_classes': {}
    }

    for ut_type, ut_file, class_name, _ in my_units:
        if class_name is None:
            # 文件级
            if ut_file not in result[ut_type]:
                result[ut_type].append(ut_file)
        else:
            # 类级
            if ut_file not in result[ut_type]:
                result[ut_type].append(ut_file)
            classes_key = 'ut_classes' if ut_type == 'ut_files' else 'op_ut_classes'
            if ut_file not in result[classes_key]:
                result[classes_key][ut_file] = []
            result[classes_key][ut_file].append(class_name)

    return result
