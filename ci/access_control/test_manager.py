import os
import re
from pathlib import Path
import torch_npu
from .strategy import (
    TestFileStrategy,
    CopyOptStrategy,
    OpStrategy,
    DirectoryMappingStrategy,
    CoreTestStrategy
)
from .constants import (
    BASE_DIR, TEST_DIR, SLOW_TEST_BLOCKLIST, INCLUDE_FILES
)


def get_test_torch_version_path():
    torch_npu_version = torch_npu.__version__
    version_list = torch_npu_version.split('.')
    if len(version_list) > 2:
        return f'test_v{version_list[0]}r{version_list[1]}_ops'
    else:
        raise RuntimeError("Invalid torch_npu version.")


class TestMgr:
    def __init__(self):
        self.modify_files = []
        self.test_files = {
            'ut_files': [],
            'op_ut_files': []
        }

    def load(self, modify_files, world_size):
        with open(modify_files) as f:
            for line in f:
                if world_size != 0 and "test/distributed/" in line:
                    continue
                line = line.strip()
                self.modify_files.append(line)

    def analyze(self):
        for modify_file in self.modify_files:
            self.test_files['ut_files'] += TestFileStrategy().identify(modify_file)
            self.test_files['ut_files'] += CopyOptStrategy().identify(modify_file)
            self.test_files['ut_files'] += OpStrategy().identify(modify_file)
            self.test_files['ut_files'] += DirectoryMappingStrategy().identify(modify_file)
            self.test_files['op_ut_files'] += OpStrategy().identify(modify_file)
            self.test_files['ut_files'] += CoreTestStrategy().identify(modify_file)
        unique_files = sorted(set(self.test_files['ut_files']))

        exist_ut_file = [
            changed_file
            for changed_file in unique_files
            if Path(changed_file).exists()
        ]
        self.test_files['ut_files'] = exist_ut_file
        self.exclude_test_files(slow_files=SLOW_TEST_BLOCKLIST)

    def load_core_ut(self):
        self.test_files['ut_files'] += [str(i) for i in (BASE_DIR / 'test/npu').rglob('test_*.py')]

    def load_distributed_ut(self):
        self.test_files['ut_files'] += [str(i) for i in (BASE_DIR / 'test/distributed').rglob('test_*.py')]

    def load_op_plugin_ut(self):
        version_path = get_test_torch_version_path()
        file_hash = {}
        for file_path in (BASE_DIR / 'third_party/op-plugin/test').rglob('test_*.py'):
            if str(file_path.parts[-2]) in [version_path, "test_custom_ops", "test_base_ops"]:
                file_name = str(file_path.name)
                if file_name in file_hash:
                    if str(file_path.parts[-2]) == version_path:
                        self.test_files['ut_files'].remove(file_hash[file_name])
                        file_hash[file_name] = str(file_path)
                        self.test_files['ut_files'].append(str(file_path))
                else:
                    file_hash[file_name] = str(file_path)
                    self.test_files['ut_files'].append(str(file_path))

    def load_all_ut(self, include_distributed_case=False, include_op_plugin_case=False):
        all_files = [str(i) for i in (BASE_DIR / 'test').rglob('test_*.py') if 'distributed' not in str(i)]
        self.test_files['ut_files'] = all_files
        if include_distributed_case:
            self.load_distributed_ut()
        if include_op_plugin_case:
            if os.path.exists(BASE_DIR / 'third_party/op-plugin/test'):
                self.load_op_plugin_ut()
            else:
                raise Exception("The path of op-plugin did not exist, check whether it had been pulled.")

    def split_test_files(self, rank, world_size):
        if rank > world_size:
            raise Exception(f'rank {rank} is greater than world_size {world_size}')

        def ordered_split(files, start, step):
            return sorted(files)[start::step] if files else []

        # node rank starts from 1
        self.test_files['ut_files'] = ordered_split(self.test_files['ut_files'], rank - 1, world_size)
        self.test_files['op_ut_files'] = ordered_split(self.test_files['op_ut_files'], rank - 1, world_size)

    def exclude_test_files(self, slow_files=None, not_run_files=None, mode="slow_test"):
        """
        Args:
            slow_files: slow test files.
            not_run_files: not_run_directly test files.
            mode: "slow_test" or "not_run_directly". Default: "slow_test".

        Returns:
        """
        def remove_test_files(key):
            block_list = not_run_files.keys() if mode == "not_run_directly" else slow_files
            for test_name in block_list:
                test_files_copy = self.test_files.get(key, [])[:]
                for test_file in test_files_copy:
                    is_remove = False
                    if mode == "slow_test" and test_name + ".py" in test_file:
                        print(f'Excluding slow test: {test_name}')
                        is_remove = True
                    if mode == "not_run_directly":
                        # remove not run directly files
                        is_folder_in = not test_name.endswith(".py") and "/" + test_name + "/" in test_file
                        is_file_in = test_name.endswith(".py") and test_name in test_file
                        if is_folder_in or is_file_in:
                            if test_name in ["jit"] and re.search("|".join(INCLUDE_FILES), test_file):
                                continue
                            is_remove = True
                            # add instead file
                            instead_files.add(str(TEST_DIR.joinpath(not_run_files[test_name])))
                    if is_remove:
                        test_files = self.test_files.get(key, [])
                        if test_file in test_files:
                            test_files.remove(test_file)

        for test_files_key in self.test_files.keys():
            instead_files = set()
            remove_test_files(test_files_key)
            for instead_file in instead_files:
                if instead_file not in self.test_files[test_files_key]:
                    self.test_files[test_files_key].append(instead_file)

    def get_test_files(self):
        return self.test_files

    def print_modify_files(self):
        print("modify files:")
        for modify_file in self.modify_files:
            print(modify_file)

    def print_ut_files(self):
        print("ut files:")
        for ut_file in self.test_files.get('ut_files', []):
            print(ut_file)

    def print_op_ut_files(self):
        print("op ut files:")
        for op_ut_file in self.test_files.get('op_ut_files', []):
            print(op_ut_file)
