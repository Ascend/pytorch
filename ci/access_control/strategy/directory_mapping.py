from pathlib import Path
from .base import AccurateTest


class DirectoryMappingStrategy(AccurateTest):
    """
    Map the modified files to the corresponding test cases
    """
    mapping_list = {
        'contrib': 'test/contrib',
        'cpp_extension': 'test/cpp_extensions',
        'distributed': 'test/distributed',
        'fx': 'test/test_fx.py',
        'optim': 'test/optim',
        'profiler': 'test/profiler',
        'onnx': 'test/onnx',
        'utils': 'test/test_utils.py',
        'testing': 'test/test_testing.py',
        'jit': 'test/test_jit.py',
        'rpc': 'test/distributed/rpc',
        'dynamo': 'test/dynamo',
        'checkpoint': 'test/distributed/checkpoint',
    }

    @staticmethod
    def get_module_name(modify_file):
        module_name = str(Path(modify_file).parts[1])
        if module_name == 'csrc':
            module_name = str(Path(modify_file).parts[2])
        for part in Path(modify_file).parts:
            if part == 'rpc' or part == 'checkpoint':
                module_name = part
        if module_name == 'utils' and Path(modify_file).parts[2] == 'cpp_extension.py':
            module_name = 'cpp_extension'
        if module_name == 'utils' and 'dynamo' in Path(modify_file).parts[2]:
            module_name = 'dynamo'
        return module_name

    def identify(self, modify_file):
        current_all_ut_path = []
        if str(Path(modify_file).parts[0]) == 'torch_npu':
            mapped_ut_path = []
            module_name = self.get_module_name(modify_file)
            if module_name in self.mapping_list:
                mapped_ut_path.append(self.mapping_list[module_name])
            file_name = str(Path(modify_file).stem)
            if file_name in self.mapping_list:
                mapped_ut_path.append(self.mapping_list[file_name])

            for mapped_path in mapped_ut_path:
                if Path.is_file(self.base_dir.joinpath(mapped_path)):
                    current_all_ut_path.append(str(self.base_dir.joinpath(mapped_path)))
                else:
                    current_all_ut_path += [str(i) for i in (self.base_dir.joinpath(mapped_path)).glob('test_*.py')]
        return current_all_ut_path
