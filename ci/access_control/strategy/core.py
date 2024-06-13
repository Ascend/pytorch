from pathlib import Path
from .base import AccurateTest


class CoreTestStrategy(AccurateTest):
    """
    Determine whether the core tests should be runned
    """
    def __init__(self):
        super().__init__()
        self.block_list = ['test', 'docs']
        self.core_test_cases = [str(i) for i in (self.base_dir / 'test/npu').rglob('test_*.py')]

    def identify(self, modify_file):
        modified_module = str(Path(modify_file).parts[0])
        if modified_module not in self.block_list:
            return self.core_test_cases
        return []
