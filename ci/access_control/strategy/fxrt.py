from pathlib import Path
from .base import AccurateTest


class FxrtStrategy(AccurateTest):
    """
    Map the modified files of the fxrt subpackage to its test cases.

    fxrt lives in the top-level fxrt directory and ships as torch_npu.fxrt, so
    DirectoryMappingStrategy does not reach it: that one only maps paths under
    torch_npu.
    """

    def identify(self, modify_file):
        if str(Path(modify_file).parts[0]) != 'fxrt':
            return []
        return [str(i) for i in (self.test_dir / 'fxrt').rglob('test_*.py')]
