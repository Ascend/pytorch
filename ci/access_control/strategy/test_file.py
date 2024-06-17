import re
from pathlib import Path
from .base import AccurateTest


class TestFileStrategy(AccurateTest):
    """
    Determine whether the modified files are test cases
    """

    def identify(self, modify_file):
        is_test_file = str(Path(modify_file).parts[0]) == "test" \
                       and re.match("test_(.+).py", Path(modify_file).name)
        return [(str(self.base_dir.joinpath(modify_file)))] if is_test_file else []
