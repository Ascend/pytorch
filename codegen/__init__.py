import os
import stat

import torchgen.gen
from codegen.utils import PathManager


def _write_if_changed_security(self, filename: str, contents: str) -> None:
    old_contents: Optional[str]
    filepath = os.path.realpath(filename)
    try:
        with open(filepath, 'r') as f:
            old_contents = f.read()
    except IOError:
        old_contents = None
    if contents != old_contents:
        with os.fdopen(os.open(filepath, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), "w") as f:
            f.write(contents)
        os.chmod(filepath, 0o550)


def apply_codegen_patches():
    torchgen.gen.FileManager._write_if_changed = _write_if_changed_security
    

apply_codegen_patches()
