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
        PathManager.remove_path_safety(filepath)
        with os.fdopen(os.open(filepath, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), "w") as f:
            f.write(contents)
        os.chmod(filepath, stat.S_IRUSR | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)


def apply_codegen_patches():
    torchgen.gen.FileManager._write_if_changed = _write_if_changed_security
    

apply_codegen_patches()
