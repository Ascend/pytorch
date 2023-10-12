import os
import stat

import torchgen.gen


def _write_if_changed_security(self, filename: str, contents: str) -> None:
    old_contents: Optional[str]
    try:
        with open(filename, 'r') as f:
            old_contents = f.read()
    except IOError:
        old_contents = None
    if contents != old_contents:
        with os.fdopen(os.open(filename, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), "w") as f:
            f.write(contents)


def apply_codegen_patches():
    torchgen.gen.FileManager._write_if_changed = _write_if_changed_security
    

apply_codegen_patches()
