"""torch_npu package version resolution.

The package version is driven entirely by the ``TORCH_VERSION`` environment
variable, which ``build.sh`` sets from its ``--torch=`` argument. The value may
carry a PEP 440 post-release suffix (e.g. ``2.13.0.post1``) to denote a
follow-up build against a given PyTorch line. When ``TORCH_VERSION`` is unset,
the version of the PyTorch installed in the current environment is used.

``version.txt`` is documentation only (a human-readable list of the versions
this source tree can produce) and is *not* read by any code.
"""

import os


def get_version() -> str:
    """Return the torch_npu package version.

    Honors the ``TORCH_VERSION`` environment variable; falls back to the
    installed PyTorch version (with any local tag, e.g. ``+cpu``, stripped)
    when it is unset.
    """
    version = os.environ.get("TORCH_VERSION")
    if version:
        return version.strip()
    import torch
    return torch.__version__.split("+")[0]
