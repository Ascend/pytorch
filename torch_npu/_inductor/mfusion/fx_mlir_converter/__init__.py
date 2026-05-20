"""FX ↔ Torch MLIR helpers for MFusion.

Importing this package must not execute ``import torch_mlir`` at module load time:
``test/npu/test_public_bindings.py::TestPublicBindings.test_modules_can_be_imported``
walks all submodules; gate jobs omit optional torch_mlir / mfusion installs.

Heavy symbols load on first access via ``__getattr__``.
"""

from typing import Any


__all__ = [
    "import_mlir_module_from_fx",
    "export_mlir_module_to_fx",
    "fake_tensor_propagate_mfusion_subgraph",
]


def __getattr__(name: str) -> Any:
    if name == "import_mlir_module_from_fx":
        from .fx_importer import import_mlir_module_from_fx

        return import_mlir_module_from_fx
    if name == "export_mlir_module_to_fx":
        from .fx_exporter import export_mlir_module_to_fx

        return export_mlir_module_to_fx
    if name == "fake_tensor_propagate_mfusion_subgraph":
        from .fx_exporter import fake_tensor_propagate_mfusion_subgraph

        return fake_tensor_propagate_mfusion_subgraph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
