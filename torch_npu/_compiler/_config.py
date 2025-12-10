import sys
from torch.utils._config_module import Config, install_config_module

# force a python GC before recording npugraphs
force_npugraph_gc: bool = Config(env_name_default="TORCH_NPUGRAPH_GC", default=False)
"""
If True (the backward-compatible behavior) then gc.collect() before recording any npugraph.
"""

install_config_module(sys.modules[__name__])