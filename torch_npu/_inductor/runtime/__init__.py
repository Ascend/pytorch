import logging
from .autotune_cache import patch_load_cached_autotuning
from .hints import patch_create_device_properties
try:
    from .triton_helpers import *  # noqa: F403
except Exception as e:
    logging.warning("import triton_helpers error: %s", e)  # noqa: G200
from .triton_heuristics import patch_triton_heuristics_cached_autotune
