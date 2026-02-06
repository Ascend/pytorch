__all__ = ["compile_fx", "register_replacement"]

from typing import Match

try:
    from torch._inductor.pattern_matcher import fwd_only
except ImportError:
    from torch._inductor.pattern_matcher import inference_graph as fwd_only

from . import experimental
from . import inference
from . import ops
from . import scope


def compile_fx(options: dict = None):
    import torchair
    from torchair.configs import npugraphex_config

    compiler_config = torchair.CompilerConfig()
    compiler_config.mode = "npugraph_ex"
    npugraphex_config._process_kwargs_options(compiler_config, {options: options})
    return torchair.get_compiler(compiler_config)


def _return_true(match: Match):
    return True


def register_replacement(search_fn, replace_fn, example_inputs, trace_fn=fwd_only, extra_check=_return_true,
                         search_fn_pattern=None, scalar_workaround=None, skip_duplicates=False):
    import torchair
    return torchair.patterns.pattern_pass_manager.register_replacement(search_fn, replace_fn, example_inputs,
                                                                       trace_fn=trace_fn, extra_check=extra_check,
                                                                       search_fn_pattern=search_fn_pattern,
                                                                       scalar_workaround=scalar_workaround, 
                                                                       skip_duplicates=skip_duplicates)