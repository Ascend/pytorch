__all__ = ["compile_fx", "register_replacement"]

from collections.abc import Collection, Generator, Iterable, Mapping, Sequence
from typing import Any, Callable, NoReturn, Optional, Protocol, TypeVar, Union, Match, List

try:
    from torch._inductor.pattern_matcher import fwd_only, SearchFn, ReplaceFn, TraceFn, PatternExpr
except ImportError:
    from torch._inductor.pattern_matcher import inference_graph as fwd_only

from . import inference
from . import scope


def compile_fx(gm, example_inputs=None, options=None):
    import npugraph_ex
    from npugraph_ex.configs import npugraphex_config

    compiler_config = npugraph_ex.CompilerConfig()
    compiler_config.mode = "npugraph_ex"
    npugraphex_config._process_kwargs_options(compiler_config, {"options": {} if options is None else options})
    compiler = npugraph_ex.get_compiler(compiler_config)
    return compiler(gm, example_inputs)


def _return_true(match: Match):
    return True


def register_replacement(search_fn: SearchFn, replace_fn: ReplaceFn, example_inputs: Iterable[Any],
                         trace_fn: TraceFn = fwd_only, extra_check: Callable[[Match], bool] = _return_true,
                         search_fn_pattern: Union[PatternExpr, None] = None,
                         scalar_workaround: Union[dict[str, Union[float, int]], None] = None,
                         skip_duplicates: bool = False):
    import npugraph_ex
    return npugraph_ex.patterns.pattern_pass_manager.register_replacement(search_fn, replace_fn, example_inputs,
                                                                       trace_fn=trace_fn, extra_check=extra_check,
                                                                       search_fn_pattern=search_fn_pattern,
                                                                       scalar_workaround=scalar_workaround, 
                                                                       skip_duplicates=skip_duplicates)