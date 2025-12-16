import types
from typing import Optional, Union

import torchair
from torchair.configs import npugraphex_config


class CompiledModel(torchair.inference._cache_compiler.CompiledModel):
    def __init__(self, meta: Union[torchair.inference._cache_compiler.ModelCacheMeta, types.FunctionType, types.MethodType]):
        super().__init__(meta)

    @staticmethod
    def get_cache_bin(func, *, dynamic: bool = True, cache_dir: Optional[str] = None, global_rank: Optional[int] = None,
                      tp_rank: Optional[int] = None, pp_rank: Optional[int] = None, **kwargs) -> str:
        compile_config = torchair.CompilerConfig()
        compile_config.mode = "reduce-overhead"
        npugraphex_config._process_kwargs_options(compile_config, kwargs)
        return torchair.inference._cache_compiler.CompiledModel.get_cache_bin(func, config=compile_config, dynamic=dynamic, cache_dir=cache_dir,
                                     tp_rank=tp_rank, pp_rank=pp_rank, ge_cache=False, global_rank=global_rank, **kwargs)


class ModelCacheSaver(torchair.inference._cache_compiler.ModelCacheSaver):
    def __init__(self, func: Union[types.FunctionType, types.MethodType], cache_bin, *, dynamic: bool = True,
                 decompositions: Optional[dict] = None, **kwargs):
        compile_config = torchair.CompilerConfig()
        compile_config.mode = "reduce-overhead"
        npugraphex_config._process_kwargs_options(compile_config, kwargs)
        super().__init__(func, config=compile_config, cache_bin=cache_bin, dynamic=dynamic, decompositions=decompositions, ge_cache=False)