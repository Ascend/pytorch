__all__ = ["cache_compile", "readable_cache"]


def cache_compile(func, *, dynamic: bool = True, cache_dir=None, global_rank=None, tp_rank=None, pp_rank=None, **kwargs):
    import npugraph_ex
    from npugraph_ex.configs import compiler_config

    compile_config = npugraph_ex.CompilerConfig()
    compile_config.mode = "npugraph_ex"
    compiler_config._process_kwargs_options(compile_config, kwargs)
    return npugraph_ex.inference.cache_compile(func, config=compile_config, dynamic=dynamic, cache_dir=cache_dir,
                                            global_rank=global_rank, tp_rank=tp_rank, pp_rank=pp_rank, **kwargs)


def readable_cache(cache_bin, print_output=True, file=None):
    import npugraph_ex
    return npugraph_ex.inference.readable_cache(cache_bin, print_output=print_output, file=file)