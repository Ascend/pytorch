__all__ = ["cache_compile", "readable_cache"]


def cache_compile(func, *, dynamic: bool = True, cache_dir=None, global_rank=None, tp_rank=None, pp_rank=None, **kwargs):
    import torchair
    from torchair.configs import npugraphex_config

    compile_config = torchair.CompilerConfig()
    compile_config.mode = "reduce-overhead"
    npugraphex_config._process_kwargs_options(compile_config, kwargs)
    return torchair.inference.cache_compile(func, config=compile_config, dynamic=dynamic, cache_dir=cache_dir,
                                            global_rank=global_rank, tp_rank=tp_rank, pp_rank=pp_rank, **kwargs)


def readable_cache(cache_bin, print_output=True, file=None):
    import torchair
    return torchair.inference.readable_cache(cache_bin, print_output=print_output, file=file)