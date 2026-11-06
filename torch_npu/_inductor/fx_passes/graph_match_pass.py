from torch._inductor import config
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files
from . import ascend_custom_passes
from .. import config as npu_config


class AscendCustomPostPass(CustomGraphPass):
    def __call__(self, graph):
        return ascend_custom_passes.run_register_post_custom_passes(graph)

    def uuid(self):
        # The registered pass set changes with feature flags / shutdown lists.
        # Include it and the implementation so an FX cache hit cannot silently
        # reuse the graph from before the passes or their implementations changed.
        from .ascend_custom_passes import ascend_graph_pass, index_reshape_index
        from .ascend_custom_passes.register_custom_pass import (
            ASCEND_CUSTOME_PASS_REGISTER,
        )
        from ..kernel import index_reshape_index as index_lowering

        names = tuple(
            (pass_type, level, tuple(fn.__name__ for fn in fns))
            for pass_type, levels in ASCEND_CUSTOME_PASS_REGISTER.items()
            for level, fns in levels.items()
        )
        files = (
            __file__,
            ascend_graph_pass.__file__,
            index_reshape_index.__file__,
            index_lowering.__file__,
        )
        return get_hash_for_files(
            files,
            extra=repr(
                (
                    names,
                    npu_config.enable_full_lowering_fallback.strip() == "allfallback",
                )
            ),
        )


def pre_grad_custom_pass_fuc():
    config.pre_grad_custom_pass = ascend_custom_passes.run_register_pre_custom_passes


def post_grad_custom_pass_fuc():
    config.post_grad_custom_post_pass = AscendCustomPostPass()
