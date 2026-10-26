from torch._inductor import config
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files
from . import ascend_custom_passes
from .ascend_custom_passes import ASCEND_CUSTOME_PASS_REGISTER

class AscendCustomPostPass(CustomGraphPass):
    def __call__(self, graph):
        return ascend_custom_passes.run_register_post_custom_passes(graph)

    def uuid(self):
        # The registered passes depend on config switches, not file contents; key on them
        # so flipping a switch does not reuse a cached graph.
        registered = ",".join(sorted(
            fn.__name__
            for levels in ASCEND_CUSTOME_PASS_REGISTER.values()
            for fns in levels.values()
            for fn in fns
        ))
        return get_hash_for_files((__file__,), extra=registered)

def pre_grad_custom_pass_fuc():
    config.pre_grad_custom_pass = ascend_custom_passes.run_register_pre_custom_passes


def post_grad_custom_pass_fuc():
    config.post_grad_custom_post_pass = AscendCustomPostPass()
