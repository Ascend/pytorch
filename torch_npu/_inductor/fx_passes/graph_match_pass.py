from torch._inductor import config
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files

from . import ascend_custom_passes


class AscendCustomPostPass(CustomGraphPass):
    def __call__(self, graph):
        return ascend_custom_passes.run_register_post_custom_passes(graph)

    def uuid(self):
        return get_hash_for_files((__file__,))


def pre_grad_custom_pass_fuc():
    config.pre_grad_custom_pass = ascend_custom_passes.run_register_pre_custom_passes


def post_grad_custom_pass_fuc():
    config.post_grad_custom_post_pass = AscendCustomPostPass()
