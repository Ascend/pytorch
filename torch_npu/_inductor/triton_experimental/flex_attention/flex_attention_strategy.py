"""Options used by the experimental FlexAttention lowering."""

from types import SimpleNamespace


class FlexAttentionBackendStrategy(SimpleNamespace):
    def prepare_kernel_options(self, kernel_options, *, score_mod_graph=None, mask_mod_graph=None):
        return self.option_transform(
            kernel_options, score_mod_graph=score_mod_graph, mask_mod_graph=mask_mod_graph
        )
