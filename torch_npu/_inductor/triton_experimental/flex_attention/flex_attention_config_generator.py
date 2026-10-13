"""Experimental tiling policy using the default FlexAttention generator."""

from torch_npu._inductor.kernel.flex_attention_config_generator import (
    FlexAttentionConfig,
    FlexAttentionConfigGenerator as _DefaultConfigGenerator,
    FlexMode,
)


class FlexAttentionConfigGenerator(_DefaultConfigGenerator):
    def _generate_block_combinations(self):
        return [
            FlexAttentionConfig(block_m=m, block_n=n, num_warps=4,
                                num_stages=2 if self.mode == FlexMode.BWDDKDV else 1)
            for m in self.valid_block_m for n in self.valid_block_n
            if self._is_valid_block_pair(m, n)
        ]


def _configs(q_block, kv_block, mode, options):
    return FlexAttentionConfigGenerator(
        sparse_q_block_size=q_block, sparse_kv_block_size=kv_block,
        mode=mode, kernel_options=options,
    ).generate_configs()


def generate_fwd_candidate_configs(sparse_q_block_size, sparse_kv_block_size, kernel_options=None):
    return _configs(sparse_q_block_size, sparse_kv_block_size, FlexMode.FWD, kernel_options)


def generate_bwd_candidate_configs(sparse_q_block_size, sparse_kv_block_size, mode, kernel_options=None):
    if mode not in (FlexMode.BWD, FlexMode.BWDDQ, FlexMode.BWDDKDV):
        raise ValueError(f"unsupported backward flex attention mode: {mode}")
    return _configs(sparse_q_block_size, sparse_kv_block_size, mode, kernel_options)
