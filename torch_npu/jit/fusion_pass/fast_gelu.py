import torch


def fast_gelu_pass(jit_mod):
    pattern = """
        graph(%x):
            %str = prim::Constant[value='none']()
            %out = aten::gelu(%x, %str)
            return (%out)
    """

    replacement = """
        graph(%x):
            %out = npu::fast_gelu(%x)
            return (%out)
    """
    
    torch._C._jit_pass_custom_pattern_based_rewrite_graph(pattern, replacement, jit_mod.graph)
