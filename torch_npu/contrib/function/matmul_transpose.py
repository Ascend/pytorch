__all__ = ["matmul_transpose"]


from torch_npu.contrib.function._matmul_transpose import MatmulApply


matmul_transpose = MatmulApply.apply
