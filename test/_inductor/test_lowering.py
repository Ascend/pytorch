import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
from torch_npu._inductor.lowering import LOWERING_OVERLOAD_OP, GENERATE_LIST, FALLBACK_LIST
from torch_npu._inductor.config import use_store_in_cat


class TestLowering(TestUtils):
    def test_embedding_cat_mode(self):
        lowering_embedding = (torch.ops.aten.embedding in GENERATE_LIST or torch.ops.aten.embedding not in FALLBACK_LIST)
        # If embedding is not overload for customop, use_store_in_cat=True
        if torch.ops.aten.embedding not in LOWERING_OVERLOAD_OP \
            and lowering_embedding:
            self.assertTrue(use_store_in_cat)
        # If embedding is overload for customop, use_store_in_cat=False
        if torch.ops.aten.embedding in LOWERING_OVERLOAD_OP\
            and lowering_embedding:
            self.assertFalse(use_store_in_cat)

instantiate_parametrized_tests(TestLowering)

if __name__ == "__main__":
    run_tests()
