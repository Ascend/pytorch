import os
from unittest import mock

os.environ["NPU_INDUCTOR_FALLBACK_LIST"] = "allfallback"

import torch
from torch.testing._internal.common_utils import run_tests
from torch._inductor.utils import run_and_get_code

from testutils import TestUtils


class TestAllFallback(TestUtils):

    def add_op(self, x, y):
        return x + y + 1

    def test_all_fallback_detection_mlir(self):

        compiled_add = torch.compile(self.add_op, backend="inductor", options={"npu_backend": "mlir"})

        x = torch.randn(4, 4, dtype=torch.float32).to("npu")
        y = torch.randn(4, 4, dtype=torch.float32).to("npu")

        _, codes = run_and_get_code(compiled_add, x, y)

        self.assertTrue('mlir_fused' not in codes[0])

    def test_all_fallback_detection_triton(self):

        compiled_add = torch.compile(self.add_op, backend="inductor")

        x = torch.randn(4, 4, dtype=torch.float32).to("npu")
        y = torch.randn(4, 4, dtype=torch.float32).to("npu")

        _, codes = run_and_get_code(compiled_add, x, y)

        self.assertTrue('_fused_' not in codes[0])

    def test_allfallback_disables_graph_partition(self):
        from torch_npu._inductor.utils import disable_graph_partition_for_allfallback

        with mock.patch.object(torch._inductor.config, "graph_partition", True):
            disable_graph_partition_for_allfallback()
            self.assertFalse(torch._inductor.config.graph_partition)

    def test_normal_fallback_keeps_graph_partition(self):
        from torch_npu._inductor.utils import disable_graph_partition_for_allfallback

        with (
            mock.patch.dict(
                os.environ,
                {"NPU_INDUCTOR_FALLBACK_LIST": "aten.add"},
            ),
            mock.patch.object(torch._inductor.config, "graph_partition", True),
        ):
            disable_graph_partition_for_allfallback()
            self.assertTrue(torch._inductor.config.graph_partition)


if __name__ == "__main__":
    run_tests()
