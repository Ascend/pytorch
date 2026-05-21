"""
Add validation cases for torch.fx lazy graph formatting APIs on NPU:

1. test/test_fx.py from PyTorch community contains broad FX validations and
   triggers historical lint issues when modified, so this file is added.
2. This file validates torch.fx._utils.lazy_format_graph_code output formatting
   for traced GraphModule instances.
"""

import torch
from torch.fx import symbolic_trace
from torch.fx._utils import lazy_format_graph_code
from torch.testing._internal.common_utils import run_tests, TestCase


class TestLazyFormatGraphCode(TestCase):
    def test_lazy_format_graph_code(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = symbolic_trace(MyModule())
        graph_code = str(lazy_format_graph_code("fx lazy graph", gm, maybe_id=1))

        self.assertIn("TRACED GRAPH", graph_code)
        self.assertIn("===== fx lazy graph 1 =====", graph_code)
        self.assertIn("def forward", graph_code)


if __name__ == "__main__":
    run_tests()
