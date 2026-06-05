"""
Add validation cases for torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts on NPU.

1. PyTorch community lacks sufficient and direct API validations for PropagateUnbackedSymInts,
   so this file is added.
2. This file validates PropagateUnbackedSymInts.run, PropagateUnbackedSymInts.run_node,
   PropagateUnbackedSymInts.placeholder, PropagateUnbackedSymInts.output, and rebind_unbacked.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._dynamo.utils import detect_fake_mode
from torch.fx.experimental.symbolic_shapes import PropagateUnbackedSymInts, rebind_unbacked

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
torch.zeros(3, 4).to(device_type)


class TestPropagateUnbackedSymInts(TestCase):

    def _create_exported_model(self):

        class M(torch.nn.Module):

            def forward(self, x: torch.Tensor):
                return torch.nonzero(x)

        inp = (torch.tensor([1, 0, 1, 0]).to(device_type),)
        gm = torch.export.export(M(), inp, strict=True).module()
        fake_inputs = [
            node.meta.get("val") for node in gm.graph.nodes if node.op == "placeholder"
        ]
        fake_mode = detect_fake_mode(fake_inputs)
        return gm, fake_inputs, fake_mode

    def test_propagate_unbacked_symints_run(self):
        """Test PropagateUnbackedSymInts.run with NPU tensor."""
        gm, fake_inputs, fake_mode = self._create_exported_model()
        with fake_mode:
            result = PropagateUnbackedSymInts(gm).run(*fake_inputs)
        self.assertIsNotNone(result)

    def test_propagate_unbacked_symints_run_node(self):
        """Test PropagateUnbackedSymInts.run_node with NPU tensor."""

        class RunNodeCapturingInterpreter(PropagateUnbackedSymInts):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.captured_results = {}

            def run_node(self, n):
                result = super().run_node(n)
                self.captured_results[n] = result
                return result

        gm, fake_inputs, fake_mode = self._create_exported_model()
        with fake_mode:
            interpreter = RunNodeCapturingInterpreter(gm)
            result = interpreter.run(*fake_inputs)
            self.assertIsNotNone(result)
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    self.assertIn(node, interpreter.captured_results)

    def test_propagate_unbacked_symints_placeholder(self):
        """Test PropagateUnbackedSymInts.placeholder with NPU tensor."""
        gm, fake_inputs, fake_mode = self._create_exported_model()
        with fake_mode:
            interpreter = PropagateUnbackedSymInts(gm)
            interpreter.args_iter = iter(fake_inputs)
            for node in gm.graph.nodes:
                if node.op == "placeholder":
                    result = interpreter.placeholder(node.target, node.args, node.kwargs)
                    self.assertIsNotNone(result)

    def test_propagate_unbacked_symints_output(self):
        """Test PropagateUnbackedSymInts.output with NPU tensor."""
        gm, fake_inputs, fake_mode = self._create_exported_model()
        with fake_mode:
            interpreter = PropagateUnbackedSymInts(gm)
            interpreter.run(*fake_inputs)
            for node in gm.graph.nodes:
                if node.op == "output":
                    result = interpreter.output(node.target, node.args, node.kwargs)
                    self.assertIsNotNone(result)

    def test_rebind_unbacked(self):
        """Test rebind_unbacked with NPU tensor via shape_prop."""
        gm, fake_inputs, fake_mode = self._create_exported_model()
        shape_prop_gm = torch.fx.passes.shape_prop.ShapeProp(
            gm=gm, fake_mode=fake_mode
        )
        shape_prop_gm.propagate(*fake_inputs)
        self.assertEqual(len(fake_mode.shape_env.pending_fresh_unbacked_symbols), 0)


if __name__ == "__main__":
    run_tests()
