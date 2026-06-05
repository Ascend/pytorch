"""
Add validation cases for torch.utils.checkpoint APIs on NPU:
1. PyTorch community tests cover these APIs mainly through checkpoint call chains, so this file adds direct API validations.
2. This file validates torch.utils.checkpoint.SelectiveCheckpointContext and torch.utils.checkpoint.detach_variable (extendable).
"""

import functools
import inspect

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils import checkpoint as checkpoint_utils


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestUtilsCheckpointAPIs(TestCase):

    def test_detach_variable_keeps_device_and_requires_grad(self):
        x = torch.randn(2, 3, device=device_type, requires_grad=True)
        y = (x * 2).relu()
        marker = object()

        detached_x, detached_y, detached_marker = checkpoint_utils.detach_variable(
            (x, y, marker)
        )

        self.assertEqual(detached_x.device.type, device_type)
        self.assertEqual(detached_y.device.type, device_type)
        self.assertTrue(detached_x.requires_grad)
        self.assertTrue(detached_y.requires_grad)
        self.assertTrue(detached_x.is_leaf)
        self.assertTrue(detached_y.is_leaf)
        self.assertIsNone(detached_x.grad_fn)
        self.assertIsNone(detached_y.grad_fn)
        self.assertEqual(detached_x, x)
        self.assertEqual(detached_y, y)
        self.assertIs(detached_marker, marker)

    def test_detach_variable_rejects_non_tuple_input(self):
        x = torch.randn(2, device=device_type)

        with self.assertRaisesRegex(RuntimeError, "Only tuple"):
            checkpoint_utils.detach_variable([x])

    def test_selective_checkpoint_context_direct_attributes(self):
        ctx = checkpoint_utils.SelectiveCheckpointContext(is_recompute=False)

        self.assertIsInstance(ctx, checkpoint_utils.SelectiveCheckpointContext)
        self.assertFalse(ctx.is_recompute)

        signature = inspect.signature(checkpoint_utils.SelectiveCheckpointContext)
        if "op_output" in signature.parameters:
            output = torch.ones(2, device=device_type)
            ctx = checkpoint_utils.SelectiveCheckpointContext(
                is_recompute=False,
                op_output=output,
            )
            self.assertIs(ctx.op_output, output)
            self.assertEqual(ctx.op_output.device.type, device_type)

    def test_selective_checkpoint_context_passed_to_policy_fn(self):
        contexts = []

        def policy_fn(ctx, op, *args, **kwargs):
            contexts.append(ctx)
            return checkpoint_utils.CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            return x.sin().cos().sum()

        x = torch.randn(4, device=device_type, requires_grad=True)
        context_fn = functools.partial(
            checkpoint_utils.create_selective_checkpoint_contexts,
            policy_fn,
        )
        out = checkpoint_utils.checkpoint(
            fn,
            x,
            use_reentrant=False,
            context_fn=context_fn,
        )
        out.backward()

        self.assertTrue(contexts)
        self.assertTrue(
            all(
                isinstance(ctx, checkpoint_utils.SelectiveCheckpointContext)
                for ctx in contexts
            )
        )
        self.assertTrue(any(ctx.is_recompute for ctx in contexts))
        self.assertTrue(any(not ctx.is_recompute for ctx in contexts))

        forward_contexts = [ctx for ctx in contexts if not ctx.is_recompute]
        if forward_contexts and hasattr(forward_contexts[0], "op_output"):
            tensor_outputs = [
                ctx.op_output
                for ctx in forward_contexts
                if isinstance(ctx.op_output, torch.Tensor)
            ]
            self.assertTrue(tensor_outputs)
            self.assertTrue(
                all(output.device.type == device_type for output in tensor_outputs)
            )


if __name__ == "__main__":
    run_tests()
