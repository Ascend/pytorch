from collections import defaultdict
import torch
from torch_npu.utils import npu_combine_tensors
from torch_npu.utils._error_code import ErrCode, pta_error
from torch_npu.optim.npu_fused_optim_base import NpuFusedOptimizerBase
import torch_npu.optim.npu_fused_rmsprop_tf as npu_fused_rmsprop_tf
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuFusedRmsPropTf(TestCase):
    def test_init_param_state_with_momentum_centered(self):
        param = torch.tensor([1.0, 2.0])
        optimizer = npu_fused_rmsprop_tf.NpuFusedRMSpropTF([param], momentum=0.9, centered=True)
        optimizer._init_param_state(param, momentum=0.9, centered=True)
        state = optimizer.state[param]
        self.assertIn('step', state)
        self.assertIn('square_avg', state)
        self.assertIn('momentum_buffer', state)
        self.assertIn('grad_avg', state)
        self.assertEqual(state['step'], 0)
        self.assertTrue(torch.allclose(state['square_avg'], torch.ones_like(param)))

    def test_maybe_init_combined_states(self):
        param = torch.tensor([1.0, 2.0])
        optimizer = npu_fused_rmsprop_tf.NpuFusedRMSpropTF([param])
        optimizer.params_lists_indexed_by_group = [[param]]
        optimizer._maybe_init_combined_states()
        self.assertTrue(optimizer.is_states_combined)
        self.assertIsNotNone(optimizer.combined_param_states_indexed_by_group[0])

    def test_maybe_init_combined_states_already_combined(self):
        param = torch.tensor([1.0, 2.0])
        optimizer = npu_fused_rmsprop_tf.NpuFusedRMSpropTF([param])
        optimizer.is_states_combined = True
        optimizer._maybe_init_combined_states()
        self.assertTrue(optimizer.is_states_combined)

    def test_setstate_backward_compatibility(self):
        param = torch.tensor([1.0, 2, 0])
        optimizer = npu_fused_rmsprop_tf.NpuFusedRMSpropTF([param])

        old_state = {
            'param_groups': [
                {
                    'params': [param],
                    'lr': 1e-2,
                    'weight_decay': 0,
                    'momentum': 0,
                    'centered': False
                }
            ]
        }

        optimizer.__setstate__(old_state)

        group = optimizer.param_groups[0]
        self.assertEqual(group['momentum'], 0)
        self.assertEqual(group['centered'], False)

    def test_init_param_state(self):
        optimizer = npu_fused_rmsprop_tf.NpuFusedRMSpropTF([torch.tensor([1.0])], momentum=0.9, centered=True)
        p = torch.tensor([1.0])
        optimizer._init_param_state(p, momentum=0.9, centered=True)
        state = optimizer.state[p]
        self.assertIn('step', state)
        self.assertIn('square_avg', state)
        self.assertIn('momentum_buffer', state)
        self.assertIn('grad_avg', state)

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            npu_fused_rmsprop_tf.NpuFusedRMSpropTF([torch.tensor([1.0])], alpha=-0.9)

    def test_invalid_weight_decay(self):
        with self.assertRaises(ValueError):
            npu_fused_rmsprop_tf.NpuFusedRMSpropTF([torch.tensor([1.0])], weight_decay=-1e-2)

    def test_invalid_momentum(self):
        with self.assertRaises(ValueError):
            npu_fused_rmsprop_tf.NpuFusedRMSpropTF([torch.tensor([1.0])], momentum=-0.1)

    def test_invalid_epsilon(self):
        with self.assertRaises(ValueError):
            npu_fused_rmsprop_tf.NpuFusedRMSpropTF([torch.tensor([1.0])], eps=-1e-10)

    def test_invalid_learning_rate(self):
        with self.assertRaises(ValueError):
            npu_fused_rmsprop_tf.NpuFusedRMSpropTF([torch.tensor([1.0])], lr=-1e-2)


if __name__ == "__main__":
    run_tests()