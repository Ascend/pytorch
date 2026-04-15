import unittest
import os
import torch
import torch_npu
from torch_npu.testing.common_utils import SkipIfNotGteCANNVersion
from torch_npu.testing.testcase import TestCase, run_tests


class TestAclgraphSuperKernelScope(TestCase):

    @SkipIfNotGteCANNVersion("9.9.0")
    def test_super_kernel_scope_begin_end_basic(self):
        """Test basic usage of super_kernel_scope_begin and super_kernel_scope_end"""
        torch.npu.set_device(0)

        g = torch.npu.NPUGraph()

        x = torch.randn(10, 10, device='npu')
        y = torch.randn(10, 10, device='npu')

        with torch.npu.graph(g):
            torch_npu.npu.super_kernel_scope_begin("test_scope")
            z = torch.matmul(x, y)
            torch_npu.npu.super_kernel_scope_end("test_scope")
        g.replay()

        expected = torch.matmul(x, y)
        self.assertTrue(torch.allclose(z, expected, rtol=1e-3, atol=1e-3))

    @SkipIfNotGteCANNVersion("9.9.0")
    def test_super_kernel_scope_begin_end_without_name(self):
        """Test super_kernel_scope_begin and super_kernel_scope_end without scope name"""
        torch.npu.set_device(0)

        g = torch.npu.NPUGraph()

        x = torch.randn(5, 5, device='npu')
        y = torch.randn(5, 5, device='npu')

        with torch.npu.graph(g):
            torch_npu.npu.super_kernel_scope_begin()
            z = x + y
            torch_npu.npu.super_kernel_scope_end()
        g.replay()

        expected = x + y
        self.assertTrue(torch.allclose(z, expected, rtol=1e-3, atol=1e-3))

    @SkipIfNotGteCANNVersion("9.9.0")
    def test_super_kernel_scope_begin_end_nested(self):
        """Test nested usage of super_kernel_scope_begin and super_kernel_scope_end"""
        torch.npu.set_device(0)

        g = torch.npu.NPUGraph()

        x = torch.randn(8, 8, device='npu')
        y = torch.randn(8, 8, device='npu')

        with torch.npu.graph(g):
            torch_npu.npu.super_kernel_scope_begin("outer_scope")
            z1 = torch.matmul(x, y)

            torch_npu.npu.super_kernel_scope_begin("inner_scope")
            z2 = z1 * 2.0
            torch_npu.npu.super_kernel_scope_end("inner_scope")

            z3 = z2 + 1.0
            torch_npu.npu.super_kernel_scope_end("outer_scope")
        g.replay()

        expected = torch.matmul(x, y) * 2.0 + 1.0
        self.assertTrue(torch.allclose(z3, expected, rtol=1e-3, atol=1e-3))

    def test_super_kernel_scope_begin_end_invalid_name(self):
        """Test that empty string scope name raises error"""
        torch.npu.set_device(0)

        with self.assertRaises(RuntimeError):
            torch_npu.npu.super_kernel_scope_begin("")

        with self.assertRaises(RuntimeError):
            torch_npu.npu.super_kernel_scope_begin("   ")


class TestAclgraphSuperKernelOptimize(TestCase):

    @SkipIfNotGteCANNVersion("9.9.0")
    def test_super_kernel_optimize_basic(self):
        """Test basic super_kernel_optimize functionality"""
        torch.npu.set_device(0)

        g = torch.npu.NPUGraph()

        x = torch.randn(16, 16, device='npu')
        y = torch.randn(16, 16, device='npu')

        with torch.npu.graph(g):
            z = torch.matmul(x, y)

        # Test with no optimize options (None)
        g.super_kernel_optimize(optimize_options=None, debug_options=None)
        g.replay()

        expected = torch.matmul(x, y)
        self.assertTrue(torch.allclose(z, expected, rtol=1e-3, atol=1e-3))

    def test_super_kernel_optimize_invalid_options_type(self):
        """Test that invalid options type raises error"""
        torch.npu.set_device(0)

        g = torch.npu.NPUGraph()

        with self.assertRaises(RuntimeError):
            g.super_kernel_optimize(optimize_options="invalid", debug_options=None)

        with self.assertRaises(RuntimeError):
            g.super_kernel_optimize(optimize_options=None, debug_options="invalid")

    def test_super_kernel_optimize_invalid_option_key(self):
        """Test that invalid option key raises error"""
        torch.npu.set_device(0)

        g = torch.npu.NPUGraph()

        with self.assertRaises(RuntimeError):
            g.super_kernel_optimize(
                optimize_options={'invalid_key': 1},
                debug_options=None
            )

        with self.assertRaises(RuntimeError):
            g.super_kernel_optimize(
                optimize_options=None,
                debug_options={'invalid_key': 1}
            )

    def test_super_kernel_optimize_invalid_option_value_type(self):
        """Test that invalid option value type raises error"""
        torch.npu.set_device(0)

        g = torch.npu.NPUGraph()

        with self.assertRaises(RuntimeError):
            g.super_kernel_optimize(
                optimize_options={'preload_code': 'invalid_string'},
                debug_options=None
            )

        with self.assertRaises(RuntimeError):
            g.super_kernel_optimize(
                optimize_options=None,
                debug_options={'debug_sync_all': 'invalid_string'}
            )

        # constant_codegen expects int, not str
        with self.assertRaises(RuntimeError):
            g.super_kernel_optimize(
                optimize_options={'constant_codegen': 'invalid'},
                debug_options=None
            )

        # auto_op_parallel expects int, not str
        with self.assertRaises(RuntimeError):
            g.super_kernel_optimize(
                optimize_options={'auto_op_parallel': 'invalid'},
                debug_options=None
            )

        # debug_op_exec_trace expects int, not str
        with self.assertRaises(RuntimeError):
            g.super_kernel_optimize(
                optimize_options=None,
                debug_options={'debug_op_exec_trace': 'invalid'}
            )

        # debug_cross_core_sync_check expects int, not str
        with self.assertRaises(RuntimeError):
            g.super_kernel_optimize(
                optimize_options=None,
                debug_options={'debug_cross_core_sync_check': 'invalid'}
            )

        # opt_extend expects str, not int
        with self.assertRaises(RuntimeError):
            g.super_kernel_optimize(
                optimize_options={'opt_extend': 123},
                debug_options=None
            )

        # debug_extend expects str, not int
        with self.assertRaises(RuntimeError):
            g.super_kernel_optimize(
                optimize_options=None,
                debug_options={'debug_extend': 123}
            )

        # debug_dcci_before_kernel_start expects list, not int
        with self.assertRaises(RuntimeError):
            g.super_kernel_optimize(
                optimize_options=None,
                debug_options={'debug_dcci_before_kernel_start': 1}
            )


class TestAclgraphSuperKernelIntegration(TestCase):

    @SkipIfNotGteCANNVersion("9.9.0")
    def test_super_kernel_scope_and_optimize_integration(self):
        """Test integration of super_kernel_scope and super_kernel_optimize"""
        torch.npu.set_device(0)

        g = torch.npu.NPUGraph()

        x = torch.randn(16, 16, device='npu')
        y = torch.randn(16, 16, device='npu')

        with torch.npu.graph(g):
            torch_npu.npu.super_kernel_scope_begin("compute_scope")
            z = torch.matmul(x, y)
            w = torch.relu(z)
            torch_npu.npu.super_kernel_scope_end("compute_scope")

        optimize_options = {
            'preload_code': 1,
            'split_mode': 1,
            'stream_fusion': 1,
            'constant_codegen': 1,
            'auto_op_parallel': 1,
            'opt_extend': 'test_opt_value'
        }
        debug_options = {
            'debug_sync_all': 1,
            'debug_dcci_disable_on_kernel': ['kernel_a', 'kernel_b'],
            'debug_dcci_before_kernel_start': ['kernel_c'],
            'debug_op_exec_trace': 1,
            'debug_cross_core_sync_check': 1,
            'debug_extend': 'test_debug_value'
        }

        g.super_kernel_optimize(optimize_options=optimize_options, debug_options=debug_options)
        g.replay()

        expected_z = torch.matmul(x, y)
        expected_w = torch.relu(expected_z)
        self.assertTrue(torch.allclose(w, expected_w, rtol=1e-3, atol=1e-3))

    @SkipIfNotGteCANNVersion("9.9.0")
    def test_super_kernel_with_nn_module(self):
        """Test super_kernel features with nn.Module"""
        torch.npu.set_device(0)

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 32).npu()

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        model.eval()

        g = torch.npu.NPUGraph()
        x = torch.randn(4, 32, device='npu')

        with torch.npu.graph(g):
            torch_npu.npu.super_kernel_scope_begin("model_forward")
            out = model(x)
            torch_npu.npu.super_kernel_scope_end("model_forward")

        optimize_options = {
            'stream_fusion': 1
        }

        g.super_kernel_optimize(optimize_options=optimize_options, debug_options=None)
        g.replay()

        expected = model(x)
        self.assertTrue(torch.allclose(out, expected, rtol=1e-3, atol=1e-3))


if __name__ == '__main__':
    run_tests()