import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestApplyAdam(TestCase):
    def test_apply_adam(self):
        var1 = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        m1 = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        v1 = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        grad1 = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        var2 = var1.to(torch.half)
        m2 = m1.to(torch.half)
        v2 = v1.to(torch.half)
        grad2 = grad1.to(torch.half)
        res1, _, v1_o = torch_npu.npu_apply_adam(1, 1, 0.2, 0.2, 0.2, 0.2, grad1, False, False, out=(var1, m1, v1))
        res2, _, v2_o = torch_npu.npu_apply_adam(1, 1, 0.2, 0.2, 0.2, 0.2, grad2, False, False, out=(var2, m2, v2))
        expect_vo = torch.tensor([[[[2.2156, -0.1393],
                                    [0.6441, 0.3087]],
                                   [[0.9008, -0.0295],
                                    [0.0776, 0.0773]]],
                                  [[[0.1105, 1.0725],
                                    [0.8731, 0.4582]],
                                   [[0.1653, 0.3091],
                                    [0.3175, 0.0998]]]], dtype=torch.float32)
        self.assertRtolEqual(expect_vo, v1_o.cpu())
        self.assertRtolEqual(expect_vo.to(torch.half), v2_o.cpu())

    def test_apply_adam_out_fp32(self):
        var = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        m = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        v = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        grad = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        bt1p = 0.9
        bt2p = 0.9
        lr = 0.2
        bt1 = 0.2
        bt2 = 0.2
        ep = 0.2
        ul = False
        un = False
        var_o, m_o, v_o = torch_npu.npu_apply_adam(bt1p, bt2p, lr, bt1, bt2, ep, grad, ul, un, out=(var, m, v))
        expect_varo = torch.tensor([[[[-0.1842, 0.6028],
                                      [0.4803, -0.3156]],
                                     [[0.9466, -0.9984],
                                      [-0.1592, -0.1908]]],
                                    [[[-0.9448, 0.6290],
                                      [0.0694, 0.3411]],
                                     [[-0.0987, 0.5370],
                                      [-0.5744, 0.3317]]]])
        expect_mo = torch.tensor([[[[-1.4744, 0.1481],
                                    [-0.6954, 0.1557]],
                                   [[-0.6090, 0.4566],
                                    [-0.4863, 0.7218]]],
                                  [[[0.5437, -1.1527],
                                    [0.6547, -0.5491]],
                                   [[-0.2247, -0.7165],
                                    [0.7963, -0.1283]]]])
        expect_vo = torch.tensor([[[[2.2156, -0.1393],
                                    [0.6441, 0.3087]],
                                   [[0.9008, -0.0295],
                                    [0.0776, 0.0773]]],
                                  [[[0.1105, 1.0725],
                                    [0.8731, 0.4582]],
                                   [[0.1653, 0.3091],
                                    [0.3175, 0.0998]]]])
        self.assertRtolEqual(expect_varo, var_o.cpu())
        self.assertRtolEqual(expect_mo, m_o.cpu())
        self.assertRtolEqual(expect_vo, v_o.cpu())


if __name__ == "__main__":
    run_tests()
