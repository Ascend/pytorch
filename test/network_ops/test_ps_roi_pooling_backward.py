import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestPsRoiPoolingBackward(TestCase):
    def test_ps_roi_pooling_backward_fp16(self):
        roi = torch.tensor([[[1], [2], [3], [4], [5]],
                            [[6], [7], [8], [9], [10]]
                            ], dtype=torch.float16).npu()
        _input = torch.tensor([[[[1]], [[2]], [[3]], [[4]],
                                [[5]], [[6]], [[7]], [[8]]],
                               [[[9]], [[10]], [[11]], [[12]],
                                [[13]], [[14]], [[15]], [[16]]]
                               ], dtype=torch.float16).npu()
        _input.requires_grad = True
        out = torch_npu.npu_ps_roi_pooling(_input, roi, 0.5, 2, 2)
        out.backward(torch.ones_like(out))
        gradout = _input.grad
        expect_gradout = torch.tensor([[[[0.]], [[0.]], [[0.]], [[0.]],
                                        [[0.]], [[0.]], [[0.]], [[0.]]],
                                       [[[0.]], [[0.]], [[0.]], [[0.]],
                                        [[0.]], [[0.]], [[0.]], [[0.]]]
                                       ], dtype=torch.float16)
        expect_out = torch.tensor([[[[0., 0.], [0., 0.]],
                                    [[0., 0.], [0., 0.]]],
                                   [[[0., 0.], [0., 0.]],
                                    [[0., 0.], [0., 0.]]]
                                   ], dtype=torch.float16)

        self.assertRtolEqual(expect_out, out.detach().cpu())
        self.assertRtolEqual(expect_gradout, gradout.cpu())


if __name__ == "__main__":
    run_tests()
