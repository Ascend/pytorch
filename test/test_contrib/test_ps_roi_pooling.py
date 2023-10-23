import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.contrib.module import PSROIPool


class TestPsRoiPooling(TestCase):
    def get_random_rois(self, shape):
        rois_init = torch.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                pi1 = torch.rand(1, 2).uniform_(0, 10)
                pi2 = torch.rand(1, 2).uniform_(10, 100)
                boxi = torch.cat((pi1, pi2), 1)
                n = torch.tensor([[float(i)]])
                boxi = torch.cat((n, boxi), 1)
                rois_init[i, j, :] = boxi
        return rois_init

    def npu_ps_roi_align(self, cls_feat, rois_tensor, pooled_height,
                         pooled_width, spatial_scale, group_size, output_dim):
        cls_feat.requires_grad = True
        model = PSROIPool(pooled_height, pooled_width, spatial_scale, group_size, output_dim)
        output = model(cls_feat, rois_tensor)  # 512,22,7,7
        output.sum().backward()
        return output.detach().cpu(), cls_feat.grad.cpu()

    def test_npu_roi_align_1(self):
        cls_feat = torch.randn(4, 1078, 84, 84).float().npu()
        rois_tensor = self.get_random_rois((4, 128, 5)).permute(0, 2, 1).float().npu()
        pooled_height = 7
        pooled_width = 7
        spatial_scale = 1 / 16.0
        group_size = 7
        output_dim = 22

        npu_output, npu_inputgrad = self.npu_ps_roi_align(cls_feat, rois_tensor, pooled_height,
                                                          pooled_width, spatial_scale, group_size, output_dim)

        expedt_cpu_output_shape = torch.randn(512, 22, 7, 7).shape
        expedt_cpu_inputgrad_shape = cls_feat.shape

        self.assertEqual(expedt_cpu_output_shape, npu_output.shape)
        self.assertEqual(expedt_cpu_inputgrad_shape, npu_inputgrad.shape)


if __name__ == "__main__":
    run_tests()
