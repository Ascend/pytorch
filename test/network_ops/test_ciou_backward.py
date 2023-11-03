import math
import torch
import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuCiouBackward(TestCase):
    def generate_ciou_data(self, n, m, dtype):
        data_bboxes = 20 * np.random.rand(4, n).astype(dtype)
        data_gtboxes = 20 * np.random.rand(4, n).astype(dtype)
        cpu_input = torch.from_numpy(data_bboxes)
        cpu_input1 = torch.from_numpy(data_gtboxes)
        npu_input = cpu_input.npu()
        npu_input1 = cpu_input1.npu()
        list1 = [cpu_input, cpu_input1, npu_input, npu_input1]
        return list1

    def bbox_iou(self, box1, box2, x1y1x2y2=True, eps=1e-9):
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        inter1 = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter1 + eps

        iou = inter1 / union
        w = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        h = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

        c2 = w ** 2 + h ** 2 + eps
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        with torch.no_grad():
            alpha = v / ((1 + eps) - iou + v)
        return iou - (rho2 / c2 + v * alpha)

    def cpu_op_exec(self, bboxes, gtboxes, trans=False, is_cross=False, mode="iou"):
        bboxes.requires_grad = True
        gtboxes.requires_grad = True

        ciou = self.bbox_iou(bboxes, gtboxes, x1y1x2y2=False)
        ciou.backward(torch.ones_like(ciou))

        bboxes_grad = bboxes.grad
        gtboxes_grad = gtboxes.grad

        bboxes_grad = bboxes_grad.detach().numpy()
        gtboxes_grad = gtboxes_grad.detach().numpy()
        ciou = ciou.detach().numpy()
        ciou = np.expand_dims(ciou, 0)

        return ciou, bboxes_grad, gtboxes_grad

    def npu_op_exec(self, box1, box2, trans=False, is_cross=False, mode=0):
        box1.requires_grad = True
        box2.requires_grad = True
        overlap = torch_npu.npu_ciou(box1, box2, trans, is_cross, mode, True)

        overlap.backward(torch.ones_like(overlap))

        overlap = overlap.to("cpu")

        box1_grad = box1.grad
        box2_grad = box2.grad
        box1_grad = box1_grad.detach().cpu().numpy()
        box2_grad = box2_grad.detach().cpu().numpy()

        overlap = overlap.detach().numpy()
        return overlap, box1_grad, box2_grad

    def test_npu_ciou_backward_shape_format(self):
        shape_list1 = [
            [6, 6],
            [32, 32],
            [100, 100]
        ]
        is_trans_list1 = [True]
        mode_list = ["iou"]
        dtype = np.float32
        shape_format1 = [[j, k, m]
                         for j in shape_list1
                         for k in is_trans_list1
                         for m in mode_list]

        for item in shape_format1:
            mode_digit = 0 if item[-1] == "iou" else 1
            is_cross = False
            list1 = self.generate_ciou_data(*item[0], dtype)
            cpu_overlap, bboxes_grad, gtboxes_grad = self.cpu_op_exec(list1[0], list1[1], item[1], is_cross, item[-1])
            overlap, box1_grad, box2_grad = self.npu_op_exec(list1[2], list1[3], item[1], is_cross, mode_digit)

            self.assertRtolEqual(cpu_overlap, overlap)
            self.assertRtolEqual(bboxes_grad, box1_grad)
            self.assertRtolEqual(gtboxes_grad, box2_grad)


if __name__ == "__main__":
    run_tests()
