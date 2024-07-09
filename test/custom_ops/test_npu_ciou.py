import math
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestCiou(TestCase):
    def ciou(self, prediction, gtbox):
        """input box format: xyhw"""
        eps = 1e-10
        # xyhw --> xyxy
        p_x1, p_x2 = prediction[0] - prediction[2] / 2, \
            prediction[0] + prediction[2] / 2
        p_y1, p_y2 = prediction[1] - prediction[3] / 2, \
            prediction[1] + prediction[3] / 2
        g_x1, g_x2 = gtbox[0] - gtbox[2] / 2, gtbox[0] + gtbox[2] / 2
        g_y1, g_y2 = gtbox[1] - gtbox[3] / 2, gtbox[1] + gtbox[3] / 2

        # intersection area
        inter = (torch.min(p_x2, g_x2) - torch.max(p_x1, g_x1)).clamp(0) * \
                (torch.min(p_y2, g_y2) - torch.max(p_y1, g_y1)).clamp(0)
        # union area
        w1, h1 = p_x2 - p_x1, p_y2 - p_y1
        w2, h2 = g_x2 - g_x1, g_y2 - g_y1
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union

        cw = torch.max(p_x2, g_x2) - torch.min(p_x1, g_x1)
        ch = torch.max(p_y2, g_y2) - torch.min(p_y1, g_y1)
        # calculate distance
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        rho2 = ((prediction[0] - gtbox[0]) ** 2 +
                (prediction[1] - gtbox[1]) ** 2)  # center distance squared

        v = (4 / math.pi ** 2) * \
            torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        alpha = v / ((1 + eps) - iou + v)

        return iou - (rho2 / c2 + v * alpha)

    def gen_data(self, n):
        coordinate = torch.rand(
            4, n, dtype=torch.float32).uniform_(0, 10).npu()
        return coordinate

    def cpu_to_exec(self, boxes1, boxes2):
        n = boxes1.shape[1]
        boxes1 = torch.transpose(boxes1, 0, 1)
        boxes2 = torch.transpose(boxes2, 0, 1)
        cious = [self.ciou(boxes1[i], boxes2[i]) for i in range(n)]
        out = torch.tensor(cious, dtype=torch.float32).reshape(1, -1)
        return out.cpu().numpy()

    def npu_to_exec(self, boxes1, boxes2):
        out = torch_npu.npu_ciou(
            boxes1, boxes2, trans=True, is_cross=False, mode=0)
        return out.cpu().numpy()

    def test_ciou(self):
        gtbox = self.gen_data(32)
        prediction = self.gen_data(32)
        cpu_out = self.cpu_to_exec(prediction, gtbox)
        npu_out = self.npu_to_exec(prediction, gtbox)
        self.assertRtolEqual(cpu_out, npu_out)


if __name__ == "__main__":
    run_tests()
