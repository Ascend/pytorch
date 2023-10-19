import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNmsV4(TestCase):
    def generate_data(self, min1, max1, shape, dtype):
        input1 = np.random.uniform(min1, max1, shape).astype(dtype)
        npu_input = torch.from_numpy(input1)
        return npu_input

    def npu_op_exec(self, boxes, scores, max_output_size, iou_threshold, scores_threshold):
        boxes = boxes.to("npu")
        scores = scores.to("npu")
        iou_threshold = iou_threshold.to("npu")
        scores_threshold = scores_threshold.to("npu")
        npu_output = torch_npu.npu_nms_v4(boxes, scores, max_output_size, iou_threshold, scores_threshold)
        return npu_output

    def test_nms_v4_float32(self, device="npu"):
        boxes = self.generate_data(0, 100, (100, 4), np.float32)
        scores = self.generate_data(0, 1, (100), np.float32)
        max_output_size = 20
        iou_threshold = torch.tensor(0.5)
        scores_threshold = torch.tensor(0.3)

        npu_output = self.npu_op_exec(boxes, scores, max_output_size, iou_threshold, scores_threshold)


if __name__ == "__main__":
    run_tests()
