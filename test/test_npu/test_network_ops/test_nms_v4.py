import torch
import numpy as np
import sys
import copy
import torch.nn as nn
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestNmsV4(TestCase):
    def generate_data(self, min, max, shape, dtype):
        input = np.random.uniform(min, max, shape).astype(dtype)
        npu_input = torch.from_numpy(input)
        return npu_input

    def npu_op_exec(self, boxes, scores, max_output_size, iou_threshold, scores_threshold):
        boxes            = boxes.to("npu")
        scores           = scores.to("npu")
        iou_threshold    = iou_threshold.to("npu")
        scores_threshold = scores_threshold.to("npu")
        npu_output = torch.npu_nms_v4(boxes, scores, max_output_size, iou_threshold, scores_threshold)
        #npu_output = npu_output.to("cpu")
        print("===npu_output===")
        print(npu_output)
        return npu_output


    def test_nms_v4_float32(self, device):
        boxes = self.generate_data(0, 100, (100, 4), np.float32)
        scores = self.generate_data(0, 1, (100), np.float32)
        max_output_size = 20
        iou_threshold = torch.tensor(0.5)
        scores_threshold = torch.tensor(0.3)

        npu_output = self.npu_op_exec(boxes, scores, max_output_size, iou_threshold, scores_threshold)


instantiate_device_type_tests(TestNmsV4, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests() 