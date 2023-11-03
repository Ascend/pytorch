import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestRotatedIou(TestCase):
    def generate_rto_data(self, item):
        np.random.seed(1234)
        minValue, maxValue = 20, 60
        scope = 20
        dtype = item[0][0]
        shape_one = item[0][-1]
        shape_two = item[1][-1]
        trans = item[-1]

        boxes_array1 = np.random.uniform(minValue, maxValue, shape_one[:2] + [2]).astype(dtype)
        boxes_wh = np.random.randint(1, scope, size=shape_one[:2] + [2])
        boxes_angle = np.random.randint(-180, 180, size=shape_one[:2] + [1])
        boxes = np.concatenate([boxes_array1, boxes_wh, boxes_angle], dtype=dtype, axis=-1)
        # query_boxes
        query_boxes_array1 = np.random.uniform(minValue, maxValue, shape_two[:2] + [2]).astype(dtype)
        query_boxes_wh = np.random.randint(1, scope, size=shape_two[:2] + [2])
        query_boxes_angle = np.random.randint(-180, 180, size=shape_two[:2] + [1])
        query_boxes = np.concatenate([query_boxes_array1, query_boxes_wh, query_boxes_angle], dtype=dtype, axis=-1)

        cpu_input1 = torch.from_numpy(boxes)
        cpu_input2 = torch.from_numpy(query_boxes)
        npu_input1 = cpu_input1.npu()
        npu_input2 = cpu_input2.npu()
        list1 = [boxes, query_boxes, npu_input1, npu_input2]
        return list1

    def cpu_expect_result(self, dtype):
        if dtype == np.float32:
            output = np.array([[[0., 0.00045966, 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0.00600622, 0.10504241, 0.]],
                               [[0., 0., 0.], [0., 0., 0.]]], dtype=np.float32)
        else:
            output = np.array([[[0., 0.00045966, 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0.00600622, 0.10504241, 0.]],
                               [[0., 0., 0.], [0., 0., 0.]]], dtype=np.float16)
        return output

    def npu_op_exec(self, box1, box2, trans=False):
        output = torch_npu.npu_rotated_iou(box1, box2, trans, 0, True, 0.0, 0.0)
        output = output.detach().cpu().numpy()
        return output

    def test_rotated_iou_shape_format_fp32(self):
        dtype = np.float32
        shape_format = [[dtype, -1, [4, 2, 5]], [dtype, -1, [4, 3, 5]], False]
        list2 = self.generate_rto_data(shape_format)
        cpu_output = self.cpu_expect_result(dtype)
        npu_output = self.npu_op_exec(list2[2], list2[3], shape_format[-1])
        self.assertRtolEqual(cpu_output, npu_output)

    def test_rotated_iou_shape_format_fp16(self):
        dtype = np.float16
        shape_format = [[dtype, -1, [4, 2, 5]], [dtype, -1, [4, 3, 5]], False]
        list1 = self.generate_rto_data(shape_format)
        cpu_output = self.cpu_expect_result(dtype)
        npu_output = self.npu_op_exec(list1[2], list1[3], shape_format[-1])
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
