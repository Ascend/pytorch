import itertools
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.contrib.function import npu_iou, npu_giou, npu_diou, npu_ciou


class TestIou(TestCase):
    def test_npu_iou_diff_shape_input(self):
        box1 = torch.FloatTensor([[10, 55, 85, 160]])
        box2 = torch.FloatTensor([[18, 45, 80, 130], [38, 85, 70, 230]])
        box1 = box1.float().npu()
        box2 = box2.float().npu()
        iou1 = npu_iou(box1, box2, mode="iou")
        iou2 = npu_iou(box1, box2)
        expedt_iou1 = torch.tensor([[0.5474, 0.2373]], dtype=torch.float32)
        expedt_iou2 = torch.tensor([[0.5474, 0.2373]], dtype=torch.float32)
        self.assertRtolEqual(expedt_iou1, iou1.cpu())
        self.assertRtolEqual(expedt_iou2, iou2.cpu())

    def test_npu_iou_same_shape_input(self):
        box1 = torch.tensor([[0, 0, 10, 10],
                            [10, 10, 20, 20],
                            [32, 32, 38, 42]], dtype=torch.float32).to("npu")
        box2 = torch.tensor([[0, 0, 10, 20],
                            [0, 10, 10, 10],
                            [10, 10, 20, 20]], dtype=torch.float32).to("npu")
        expedt_iou = torch.tensor([[0.4985, 0.0000, 0.0000],
                                   [0.0000, 0.0000, 0.9961],
                                   [0.0000, 0.0000, 0.0000]], dtype=torch.float32)
        iou1 = npu_iou(box1, box2, mode="iou")
        iou2 = npu_iou(box1, box2, mode='ptiou')
        self.assertRtolEqual(expedt_iou, iou1.cpu())
        self.assertRtolEqual(expedt_iou, iou2.cpu())

    def test_npu_iou_multiparameters(self):
        box1 = torch.FloatTensor([[10, 55, 85, 160]])
        box2 = torch.FloatTensor([[18, 45, 80, 130], [38, 85, 70, 230]])
        box1 = box1.float().npu() / 100.
        box2 = box2.float().npu() / 100.
        iou1 = npu_iou(box1, box2, mode="iou", is_normalized=True, normalized_scale=100.)
        iou2 = npu_iou(box1, box2, is_normalized=True, normalized_scale=100.)
        expedt_iou1 = torch.tensor([[0.5474, 0.2373]], dtype=torch.float32)
        expedt_iou2 = torch.tensor([[0.5474, 0.2373]], dtype=torch.float32)
        self.assertRtolEqual(expedt_iou1, iou1.cpu())
        self.assertRtolEqual(expedt_iou2, iou2.cpu())

    def test_npu_giou(self):
        box1 = torch.tensor([[0, 0, 10, 10],
                            [10, 10, 20, 20],
                            [32, 32, 38, 42]], dtype=torch.float32).to("npu")
        box1.requires_grad = True
        box2 = torch.tensor([[0, 0, 10, 20],
                            [0, 10, 10, 10],
                            [10, 10, 20, 20]], dtype=torch.float32).to("npu")
        iou = npu_giou(box1, box2)
        expedt_iou = torch.tensor([[0.5000],
                                   [0.0111],
                                   [-0.2523]], dtype=torch.float32)
        self.assertRtolEqual(expedt_iou, iou.cpu().detach())

    def test_npu_diou(self):
        def generate_diou_data(n, m, dtype):
            data_bboxes = np.array([]).astype(dtype)
            for i in range(4):
                data_bboxes_array = i // 2 + math.pow(-1, i // 2) * 0.5 * np.random.rand(1, n).astype(dtype)
                data_bboxes = np.append(data_bboxes, data_bboxes_array)
            data_bboxes = data_bboxes.reshape([4, n])
            data_gtboxes = np.array([]).astype(dtype)
            for i in range(4):
                data_gtboxes_array = i // 2 + math.pow(-1, i // 2) * 0.5 * np.random.rand(1, m).astype(dtype)
                data_gtboxes = np.append(data_gtboxes, data_gtboxes_array)
            data_gtboxes = data_gtboxes.reshape([4, m])
            cpu_input1 = torch.from_numpy(data_bboxes)
            cpu_input2 = torch.from_numpy(data_gtboxes)
            npu_input1 = cpu_input1.npu()
            npu_input2 = cpu_input2.npu()
            list1 = [cpu_input1, cpu_input2, npu_input1, npu_input2]
            return list1

        def cpu_op_exec(box1, box2, trans=False, is_cross=False, mode="iou", eps=1e-9):
            box3 = box1.numpy()
            dtype = box3.dtype
            _, n = box1.shape
            _, m = box2.shape
            if trans:
                b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
                b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
                b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
                b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
            else:
                b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
                b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

            diou_res = np.array([], dtype=dtype)
            iter_list = itertools.product(list(range(n)), list(range(m)))
            for i, j in iter_list:
                cw = torch.max(b1_x2[i], b2_x2[j]) - torch.min(b1_x1[i], b2_x1[j])
                ch = torch.max(b1_y2[i], b2_y2[j]) - torch.min(b1_y1[i], b2_y1[j])
                c2 = cw ** 2 + ch ** 2 + eps
                rho2 = ((b2_x1[i] + b2_x2[j] - b1_x1[i] - b1_x2[j]) ** 2 +
                        (b2_y1[i] + b2_y2[j] - b1_y1[i] - b1_y2[j]) ** 2) / 4
                inter_area = (torch.min(b1_x2[i], b2_x2[j]) - torch.max(b1_x1[i], b2_x1[j])).clamp(0) * \
                    (torch.min(b1_y2[i], b2_y2[j]) - torch.max(b1_y1[i], b2_y1[j])).clamp(0)
                w1, h1 = b1_x2[i] - b1_x1[i], b1_y2[i] - b1_y1[i] + eps
                w2, h2 = b2_x2[j] - b2_x1[j], b2_y2[j] - b2_y1[j] + eps
                union_area = w1 * h1 + w2 * h2 - inter_area + eps
                diou_ij = inter_area / union_area - (rho2 / c2)
                if not is_cross:
                    if i == j:
                        diou_res = np.append(diou_res, diou_ij)
                else:
                    diou_res = np.append(diou_res, diou_ij)

            if not is_cross:
                res = diou_res.reshape(1, n)
            else:
                res = diou_res.reshape(n, m)
            return res

        def test_npu_diou_shape_format_fp32():
            _test_npu_diou_shape_format(np.float32)

        def test_npu_diou_shape_format_fp16():
            _test_npu_diou_shape_format(np.float16)

        def _test_npu_diou_shape_format(dtype):
            shape_list = [
                [10, 10],
                [12, 12],
                [100, 100]
            ]
            is_trans_list = [True]
            mode_list = ["iou"]
            shape_format = [[j, k, m]
                            for j in shape_list
                            for k in is_trans_list
                            for m in mode_list]

            for item in shape_format:
                mode_digit = 0 if item[-1] == "iou" else 1
                is_cross = False if item[0][0] == item[0][1] else True
                if dtype == np.float16:
                    dtype = np.float32
                list1 = self.generate_diou_data(*item[0], dtype)
                cpu_output = cpu_op_exec(list1[0], list1[1], item[1], is_cross, item[-1])
                npu_output = npu_diou(list1[2], list1[3], item[1], is_cross, mode_digit)
                cpu_output = cpu_output.astype(npu_output.dtype)

                if dtype == np.float16:
                    self.assertRtolEqual(cpu_output, npu_output, prec16=1e-2)
                else:
                    self.assertRtolEqual(cpu_output, npu_output)

    def test_npu_ciou(self):
        box1 = torch.tensor([[0, 0, 10, 10],
                            [10, 10, 20, 20],
                            [32, 32, 38, 42],
                            [8, 8, 4, 4]], dtype=torch.float32).to("npu")
        box1.requires_grad = True
        box2 = torch.tensor([[0, 0, 10, 20],
                            [0, 10, 10, 10],
                            [10, 10, 20, 20],
                            [8, 8, 4, 4]], dtype=torch.float32).to("npu")
        expedt_ciou = torch.tensor([[-0.0794, 0.3052, -0.0610, -0.1021]], dtype=torch.float32)
        box2.requires_grad = True
        ciou = npu_ciou(box1, box2)
        self.assertRtolEqual(expedt_ciou, ciou.cpu().detach())


if __name__ == "__main__":
    run_tests()
