# Copyright (c) 2020-2023, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import functools
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

INDEX_X = 0
INDEX_Y = 1
INDEX_W = 2
INDEX_H = 3
INDEX_ANGLE = 4

INDEX_X1 = 0
INDEX_Y1 = 1
INDEX_X2 = 2
INDEX_Y2 = 3

POINT_LU = 0
POINT_RU = 1
POINT_LD = 2
POINT_RD = 3

POINT_NUM = 4
LENS_TWO = 2
TOTAL_INTER_POINTS = 24
EPS = 1e-14


class RectInfo:
    def __init__(self):
        self.cx = 0.
        self.cy = 0.
        self.w = 0.
        self.h = 0.
        self.angle = 0.
        self.size = 0.

    def set_info(self, rect):
        self.cx = rect[INDEX_X]
        self.cy = rect[INDEX_Y]
        self.w = rect[INDEX_W]
        self.h = rect[INDEX_H]
        self.angle = rect[INDEX_ANGLE]
        self.size = self.w * self.h

    def set_info_with_xy(self, rect):
        # input with x1y1x2y2+counter-clockwise
        self.cx = (rect[INDEX_X1] + rect[INDEX_X2]) * 0.5
        self.cy = (rect[INDEX_Y1] + rect[INDEX_Y2]) * 0.5
        self.w = rect[INDEX_X2] - rect[INDEX_X1]
        self.h = rect[INDEX_Y2] - rect[INDEX_Y1]
        self.angle = -rect[INDEX_ANGLE]
        self.size = self.w * self.h


class TestNpuNmsRotated(TestCase):
    def rect_to_points(self, rect):
        # theta is equal to M_PI / 180.
        theta = rect.angle * 0.01745329251
        b = float(math.cos(theta)) * 0.5
        a = float(math.sin(theta)) * 0.5

        pts = torch.zeros(POINT_NUM, LENS_TWO)
        pts[POINT_LU][INDEX_X] = rect.cx - a * rect.h - b * rect.w
        pts[POINT_LU][INDEX_Y] = rect.cy + b * rect.h - a * rect.w
        pts[POINT_RU][INDEX_X] = rect.cx + a * rect.h - b * rect.w
        pts[POINT_RU][INDEX_Y] = rect.cy - b * rect.h - a * rect.w
        pts[POINT_LD][INDEX_X] = 2. * rect.cx - pts[POINT_LU][INDEX_X]
        pts[POINT_LD][INDEX_Y] = 2. * rect.cy - pts[POINT_LU][INDEX_Y]
        pts[POINT_RD][INDEX_X] = 2. * rect.cx - pts[POINT_RU][INDEX_X]
        pts[POINT_RD][INDEX_Y] = 2. * rect.cy - pts[POINT_RU][INDEX_Y]
        return pts

    def cross2d(self, p1, p2):
        return p1[INDEX_X] * p2[INDEX_Y] - p1[INDEX_Y] * p2[INDEX_X]

    def vec_dot(self, p1, p2):
        return p1[INDEX_X] * p2[INDEX_X] + p1[INDEX_Y] * p2[INDEX_Y]

    def triangle_area(self, a, b, c):
        return (a[INDEX_X] - c[INDEX_X]) * (b[INDEX_Y] - c[INDEX_Y]) - (a[INDEX_Y] - c[INDEX_Y]) * (
            b[INDEX_X] - c[INDEX_X])

    def contour_area(self, inter_pts, num):
        area = 0.0
        for i in range(num - 1):
            area += math.fabs(self.triangle_area(inter_pts[0], inter_pts[i], inter_pts[i + 1]))
        return area * 0.5

    def is_same_rect(self, pts1, pts2):
        same_point_eps = 0.00001
        is_same_point = True
        for i in range(POINT_NUM):
            if math.fabs(pts1[i][INDEX_X] - pts2[i][INDEX_X]) > same_point_eps or math.fabs(
                    pts1[i][INDEX_Y] - pts2[i][INDEX_Y]) > same_point_eps:
                is_same_point = False
                break
        return is_same_point

    def calculate_vertices(self, cur_pts, other_pts, other_vec, inter_pts, num):
        ab = other_vec[0]
        da = other_vec[3]
        ab_dot_ab = self.vec_dot(ab, ab)
        ad_dot_ad = self.vec_dot(da, da)

        for i in range(POINT_NUM):
            # assume ABCD is the rectangle, and P is the point to be judged
            # P is inside ABCD iff. P's projection on ab lies within ab
            # and P's projection on AD lies within AD
            ap = cur_pts[i] - other_pts[0]
            ap_dot_ab = self.vec_dot(ap, ab)
            ap_dot_ad = -self.vec_dot(ap, da)
            is_dot_valid1 = (ap_dot_ab >= 0) and (ap_dot_ad >= 0)
            is_dot_valid2 = (ap_dot_ab <= ab_dot_ab) and (ap_dot_ad <= ad_dot_ad)
            if is_dot_valid1 and is_dot_valid2:
                inter_pts[num] = cur_pts[i]
                num += 1
        return inter_pts, num

    def calculate_intersection(self, pts1, pts2, inter_pts):
        # Line vector
        # A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
        vec1 = torch.zeros(POINT_NUM, LENS_TWO)
        vec2 = torch.zeros(POINT_NUM, LENS_TWO)
        for i in range(POINT_NUM):
            iNext = (i + 1) & 0x3
            vec1[i] = pts1[iNext] - pts1[i]
            vec2[i] = pts2[iNext] - pts2[i]

        # Line test - test all line combos for intersection
        num = 0
        for i in range(POINT_NUM):
            for j in range(POINT_NUM):
                # Solve for 2x2 Ax=b
                # This takes care of parallel lines
                det = self.cross2d(vec2[j], vec1[i])
                if math.fabs(det) <= EPS:
                    continue

                vec12 = pts2[j] - pts1[i]
                t1 = self.cross2d(vec2[j], vec12) / det
                t2 = self.cross2d(vec1[i], vec12) / det
                if 0. <= t1 <= 1. and 0. <= t2 <= 1.:
                    inter_pts[num] = pts1[i] + vec1[i] * t1
                    num += 1

        # Check for vertices from rect1 inside rect2
        inter_pts, num = self.calculate_vertices(pts1, pts2, vec2, inter_pts, num)
        # Check for vertices from rect2 inside rect1
        inter_pts, num = self.calculate_vertices(pts2, pts1, vec1, inter_pts, num)
        return inter_pts, num

    def tcmp(self, p1, p2):
        temp = self.cross2d(p1, p2)
        if math.fabs(temp) < 1e-6:
            if self.vec_dot(p1, p1) < self.vec_dot(p2, p2):
                return -1
            else:
                return 1
        else:
            if temp > 0:
                return -1
            else:
                return 1

    def convex_hull_graham(self, inter_pts, num):
        # Step 1:
        # Find point with minimum y
        # if more than 1 points have the same minimum y,
        # pick the one with the mimimum x.
        t = 0
        for i in range(1, num):
            if inter_pts[i][INDEX_Y] < inter_pts[t][INDEX_Y] or (
                    inter_pts[i][INDEX_Y] == inter_pts[t][INDEX_Y] and inter_pts[i][INDEX_X] < inter_pts[t][INDEX_X]):
                t = i
        s = inter_pts[t]  # starting point

        # Step 2:
        # Subtract starting point from every points (for sorting in the next step)
        ordered_pts = torch.zeros(num, LENS_TWO)
        for i in range(num):
            ordered_pts[i] = inter_pts[i] - s

        # Swap the starting point to position 0
        tmp = ordered_pts[0].clone()
        ordered_pts[0] = ordered_pts[t]
        ordered_pts[t] = tmp

        # Step 3:
        # Sort point 1 ~ num according to their relative cross-product values
        # (essentially sorting according to angles)
        ordered_pts = ordered_pts.tolist()
        ordered_pts[1:] = sorted(ordered_pts[1:], key=functools.cmp_to_key(self.tcmp))
        ordered_pts = torch.tensor(ordered_pts)

        # Step 4:
        # Make sure there are at least 2 points (that don't overlap with each other)
        # in the stack
        k = 1  # index of the non-overlapped second point
        while k < num:
            if self.vec_dot(ordered_pts[k], ordered_pts[k]) > 1e-8:
                break
            k += 1
        if k == num:
            # We reach the end, which means the convex hull is just one point
            ordered_pts[0] = inter_pts[t]
            return 1, ordered_pts
        ordered_pts[1] = ordered_pts[k]

        # Step 5:
        # Finally we can start the scanning process.
        # If we find a non-convex relationship between the 3 points,
        # we pop the previous point from the stack until the stack only has two
        # points, or the 3-point relationship is convex again
        m = 2  # 2 elements in the stack
        previous_index = 2
        for i in range(k + 1, num):
            while m > 1 and self.triangle_area(ordered_pts[i], ordered_pts[m - 1],
                                               ordered_pts[m - previous_index]) >= 0:
                m -= 1
            ordered_pts[m] = ordered_pts[i]
            m += 1

        # Step 6 (Optional):
        # In general sense we need the original coordinates, so we
        # need to shift the points back (reverting Step 2)
        # But if we're only interested in getting the area/perimeter of the shape
        # We can simply return.
        return m, ordered_pts

    def rotated_rect_iou(self, rotated_rect_a, rotated_rect_b, mode_type):
        # Shift rectangles closer to origin (0, 0) to improve the calculation of the intesection region
        # To do that, the average center of the rectangles is moved to the origin
        shifted_a = RectInfo()
        shifted_b = RectInfo()
        if mode_type == 0:
            shifted_a.set_info(rotated_rect_a.clone())
            shifted_b.set_info(rotated_rect_b.clone())
        else:
            shifted_a.set_info_with_xy(rotated_rect_a.clone())
            shifted_b.set_info_with_xy(rotated_rect_b.clone())

        shifted_center_x = (shifted_a.cx + shifted_b.cx) * 0.5
        shifted_center_y = (shifted_a.cy + shifted_b.cy) * 0.5

        shifted_a.cx -= shifted_center_x
        shifted_a.cy -= shifted_center_y
        shifted_b.cx -= shifted_center_x
        shifted_b.cy -= shifted_center_y

        pts1 = self.rect_to_points(shifted_a)
        pts2 = self.rect_to_points(shifted_b)

        # Special case of overlap = 0
        if shifted_a.size < EPS or shifted_b.size < EPS:
            return 0.

        # Specical case of rect1 == rect2
        if self.is_same_rect(pts1, pts2):
            return 1.0

        # There are up to 4 x 4 + 4 + 4 = 24 intersections (including dups) returned
        # from rotated_rect_intersection_pts
        inter_pts = torch.zeros(TOTAL_INTER_POINTS, LENS_TWO)
        inter_pts, num = self.calculate_intersection(pts1, pts2, inter_pts)
        MINI_PTS = 2
        if num <= MINI_PTS:
            return 0.0

        # Convex Hull to order the intersection points in clockwise or
        # counter-clockwise order and find the countour area.
        num_convex, ordered_pts = self.convex_hull_graham(inter_pts, num)
        inter_area = self.contour_area(ordered_pts, num_convex)
        return inter_area / (shifted_a.size + shifted_b.size - inter_area)

    def get_sorted_index_by_core(self, scores_tensor):
        score_index_vec = list(enumerate(scores_tensor))
        score_index_vec = sorted(score_index_vec, key=lambda x: x[1], reverse=True)
        return score_index_vec

    def compute(self, boxes_tensor, scores_tensor, iou_threshold, mode):
        boxes_tensor = boxes_tensor.float()
        scores_tensor = scores_tensor.float()
        score_index_vec = self.get_sorted_index_by_core(scores_tensor)
        indices = []
        selected_box = []
        total_num = 0

        for i, _ in enumerate(score_index_vec):
            idx = score_index_vec[i][0]
            keep = True
            k = 0

            while (k < total_num) and keep:
                kept_idx = indices[k]
                overlap = self.rotated_rect_iou(boxes_tensor[idx], boxes_tensor[kept_idx], mode)
                keep = (overlap <= iou_threshold)
                k += 1

            if keep:
                indices.append(idx)
                selected_box.append(boxes_tensor[idx])
                total_num += 1

        return indices, total_num

    def cpu_op_exec(self, det, score, iou_threshold, mode):
        output1, output2 = self.compute(det, score, iou_threshold, mode)
        return output1, [output2]

    def npu_op_exec(self, det, score, iou_threshold, mode):
        output1, output2 = torch_npu.npu_nms_rotated(det, score, iou_threshold=iou_threshold, mode=mode,
                                                     scores_threshold=0, max_output_size=-1)
        return output1, output2

    def test_npu_nms_rotated(self):
        items = [[[np.float32, 0, [20, 5]], [np.float32, 0, [20]], 0.2, 0],
                 [[np.float32, 0, [20, 5]], [np.float32, 0, [20]], 0.2, 1],
                 [[np.float16, 0, [20, 5]], [np.float16, 0, [20]], 0.2, 0],
                 [[np.float16, 0, [20, 5]], [np.float16, 0, [20]], 0.2, 1]]

        for item in items:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100., 100.)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0., 1.)
            iou_threshold = item[2]
            mode = item[3]

            cpu_output1, cpu_output2 = self.cpu_op_exec(cpu_input1, cpu_input2, iou_threshold, mode)
            npu_output1, npu_output2 = self.npu_op_exec(npu_input1, npu_input2, iou_threshold, mode)
            self.assertEqual(cpu_output1, npu_output1.cpu().numpy())
            self.assertEqual(cpu_output2, npu_output2.cpu().numpy())


if __name__ == "__main__":
    run_tests()
