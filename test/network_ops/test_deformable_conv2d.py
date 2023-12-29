# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestDeformableConv2d(TestCase):
    def create_single_npu_tensor(self, item, minvalue, maxvalue):
        dtype = item[0]
        format1 = item[1]
        shape = item[2]
        input1 = np.random.uniform(minvalue, maxvalue, shape).astype(dtype)
        npu_input = torch.from_numpy(input1).to("npu")
        if format1 != -1:
            npu_input = npu_input.npu_format_cast(format1)
        return npu_input

    def helper_gen(self, offsets_shape, kernel_sizes, strides, pads, dialation):
        H_OUT = offsets_shape[1]
        W_OUT = offsets_shape[2]
        K_H, K_W = kernel_sizes
        STRIDED_H, STRIDED_W = strides[1], strides[2]
        dialation_h, dialation_w = dialation[1], dialation[2]
        try:
            group = offsets_shape[3] / 3 / kernel_sizes[0] / kernel_sizes[1]
        except ZeroDivisionError as e:
            print("kernel_sizes can not be 0.")
        group = int(group)

        pad_top, pad_left = pads[0], pads[2]
        helper_tensor = np.zeros((H_OUT, W_OUT, 3 * group * K_H * K_W), np.float32)
        for h in range(H_OUT):
            for w in range(W_OUT):
                for k_h in range(K_H):
                    for k_w in range(K_W):
                        for g in range(group):
                            helper_tensor[h][w][0 * group * K_H * K_W + g * K_H * K_W + k_h * K_W +
                                                k_w] = w * STRIDED_W - pad_left + k_w * dialation_w
                            helper_tensor[h][w][1 * group * K_H * K_W + g * K_H * K_W + k_h * K_W +
                                                k_w] = h * STRIDED_H - pad_top + k_h * dialation_h

        return helper_tensor

    def deformable_offsets(self, x, offsets, args):
        kernel_size, strides, pads, dilations = args
        dtype = x.dtype
        if dtype == np.float16:
            x = x.astype(np.float32)
            offsets = offsets.astype(np.float32)
        N, H_OUT, W_OUT, _ = offsets.shape
        H_IN = x.shape[1]
        W_IN = x.shape[2]
        C = x.shape[-1]
        K_H, K_W = kernel_size
        GROUP = offsets.shape[-1] // K_H // K_W // 3
        GROUP_C = C // GROUP
        helper = self.helper_gen(offsets.shape, kernel_size, strides, pads, dilations)

        x = x.reshape((N, H_IN, W_IN, GROUP, GROUP_C))
        offsets = offsets.reshape((N, H_OUT, W_OUT, 3, GROUP, K_H, K_W))
        helper = helper.reshape((H_OUT, W_OUT, 3, GROUP, K_H, K_W))
        index_offsets = offsets + helper

        floor_index = np.floor(index_offsets)
        ceil_index = floor_index + 1

        int32_ceil_index = ceil_index.astype(np.int32)
        int32_floor_index = floor_index.astype(np.int32)

        l_t_tensor = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        l_b_tensor = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        r_t_tensor = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        r_b_tensor = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)

        for n in range(N):
            for h_out in range(H_OUT):
                for k_h in range(K_H):
                    for w_out in range(W_OUT):
                        for k_w in range(K_W):
                            for g in range(GROUP):
                                l_t_h = int32_floor_index[n][h_out][w_out][1][g][k_h][k_w]
                                l_t_w = int32_floor_index[n][h_out][w_out][0][g][k_h][k_w]

                                if 0 <= l_t_h < H_IN and 0 <= l_t_w < W_IN:
                                    l_t_tensor[n][h_out][k_h][w_out][k_w] = x[n][l_t_h][l_t_w][g]
                                else:
                                    l_t_tensor[n][h_out][k_h][w_out][k_w] = 0

                                l_b_h = int32_ceil_index[n][h_out][w_out][1][g][k_h][k_w]
                                l_b_w = int32_floor_index[n][h_out][w_out][0][g][k_h][k_w]

                                if 0 <= l_b_h < H_IN and 0 <= l_b_w < W_IN:
                                    l_b_tensor[n][h_out][k_h][w_out][k_w] = x[n][l_b_h][l_b_w][g]
                                else:
                                    l_b_tensor[n][h_out][k_h][w_out][k_w] = 0

                                r_t_h = int32_floor_index[n][h_out][w_out][1][g][k_h][k_w]
                                r_t_w = int32_ceil_index[n][h_out][w_out][0][g][k_h][k_w]

                                if 0 <= r_t_h < H_IN and 0 <= r_t_w < W_IN:
                                    r_t_tensor[n][h_out][k_h][w_out][k_w] = x[n][r_t_h][r_t_w][g]
                                else:
                                    r_t_tensor[n][h_out][k_h][w_out][k_w] = 0

                                r_b_h = int32_ceil_index[n][h_out][w_out][1][g][k_h][k_w]
                                r_b_w = int32_ceil_index[n][h_out][w_out][0][g][k_h][k_w]

                                if 0 <= r_b_h < H_IN and 0 <= r_b_w < W_IN:
                                    r_b_tensor[n][h_out][k_h][w_out][k_w] = x[n][r_b_h][r_b_w][g]
                                else:
                                    r_b_tensor[n][h_out][k_h][w_out][k_w] = 0

        ceil_sub_value = ceil_index - index_offsets
        ceil_sub_value = 1 - ceil_sub_value
        sub_floor_value = index_offsets - floor_index
        sub_floor_value = 1 - sub_floor_value

        l_t_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        l_b_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        r_t_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        r_b_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)

        scale_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        for n in range(N):
            for h_out in range(H_OUT):
                for k_h in range(K_H):
                    for w_out in range(W_OUT):
                        for k_w in range(K_W):
                            for g in range(GROUP):
                                l_t_h = sub_floor_value[n][h_out][w_out][1][g][k_h][k_w]
                                l_t_w = sub_floor_value[n][h_out][w_out][0][g][k_h][k_w]
                                l_t_weight[n][h_out][k_h][w_out][k_w][g] = l_t_h * l_t_w

                                l_b_h = ceil_sub_value[n][h_out][w_out][1][g][k_h][k_w]
                                l_b_w = sub_floor_value[n][h_out][w_out][0][g][k_h][k_w]
                                l_b_weight[n][h_out][k_h][w_out][k_w][g] = l_b_h * l_b_w

                                r_t_h = sub_floor_value[n][h_out][w_out][1][g][k_h][k_w]
                                r_t_w = ceil_sub_value[n][h_out][w_out][0][g][k_h][k_w]
                                r_t_weight[n][h_out][k_h][w_out][k_w][g] = r_t_h * r_t_w

                                r_b_h = ceil_sub_value[n][h_out][w_out][1][g][k_h][k_w]
                                r_b_w = ceil_sub_value[n][h_out][w_out][0][g][k_h][k_w]
                                r_b_weight[n][h_out][k_h][w_out][k_w][g] = r_b_h * r_b_w

                                scale_weight[n][h_out][k_h][w_out][k_w][g] = offsets[n][h_out][w_out][2][g][k_h][k_w]
        out_tensor = \
            l_t_tensor * l_t_weight + l_b_tensor * l_b_weight + r_t_tensor * r_t_weight + r_b_tensor * r_b_weight
        out_tensor = out_tensor * scale_weight
        if dtype == np.float16:
            out_tensor = out_tensor.astype(np.float16)
        return out_tensor.reshape((N, H_OUT * K_H, W_OUT * K_W, C))

    def deformable_offsets_grad(self, grad_y, x, offsets, args):
        kernel_size, strides, pads, dilation = args
        flag = True
        if x.dtype == np.float16:
            flag = False
            grad_y = grad_y.astype(np.float32)
            x = x.astype(np.float32)
            offsets = offsets.astype(np.float32)
        N, H_IN, W_IN, C = x.shape
        K_H, K_W = kernel_size
        H_OUT = grad_y.shape[1] // K_H
        W_OUT = grad_y.shape[2] // K_W
        GROUP = offsets.shape[3] // K_H // K_W // 3
        GROUP_C = C // GROUP

        helper = self.helper_gen(offsets.shape, kernel_size, strides, pads, dilation)

        index_offsets = offsets + helper
        x = x.reshape((N, H_IN, W_IN, GROUP, GROUP_C))
        offsets = offsets.reshape((N, H_OUT, W_OUT, 3, GROUP, K_H, K_W))
        index_offsets = index_offsets.reshape((N, H_OUT, W_OUT, 3, GROUP, K_H, K_W))
        grad_y = grad_y.reshape((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C))

        ceil_index = np.floor(index_offsets) + 1
        floor_index = np.floor(index_offsets)
        ceil_sub_value = ceil_index - index_offsets
        ceil_sub_value = 1 - ceil_sub_value

        sub_floor_value = index_offsets - floor_index
        sub_floor_value = 1 - sub_floor_value

        int32_ceil_index = ceil_index.astype(np.int32)
        int32_floor_index = floor_index.astype(np.int32)

        l_t_h_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        l_t_w_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)

        l_b_h_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        l_b_w_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)

        r_t_h_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        r_t_w_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)

        r_b_h_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        r_b_w_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)

        scale_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)

        for n in range(N):
            for h_out in range(H_OUT):
                for k_h in range(K_H):
                    for w_out in range(W_OUT):
                        for k_w in range(K_W):
                            for g in range(GROUP):
                                l_t_h = sub_floor_value[n][h_out][w_out][1][g][k_h][k_w]
                                l_t_w = sub_floor_value[n][h_out][w_out][0][g][k_h][k_w]
                                l_t_h_weight[n][h_out][k_h][w_out][k_w][g] = l_t_h
                                l_t_w_weight[n][h_out][k_h][w_out][k_w][g] = l_t_w

                                l_b_h = ceil_sub_value[n][h_out][w_out][1][g][k_h][k_w]
                                l_b_w = sub_floor_value[n][h_out][w_out][0][g][k_h][k_w]
                                l_b_h_weight[n][h_out][k_h][w_out][k_w][g] = l_b_h
                                l_b_w_weight[n][h_out][k_h][w_out][k_w][g] = l_b_w

                                r_t_h = sub_floor_value[n][h_out][w_out][1][g][k_h][k_w]
                                r_t_w = ceil_sub_value[n][h_out][w_out][0][g][k_h][k_w]
                                r_t_h_weight[n][h_out][k_h][w_out][k_w][g] = r_t_h
                                r_t_w_weight[n][h_out][k_h][w_out][k_w][g] = r_t_w

                                r_b_h = ceil_sub_value[n][h_out][w_out][1][g][k_h][k_w]
                                r_b_w = ceil_sub_value[n][h_out][w_out][0][g][k_h][k_w]
                                r_b_h_weight[n][h_out][k_h][w_out][k_w][g] = r_b_h
                                r_b_w_weight[n][h_out][k_h][w_out][k_w][g] = r_b_w

                                scale_weight[n][h_out][k_h][w_out][k_w][g] = offsets[n][h_out][w_out][2][g][k_h][k_w]

        dx = np.zeros((N, H_IN, W_IN, GROUP, GROUP_C), np.float32)
        grad_y_mul_scale = grad_y * scale_weight

        grad_y_mul_scale_mul_l_t_weight = grad_y_mul_scale * l_t_h_weight * l_t_w_weight
        grad_y_mul_scale_mul_l_b_weight = grad_y_mul_scale * l_b_h_weight * l_b_w_weight
        grad_y_mul_scale_mul_r_t_weight = grad_y_mul_scale * r_t_h_weight * r_t_w_weight
        grad_y_mul_scale_mul_r_b_weight = grad_y_mul_scale * r_b_h_weight * r_b_w_weight
        for n in range(N):
            for h_out in range(H_OUT):
                for k_h in range(K_H):
                    for w_out in range(W_OUT):
                        for k_w in range(K_W):
                            for g in range(GROUP):
                                index_offsets_h = index_offsets[n][h_out][w_out][1][g][k_h][k_w]
                                index_offsets_w = index_offsets[n][h_out][w_out][0][g][k_h][k_w]

                                l_t_h = int32_floor_index[n][h_out][w_out][1][g][k_h][k_w]
                                l_t_w = int32_floor_index[n][h_out][w_out][0][g][k_h][k_w]
                                if -1 < index_offsets_h < H_IN and -1 < index_offsets_w < W_IN:
                                    if 0 <= l_t_h < H_IN and 0 <= l_t_w < W_IN:
                                        dx[n][l_t_h][l_t_w][g] += \
                                            grad_y_mul_scale_mul_l_t_weight[n][h_out][k_h][w_out][k_w][g]

                                l_b_h = int32_ceil_index[n][h_out][w_out][1][g][k_h][k_w]
                                l_b_w = int32_floor_index[n][h_out][w_out][0][g][k_h][k_w]
                                if -1 < index_offsets_h < H_IN and -1 < index_offsets_w < W_IN:
                                    if 0 <= l_b_h < H_IN and 0 <= l_b_w < W_IN:
                                        dx[n][l_b_h][l_b_w][g] += \
                                            grad_y_mul_scale_mul_l_b_weight[n][h_out][k_h][w_out][k_w][g]

                                r_t_h = int32_floor_index[n][h_out][w_out][1][g][k_h][k_w]
                                r_t_w = int32_ceil_index[n][h_out][w_out][0][g][k_h][k_w]
                                if -1 < index_offsets_h < H_IN and -1 < index_offsets_w < W_IN:
                                    if 0 <= r_t_h < H_IN and 0 <= r_t_w < W_IN:
                                        dx[n][r_t_h][r_t_w][g] += \
                                            grad_y_mul_scale_mul_r_t_weight[n][h_out][k_h][w_out][k_w][g]

                                r_b_h = int32_ceil_index[n][h_out][w_out][1][g][k_h][k_w]
                                r_b_w = int32_ceil_index[n][h_out][w_out][0][g][k_h][k_w]
                                if -1 < index_offsets_h < H_IN and -1 < index_offsets_w < W_IN:
                                    if 0 <= r_b_h < H_IN and 0 <= r_b_w < W_IN:
                                        dx[n][r_b_h][r_b_w][g] += \
                                            grad_y_mul_scale_mul_r_b_weight[n][h_out][k_h][w_out][k_w][g]

        dx = dx.reshape((N, H_IN, W_IN, C))

        A_tensor = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C)).astype("float32")
        B_tensor = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C)).astype("float32")
        C_tensor = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C)).astype("float32")
        D_tensor = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C)).astype("float32")
        for n in range(N):
            for h_out in range(H_OUT):
                for w_out in range(W_OUT):
                    for g in range(GROUP):
                        for k_h in range(K_H):
                            for k_w in range(K_W):
                                index_offsets_h = index_offsets[n][h_out][w_out][1][g][k_h][k_w]
                                index_offsets_w = index_offsets[n][h_out][w_out][0][g][k_h][k_w]

                                l_t_h = int32_floor_index[n][h_out][w_out][1][g][k_h][k_w]
                                l_t_w = int32_floor_index[n][h_out][w_out][0][g][k_h][k_w]
                                if -1 < index_offsets_h < H_IN and -1 < index_offsets_w < W_IN:
                                    if 0 <= l_t_h < H_IN and 0 <= l_t_w < W_IN:
                                        A_tensor[n][h_out][k_h][w_out][k_w][g] = x[n][l_t_h][l_t_w][g]
                                    else:
                                        A_tensor[n][h_out][k_h][w_out][k_w][g] = 0

                                l_b_h = int32_ceil_index[n][h_out][w_out][1][g][k_h][k_w]
                                l_b_w = int32_floor_index[n][h_out][w_out][0][g][k_h][k_w]
                                if -1 < index_offsets_h < H_IN and -1 < index_offsets_w < W_IN:
                                    if 0 <= l_b_h < H_IN and 0 <= l_b_w < W_IN:
                                        B_tensor[n][h_out][k_h][w_out][k_w][g] = x[n][l_b_h][l_b_w][g]
                                    else:
                                        B_tensor[n][h_out][k_h][w_out][k_w][g] = 0

                                r_t_h = int32_floor_index[n][h_out][w_out][1][g][k_h][k_w]
                                r_t_w = int32_ceil_index[n][h_out][w_out][0][g][k_h][k_w]
                                if -1 < index_offsets_h < H_IN and -1 < index_offsets_w < W_IN:
                                    if 0 <= r_t_h < H_IN and 0 <= r_t_w < W_IN:
                                        C_tensor[n][h_out][k_h][w_out][k_w][g] = x[n][r_t_h][r_t_w][g]
                                    else:
                                        C_tensor[n][h_out][k_h][w_out][k_w][g] = 0

                                r_b_h = int32_ceil_index[n][h_out][w_out][1][g][k_h][k_w]
                                r_b_w = int32_ceil_index[n][h_out][w_out][0][g][k_h][k_w]
                                if -1 < index_offsets_h < H_IN and -1 < index_offsets_w < W_IN:
                                    if 0 <= r_b_h < H_IN and 0 <= r_b_w < W_IN:
                                        D_tensor[n][h_out][k_h][w_out][k_w][g] = x[n][r_b_h][r_b_w][g]
                                    else:
                                        D_tensor[n][h_out][k_h][w_out][k_w][g] = 0

        grad_offset_h = grad_y_mul_scale * (
            -A_tensor * l_t_w_weight + B_tensor * l_b_w_weight - C_tensor * r_t_w_weight + D_tensor * r_b_w_weight)
        grad_offset_w = grad_y_mul_scale * (
            -A_tensor * l_t_h_weight - B_tensor * l_b_h_weight + C_tensor * r_t_h_weight + D_tensor * r_b_h_weight)

        grad_scale = A_tensor * l_t_h_weight * l_t_w_weight * grad_y + \
            B_tensor * l_b_h_weight * l_b_w_weight * grad_y + \
            C_tensor * r_t_h_weight * r_t_w_weight * grad_y + \
            D_tensor * r_b_h_weight * r_b_w_weight * grad_y

        grad_offset = np.concatenate([grad_offset_w, grad_offset_h, grad_scale], axis=2)
        grad_offset = grad_offset.reshape(
            (N, H_OUT, 3, K_H, W_OUT, K_W, GROUP, GROUP_C)
        ).transpose(
            (0, 1, 4, 2, 6, 3, 5, 7)
        ).reshape(
            (N, H_OUT, W_OUT, 3 * GROUP * K_H * K_W, GROUP_C)
        )
        grad_offset = np.sum(grad_offset, axis=-1, keepdims=False)

        return dx, grad_offset

    def get_fwd_golden(self, x, weight, offset, args):
        ksize, strides, pads, dilations, groups = args
        x_nhwc = torch_npu.npu_transpose(x, (0, 2, 3, 1), True).cpu().numpy()
        o_nhwc = torch_npu.npu_transpose(offset, (0, 2, 3, 1), True).cpu().numpy()
        deformable_offsets_args = (ksize, strides, pads, dilations)
        deformable_offsets_out = self.deformable_offsets(x_nhwc, o_nhwc, deformable_offsets_args)
        deformable_offsets_out_nchw = torch_npu.npu_transpose(
            torch.from_numpy(deformable_offsets_out).npu(), (0, 3, 1, 2), True)
        conv2d_out = torch_npu.npu_conv2d(
            deformable_offsets_out_nchw, weight, None, ksize, (0, 0, 0, 0), (1, 1), groups)
        return conv2d_out, deformable_offsets_out_nchw

    def get_bkw_golden(self, inputs, args):
        x, grad, deformable_offsets_out, weight, offset = inputs
        ksize, strides, pads, dilations = args
        conv2d_dx, conv2d_dw, conv2d_db = torch.ops.aten.npu_conv2d_backward(
            deformable_offsets_out, grad, weight, ksize, (0, 0, 0, 0), (1, 1), 1, (True, True, True))
        conv2d_dx_nhwc = torch_npu.npu_transpose(conv2d_dx, (0, 2, 3, 1), True).cpu().numpy()
        x_nhwc = torch_npu.npu_transpose(x, (0, 2, 3, 1), True).cpu().numpy()
        o_nhwc = torch_npu.npu_transpose(offset, (0, 2, 3, 1), True).cpu().numpy()
        dx_nhwc, do_nhwc = self.deformable_offsets_grad(
            conv2d_dx_nhwc, x_nhwc, o_nhwc, args)
        dx = torch_npu.npu_transpose(torch.from_numpy(dx_nhwc).npu(), (0, 3, 1, 2), True)
        do = torch_npu.npu_transpose(torch.from_numpy(do_nhwc).npu(), (0, 3, 1, 2), True)
        return [dx, conv2d_dw, do, conv2d_db]

    def test_deformable_conv2d(self):
        npu_x = self.create_single_npu_tensor([np.float32, 0, (1, 32, 32, 32)], 0, 10)
        npu_w = self.create_single_npu_tensor([np.float32, 0, (32, 32, 5, 5)], 0, 10)
        npu_o = self.create_single_npu_tensor([np.float32, 0, (1, 75, 32, 32)], 0, 10)
        ksize = [5, 5]
        strides = [1, 1, 1, 1]
        pads = [2, 2, 2, 2]
        dilations = [1, 1, 1, 1]
        groups = 1

        dcn_out, deformable_offsets_out = torch_npu.npu_deformable_conv2d(
            npu_x, npu_w, npu_o, None, kernel_size=ksize, stride=strides, padding=pads)
        args = (ksize, strides, pads, dilations, groups)
        dcn_golden, deformable_offsets_golden = self.get_fwd_golden(npu_x, npu_w, npu_o, args)
        self.assertRtolEqual(dcn_golden.cpu().detach(), dcn_out.cpu().detach())
        self.assertRtolEqual(deformable_offsets_golden.cpu().detach(), deformable_offsets_out.cpu().detach())

        npu_grad = torch.ones_like(dcn_out)
        dcn_dx, dcn_dw, dcn_do, dcn_db = torch_npu.npu_deformable_conv2dbk(
            npu_x, npu_grad, deformable_offsets_out, npu_w, npu_o, kernel_size=ksize, stride=strides, padding=pads)
        bkw_inputs = (npu_x, npu_grad, deformable_offsets_out, npu_w, npu_o)
        dcn_bk_golden_list = self.get_bkw_golden(bkw_inputs, args[:4])
        self.assertRtolEqual(dcn_bk_golden_list[0].cpu().detach(), dcn_dx.cpu().detach())
        self.assertRtolEqual(dcn_bk_golden_list[1].cpu().detach(), dcn_dw.cpu().detach())
        self.assertRtolEqual(dcn_bk_golden_list[2].cpu().detach(), dcn_do.cpu().detach(), 0.0002)
        self.assertRtolEqual(dcn_bk_golden_list[3].cpu().detach(), dcn_db.cpu().detach())


if __name__ == "__main__":
    np.random.seed(123)
    torch.npu.conv.allow_hf32 = False
    run_tests()
