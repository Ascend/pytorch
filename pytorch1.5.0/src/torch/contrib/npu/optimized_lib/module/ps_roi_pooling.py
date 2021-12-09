# Copyright (c) 2020 Huawei Technologies Co., Ltd
# All rights reserved.
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
from torch import nn

class PSROIPool(nn.Module):
    def __init__(self, pooled_height=7, pooled_width=7, spatial_scale=1 / 16.0, group_size=7, output_dim=22):
        """ROIAlign using npu api.

        Origin implement is
        https://github.com/RebornL/RFCN-pytorch.1.0/blob/master/lib/model/roi_layers/ps_roi_pool.py

        Args:
            pooled_height (int): pooled_height
            pooled_width (int): pooled_width
            spatial_scale (float): scale the input boxes by this number
            group_size (int): number of groups encoding position sensitive score maps
            output_dim (int):number of output channels

        Note:
            only pooled_height == pooled_width == group_size implemented.

        Examples::
            >>> model = PSROIPool(pooled_height=7, pooled_width=7, spatial_scale=1 / 16.0, group_size=7, output_dim=22)

        .. _R-FCN\: Object Detection via Region-based Fully Convolutional Networks
            https://arxiv.org/abs/1605.06409
        """

        super(PSROIPool, self).__init__()

        assert (pooled_height == pooled_width == group_size), \
            "only pooled_height == pooled_width == group_size supported."

        self.group_size = group_size
        self.spatial_scale = spatial_scale
        self.output_dim = output_dim

    def forward(self, features, rois):
        '''
        rois needs to follow the specified format, please refer to get_random_rois function in this scripts.
        '''

        return torch.npu_ps_roi_pooling(features,
                                        rois,
                                        self.spatial_scale,
                                        self.group_size,
                                        self.output_dim)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "pooled_width=" + str(self.pooled_width)
        tmpstr += ", pooled_height=" + str(self.pooled_height)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", group_size=" + str(self.group_size)
        tmpstr += ", output_dim=" + str(self.output_dim)
        tmpstr += ")"
        return tmpstr


def get_random_rois(shape):
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


if __name__ == "__main__":
    cls_feat = torch.randn(4, 1078, 84, 84).float()
    cls_feat.requires_grad = True
    rois_tensor = get_random_rois((4, 128, 5)).permute(0, 2, 1).float()

    model = PSROIPool(pooled_height=7, pooled_width=7, spatial_scale=1 / 16.0, group_size=7, output_dim=22)

    torch.npu.set_device(0)
    cls_feat = cls_feat.npu()
    rois_tensor = rois_tensor.npu()

    x = model(cls_feat, rois_tensor)  # 512,22,7,7
    l = x.sum()
    l.backward()
