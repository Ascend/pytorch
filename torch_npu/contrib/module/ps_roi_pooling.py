import torch
from torch import nn
import torch_npu


class PSROIPool(nn.Module):
    def __init__(self, pooled_height=7, pooled_width=7, spatial_scale=1 / 16.0, group_size=7, output_dim=22):
        """ROIAlign using npu api.

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
        """

        super(PSROIPool, self).__init__()

        if not (pooled_height == pooled_width == group_size):
            raise ValueError("only pooled_height == pooled_width == group_size supported.")

        self.group_size = group_size
        self.spatial_scale = spatial_scale
        self.output_dim = output_dim

    def forward(self, features, rois):
        '''
        rois needs to follow the specified format, please refer to get_random_rois function in this scripts.
        '''

        return torch_npu.npu_ps_roi_pooling(features,
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
