import torch
from torch import _VF

import torch_npu


class NPULinearOP(object):

    @staticmethod
    def forward(input_, weight, bias=None):
        if torch.onnx.is_in_onnx_export():
            return torch._C._nn.linear(input_, weight, bias)
        return torch.ops.npu.npu_linear(input_, weight, bias)


class NPUTransposeOP(object):

    @staticmethod
    def forward(self, perm, require_contiguous=True, out=None):
        if torch.onnx.is_in_onnx_export():
            if require_contiguous:
                out = torch.permute(self, perm).contiguous()
            else:
                out = torch.permute(self, perm)
            return out
        out = torch.ops.npu.npu_transpose(
            self, perm, require_contiguous)
        return out


class NPUBroadcastOP(object):

    @staticmethod
    def forward(self, size, out=None):
        if torch.onnx.is_in_onnx_export():
            out = self.expand(size)
            return out
        out = torch.ops.npu.npu_broadcast(self, size)
        return out


class NPUConvTranspose2dOP(object):

    @staticmethod
    def forward(input_, weight, bias, padding, output_padding, stride, dilation, groups):
        if torch.onnx.is_in_onnx_export():
            return torch.conv_transpose2d(input_, weight, bias, stride, padding,
                                          output_padding, groups, dilation)
        return torch.ops.npu.npu_conv_transpose2d(input_, weight, bias,
                                                                         padding, output_padding,
                                                                         stride, dilation, groups)


class NPUConv2dOP(object):

    @staticmethod
    def forward(input_, weight, bias, stride, padding, dilation, groups):
        if torch.onnx.is_in_onnx_export():
            return torch.conv2d(input_, weight, bias, stride, padding, dilation, groups)
        return torch.ops.npu.npu_conv2d(input_, weight, bias, stride,
                                                               padding, dilation, groups)


class NPUConv3dOP(object):

    @staticmethod
    def forward(input_, weight, bias, stride, padding, dilation, groups):
        if torch.onnx.is_in_onnx_export():
            return torch.conv3d(input_, weight, bias, stride, padding, dilation, groups)
        return torch.ops.npu.npu_conv3d(input_, weight, bias, stride,
                                                               padding, dilation, groups)


class NPUStrideCopyOP(object):

    @staticmethod
    def forward(self, shape, stride, storage_offset, out=None):
        if torch.onnx.is_in_onnx_export():
            out = torch.as_strided(self, shape, stride, 0).clone()
            return out
        out = torch.ops.npu.npu_stride_copy(self, shape, stride, storage_offset)
        return out


class NPUSortV2OP(object):

    @staticmethod
    def forward(self, dim=-1, descending=False, out=None):
        if torch.onnx.is_in_onnx_export():
            out, indices = torch.sort(self, dim, descending)
            return out
        out = torch.ops.npu.npu_sort_v2(self, dim, descending)
        return out


class NPULayerNormEvalOP(object):

    @staticmethod
    def forward(input_, normalized_shape, weight=None, bias=None, eps=1e-05):
        if torch.onnx.is_in_onnx_export():
            return torch.layer_norm(input_, normalized_shape, weight, bias, eps, False)
        return torch.ops.npu.npu_layer_norm_eval(input_, normalized_shape, 
                                                                   weight, bias, eps)


class NPUReshapeOP(object):

    @staticmethod
    def forward(self, shape, can_refresh=False, out=None):
        if torch.onnx.is_in_onnx_export():
            if can_refresh:
                out = torch.reshape(self, shape).clone()
            else:
                out = torch.reshape(self, shape)
            return out
        out = torch.ops.npu.npu_reshape(self, shape, can_refresh)
        return out


class NPUPadOP(object):

    @staticmethod
    def forward(input_, paddings):
        if torch.onnx.is_in_onnx_export():
            return torch.nn.functional.pad(input_, paddings[2:] + paddings[:2], "constant", 0)
        return torch.ops.npu.npu_pad(input_, paddings)


class NPUConvolutionOP(object):

    @staticmethod
    def forward(input_, weight, bias, stride, padding, dilation, groups):
        if torch.onnx.is_in_onnx_export():
            dim = input_.dim()
            if dim == 4:
                output = torch.nn.functional.conv2d(input_, weight, bias, stride,
                                                    padding, dilation, groups)
            elif dim == 5:
                is_dilated = False
                for d in dilation:
                    is_dilated |= (d != 1)
                if groups == 1 and not is_dilated:
                    output = torch._C._nn.slow_conv3d(input_, weight, weight.size()[2],
                                                      bias, stride, padding)
                else:
                    output = torch.nn.functional.conv3d(input_, weight, bias, stride,
                                                        padding, dilation, groups)
            else:
                raise ValueError("input dim must be 4 or 5, but got ", dim)
            return output
        else:
            return torch.ops.npu.npu_convolution(input_, weight, bias,
                                                                   stride, padding, dilation, groups)


class NPUConvolutionTransposeOP(object):

    @staticmethod
    def forward(input_, weight, bias, padding, output_padding, stride, dilation, groups):
        if torch.onnx.is_in_onnx_export():
            dim = input_.dim()
            if dim == 4:
                output = torch.conv_transpose2d(input_, weight, bias, stride,
                                                padding, output_padding, groups, dilation)
            elif dim == 5:
                output = torch.conv_transpose3d(input_, weight, bias, stride,
                                                padding, output_padding, groups, dilation)
            else:
                raise ValueError("input dim must be 4 or 5, but got ", dim)
            return output
        else:
            return torch.ops.npu.npu_convolution_transpose(
                input_, weight, bias, padding, output_padding, stride, dilation, groups)


class NPUConfusionTransposeOP(object):

    @staticmethod
    def forward(self, perm, shape, transpose_first):
        if torch.onnx.is_in_onnx_export():
            if transpose_first:
                return self.permute(*perm).contiguous().view(shape)
            else:
                return self.view(shape).permute(*perm)
        return torch.ops.npu.npu_confusion_transpose(self, perm, shape, transpose_first)


class NPUMaxOP(object):

    @staticmethod
    def forward(self, dim, keepdim=False):
        if torch.onnx.is_in_onnx_export():
            values, indices = torch.max(self, dim, keepdim)
            indices = indices.to(torch.int32)
            return values, indices
        return torch.ops.npu.npu_max(self, dim, keepdim)


class NPUBmmV2OP(object):

    @staticmethod
    def forward(self, mat2, output_sizes):
        if torch.onnx.is_in_onnx_export():
            return torch.matmul(self, mat2)
        return torch.ops.npu.npu_bmmV2(self, mat2, output_sizes)


class NPUDtypeCastOP(object):

    @staticmethod
    def forward(self, dtype):
        if torch.onnx.is_in_onnx_export():
            return self.to(dtype)
        return torch.ops.npu.npu_dtype_cast(self, dtype)


class NPUSiluOP(object):

    @staticmethod
    def forward(self):
        if torch.onnx.is_in_onnx_export():
            return self * torch.sigmoid(self)
        return torch.ops.npu.npu_silu(self)


class NPUMinOP(object):

    @staticmethod
    def forward(self, dim, keepdim=False):
        if torch.onnx.is_in_onnx_export():
            outputs, indices = torch.min(self, dim, keepdim)
            indices = indices.to(torch.int32)
            return outputs, indices
        return torch.ops.npu.npu_min(self, dim, keepdim)


class NPUFusedAttentionLayernormQkvFwdOP(object):

    @staticmethod
    def confusion_transpose(x, new_shape):
        perm = (0, 2, 1, 3)
        return torch_npu.npu_confusion_transpose(x, perm, new_shape, False).contiguous()

    @staticmethod
    def forward(x, kernel_query, kernel_key, kernel_value, gamma, beta, 
                bias_query=None, bias_key=None, bias_value=None, seq_len=128, num_heads=12, eps=1e-05):
        if torch.onnx.is_in_onnx_export():
            kernel_query = kernel_query.t().contiguous()
            kernel_key = kernel_key.t().contiguous()
            kernel_value = kernel_value.t().contiguous()

            norm_shape = [x.shape[-1]]
            new_shape = (int(x.shape[0] / seq_len), seq_len, num_heads, int(x.shape[1] / num_heads))

            norm, mean, variance = torch.native_layer_norm(x, norm_shape, gamma, beta, eps=1e-05)
            q_layer = NPUFusedAttentionLayernormQkvFwdOP.confusion_transpose(
                    torch.nn.functional.linear(norm, kernel_query, bias_query), new_shape)
            k_layer = NPUFusedAttentionLayernormQkvFwdOP.confusion_transpose(
                    torch.nn.functional.linear(norm, kernel_key, bias_key), new_shape)
            v_layer = NPUFusedAttentionLayernormQkvFwdOP.confusion_transpose(
                    torch.nn.functional.linear(norm, kernel_value, bias_value), new_shape)

            return [norm, q_layer, k_layer, v_layer, mean, variance]
        
        return torch.ops.npu.npu_fused_attention_layernorm_qkv_fwd(
                                x, kernel_query, kernel_key, kernel_value, gamma, beta, 
                                bias_query=bias_query, bias_key=bias_key, bias_value=bias_value, 
                                seq_len=seq_len, num_heads=num_heads, eps=eps)


def add_ops_combined_for_onnx():
    torch_npu.npu_linear = NPULinearOP.forward
    torch_npu.npu_transpose = NPUTransposeOP.forward
    torch_npu.npu_broadcast = NPUBroadcastOP.forward
    torch_npu.npu_conv_transpose2d = NPUConvTranspose2dOP.forward
    torch_npu.npu_conv2d = NPUConv2dOP.forward
    torch_npu.npu_conv3d = NPUConv3dOP.forward
    torch_npu.npu_stride_copy = NPUStrideCopyOP.forward
    torch_npu.npu_sort_v2 = NPUSortV2OP.forward
    torch_npu.npu_layer_norm_eval = NPULayerNormEvalOP.forward
    torch_npu.npu_reshape = NPUReshapeOP.forward
    torch_npu.npu_pad = NPUPadOP.forward
    torch_npu.npu_convolution = NPUConvolutionOP.forward
    torch_npu.npu_convolution_transpose = NPUConvolutionTransposeOP.forward
    torch_npu.npu_confusion_transpose = NPUConfusionTransposeOP.forward
    torch_npu.npu_max = NPUMaxOP.forward
    torch_npu.npu_bmmV2 = NPUBmmV2OP.forward
    torch_npu.npu_dtype_cast = NPUDtypeCastOP.forward
    torch_npu.npu_silu = NPUSiluOP.forward
    torch_npu.npu_min = NPUMinOP.forward
    torch_npu.npu_fused_attention_layernorm_qkv_fwd = NPUFusedAttentionLayernormQkvFwdOP.forward
