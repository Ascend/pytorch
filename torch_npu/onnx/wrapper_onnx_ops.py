from typing import Optional, List

import torch
from torch import Tensor

import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error

__all__ = []


class _NPUOneHotOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_one_hot(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: torch.Tensor, num_classes: int = -1, depth: int = 1,
                 on_value: int = 1, off_value: int = 0):
        return g.op("npu::NPUOneHot", self, num_classes_i=num_classes, depth_i=depth,
                    on_value_i=on_value, off_value_i=off_value)


class _NPUSliceOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_slice(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: torch.Tensor, offsets: List[int], size: List[int]):
        return g.op("npu::NPUSlice", self, offsetss_i=offsets, sizes_i=size)


class _NPURoiAlignOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_roi_align(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: torch.Tensor, rois: torch.Tensor, spatial_scale: float,
                 pooled_height: int, pooled_width: int, sample_num: int, roi_end_mode: int):
        return g.op("npu::NPURoiAlign", self, rois, spatial_scale_f=spatial_scale,
                    pooled_height_i=pooled_height, pooled_width_i=pooled_width,
                    sample_num_i=sample_num, roi_end_mode_i=roi_end_mode)


class _NPUIouOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_iou(*args, **kwargs)

    @staticmethod
    def symbolic(g, bboxes: torch.Tensor, gtboxes: torch.Tensor, mode: int = 0):
        return g.op("npu::NPUIou", bboxes, gtboxes, mode_i=mode)


class _NPUBatchNmsOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_batch_nms(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: torch.Tensor, scores: torch.Tensor, score_threshold: float,
                 iou_threshold: float, max_size_per_class: int, max_total_size: int,
                 change_coordinate_frame: bool = False, transpose_box: bool = False):
        return g.op("npu::NPUBatchNms", self, scores, score_threshold_f=score_threshold,
                    iou_threshold_f=iou_threshold, max_size_per_class_i=max_size_per_class,
                    max_total_size_i=max_total_size, change_coordinate_frame_i=change_coordinate_frame,
                    transpose_box_i=transpose_box, outputs=4)


class _NPUFastGeluOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.fast_gelu(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: torch.Tensor):
        return g.op("npu::NPUFastGelu", self)


class _NPUGeGluOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_geglu(*args, **kwargs)
    
    @staticmethod
    def symbolic(g, self: torch.Tensor, dim: int = -1, approximate: int = 1, activate_left: bool = False):
        return g.op("npu::NPUGeGlu", self, dim_i=dim, approximate_i=approximate, 
                    activate_left_i=activate_left, outputs=2)


class _NPUFusedAttentionScoreOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_fused_attention_score(*args, **kwargs)

    @staticmethod
    def symbolic(g, query_layer: Tensor, key_layer: Tensor, value_layer: Tensor, attention_mask: Tensor,
                 scale: float, keep_prob: float, query_transpose: bool = False, key_transpose: bool = False,
                 bmm_score_transpose_a: bool = False, bmm_score_transpose_b: bool = False, value_transpose:
                 bool = False, dx_transpose: bool = False):
        return g.op("npu::NPUFusedAttentionScore", query_layer, key_layer, value_layer, attention_mask,
                    keep_prob_f=keep_prob, scale_f=scale, query_transpose_i=query_transpose,
                    key_transpose_i=key_transpose, bmm_score_transpose_a_i=bmm_score_transpose_a,
                    bmm_score_transpose_b_i=bmm_score_transpose_b, value_transpose_i=value_transpose,
                    dx_transpose_i=dx_transpose)


class _NPUCiouOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_ciou(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, gtboxes: Tensor, trans: bool = False, is_cross: bool = True,
                 mode: int = 0, atan_sub_flag: bool = False):
        return g.op("npu::NPUCiou", self, gtboxes, trans_i=trans, is_cross_i=is_cross, mode_i=mode,
                    atan_sub_flag_i=atan_sub_flag)


class _NPUGroupNormSiluOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch_npu._C._VariableFunctionsClass.npu_group_norm_silu(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, gamma: Optional[Tensor], beta: Optional[Tensor],
                 group: int, eps: float = 0.00001):
        if gamma is None:
            gamma = g.op("Constant", value_t=torch.tensor([]).to(torch.float))
        if beta is None:
            beta = g.op("Constant", value_t=torch.tensor([]).to(torch.float))
        return g.op("npu::NPUGroupNormSilu", self, gamma, beta, group_i=group, eps_f=eps,
                    outputs=3)


class _NPUMultiHeadAttentionOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return tuple(torch.ops.npu.npu_multi_head_attention(*args, **kwargs))

    @staticmethod
    def symbolic(g, query: Tensor, key: Tensor, value: Tensor, query_weight: Tensor, key_weight: Tensor,
                 value_weight: Tensor, attn_mask: Tensor, out_proj_weight: Tensor, query_bias: Tensor,
                 key_bias: Tensor, value_bias: Tensor, out_proj_bias: Tensor, dropout_mask: Tensor,
                 attn_head_num: int, attn_dim_per_head: int, src_len: int, tgt_len: int, dropout_prob: float,
                 softmax_use_float: bool):
        dtype = torch.float
        if query_bias is None:
            query_bias = g.op("Constant", value_t=torch.tensor([]).to(dtype))
        if key_bias is None:
            key_bias = g.op("Constant", value_t=torch.tensor([]).to(dtype))
        if value_bias is None:
            value_bias = g.op("Constant", value_t=torch.tensor([]).to(dtype))
        if out_proj_bias is None:
            out_proj_bias = g.op(
                "Constant", value_t=torch.tensor([]).to(dtype))
        return g.op("npu::NPUMultiHeadAttention", query, key, value, query_weight, key_weight, value_weight,
                    attn_mask, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, dropout_mask,
                    attn_head_num_i=attn_head_num, attn_dim_per_head_i=attn_dim_per_head, src_len_i=src_len,
                    tgt_len_i=tgt_len, dropout_prob_f=dropout_prob, softmax_use_float_i=softmax_use_float,
                    outputs=8)


class _NPUDeepNormOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_deep_norm(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, gx: Tensor, beta: Tensor, gamma: Tensor, alpha: float = 0.3, epsilon: float = 1e-6):
        return g.op("npu::NPUDeepNorm", self, gx, beta, gamma, alpha_f=alpha, epsilon_f=epsilon, outputs=3)


class _NPURmsNormOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_rms_norm(*args, **kwargs)
    
    @staticmethod
    def symbolic(g, self: Tensor, gamma: Tensor, epsilon: float = 1e-6):
        return g.op("npu::NPURmsNorm", self, gamma, epsilon_f=epsilon, outputs=2)


class _NPUAddRmsNormOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_add_rms_norm(*args, **kwargs)
    
    @staticmethod
    def symbolic(g, x1: Tensor, x2: Tensor, gamma: Tensor, epsilon: float = 1e-6):
        return g.op("npu::NPURmsNorm", x1, x2, gamma, epsilon_f=epsilon, outputs=3)


class _NPUDiouOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_diou(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, gtboxes: Tensor, trans: bool = False, is_cross: bool = False,
                 mode: int = 0):
        return g.op("npu::NPUDiou", self, gtboxes, trans_i=trans, is_cross_i=is_cross, mode_i=mode)


class _NPUGiouOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_giou(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, gtboxes: Tensor, trans: bool = False, is_cross: bool = False,
                 mode: int = 0):
        return g.op("npu::NPUGiou", self, gtboxes, trans_i=trans, is_cross_i=is_cross, mode_i=mode)


class _NPUDeformableConv2dOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_deformable_conv2d(*args, **kwargs)

    @staticmethod
    def symbolic(g, inputs: Tensor, weight: Tensor, offset: Tensor, bias: Optional[Tensor], kernel_size: List[int],
                 stride: List[int], padding: List[int], dilation: List[int] = [1, 1, 1, 1], groups: int = 1,
                 deformable_groups: int = 1, modulated: bool = True):
        if bias is None:
            bias = g.op("Constant", value_t=torch.tensor([]).to(torch.float))
        return g.op("npu::NPUDeformableConv2d", inputs, weight, offset, bias, kernel_sizes_i=kernel_size,
                    strides_i=stride, paddings_i=padding, dilations_i=dilation, groups_i=groups,
                    deformable_groups_i=deformable_groups, modulated_i=modulated, outputs=2)


class _NPUFormatCastOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_format_cast(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, acl_format: int):
        return g.op("npu::NPUFormatCast", self, acl_format_i=acl_format)


class _NPUSoftmaxCrossEntropyWithLogitsOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_softmax_cross_entropy_with_logits(
            *args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, labels: Tensor):
        return g.op("npu::NPUSoftmaxCrossEntropyWithLogits", self, labels)


class _NPUPsRoiPoolingOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_ps_roi_pooling(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, rois: Tensor, spatial_scale: float, group_size: int,
                 output_dim: int):
        return g.op("npu::NPUPsRoiPooling", self, rois, spatial_scale_f=spatial_scale,
                    group_size_i=group_size, output_dim_i=output_dim)


class _NPUGridAssignPositiveOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_grid_assign_positive(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, overlaps: Tensor, box_responsible_flags: Tensor,
                 max_overlaps: Tensor, argmax_overlaps: Tensor, gt_max_overlaps: Tensor,
                 gt_argmax_overlaps: Tensor, num_gts: int, pos_iou_thr: float,
                 min_pos_iou: float, gt_max_assign_all: bool):
        return g.op("npu::NPUGridAssignPositive", self, overlaps, box_responsible_flags,
                    max_overlaps, argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps,
                    num_gts_i=num_gts, pos_iou_thr_f=pos_iou_thr, min_pos_iou_f=min_pos_iou,
                    gt_max_assign_all_i=gt_max_assign_all)


class _NPUIfmrOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_ifmr(*args, **kwargs)

    @staticmethod
    def symbolic(g, data: Tensor, data_min: Tensor, data_max: Tensor, cumsum: Tensor,
                 min_percentile: float, max_percentile: float, search_start: float,
                 search_end: float, search_step: float, with_offset: bool):
        return g.op("npu::NPUIfmr", data, data_min, data_max, cumsum, min_percentile_f=min_percentile,
                    max_percentile_f=max_percentile, search_start_f=search_start, search_end_f=search_end,
                    search_step_f=search_step, with_offset_i=with_offset, outputs=2)


class _NPUFusedAttentionScoreFwdOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_fused_attention_score_fwd(*args, **kwargs)

    @staticmethod
    def symbolic(g, query_layer: Tensor, key_layer: Tensor, value_layer: Tensor, attention_mask: Tensor,
                 scale: float, keep_prob: float, query_transpose: bool = False, key_transpose: bool = False,
                 bmm_score_transpose_a: bool = False, bmm_score_transpose_b: bool = False,
                 value_transpose: bool = False, dx_transpose: bool = False):
        return g.op("npu::NPUFusedAttentionScoreFwd", query_layer, key_layer, value_layer, attention_mask,
                    scale_f=scale, keep_prob_f=keep_prob, query_transpose_i=query_transpose,
                    key_transpose_i=key_transpose, bmm_score_transpose_a_i=bmm_score_transpose_a,
                    bmm_score_transpose_b_i=bmm_score_transpose_b, value_transpose_i=value_transpose,
                    dx_transpose_i=dx_transpose, outputs=3)


class _NPUSignBitsUnpackOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_sign_bits_unpack(*args, **kwargs)

    @staticmethod
    def symbolic(g, inputs: Tensor, size: int, dtype: torch.dtype):
        if dtype == torch.float32:
            dtype = 0
        elif dtype == torch.float16:
            dtype = 1
        else:
            raise TypeError("The argument 'dtype' must be torch.float32 or torch.float16" + pta_error(ErrCode.TYPE))
        return g.op("npu::NPUSignBitsUnpack", inputs, size_i=size, dtype_i=dtype)


class _NPUPtiouOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_ptiou(*args, **kwargs)

    @staticmethod
    def symbolic(g, bboxes: Tensor, gtboxes: Tensor, mode: int = 0):
        return g.op("npu::NPUIou", bboxes, gtboxes, mode_i=mode)


class _NPUNormalizeBatchOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_normalize_batch(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, seq_len: Tensor, normalize_type: int = 0):
        return g.op("npu::NPUNormalizeBatch", self, seq_len, normalize_type_i=normalize_type)


class _NPUNmsV4OP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_nms_v4(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, scores: Tensor, max_output_size: int, iou_threshold: Tensor,
                 scores_threshold: Tensor, pad_to_max_output_size: bool = False):
        return g.op("npu::NPUNmsV4", self, scores, iou_threshold, scores_threshold, max_output_size_i=max_output_size,
                    pad_to_max_output_size_i=pad_to_max_output_size, outputs=2)


class _NPUBoundingBoxDecodeOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_bounding_box_decode(*args, **kwargs)

    @staticmethod
    def symbolic(g, rois: Tensor, deltas: Tensor, means0: float, means1: float, means2: float,
                 means3: float, stds0: float, stds1: float, stds2: float, stds3: float,
                 max_shape: List[int], wh_ratio_clip: float):
        return g.op("npu::NPUBoundingBoxDecode", rois, deltas, means0_f=means0, means1_f=means1,
                    means2_f=means2, means3_f=means3, stds0_f=stds0, stds1_f=stds1, stds2_f=stds2,
                    stds3_f=stds3, max_shapes_i=max_shape, wh_ratio_clip_f=wh_ratio_clip)


class _NPUBoundingBoxEncodeOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_bounding_box_encode(*args, **kwargs)

    @staticmethod
    def symbolic(g, anchor_box: Tensor, ground_truth_box: Tensor, means0: float, means1: float,
                 means2: float, means3: float, stds0: float, stds1: float, stds2: float, stds3: float):
        return g.op("npu::NPUBoundingBoxEncode", anchor_box, ground_truth_box, means0_f=means0,
                    means1_f=means1, means2_f=means2, means3_f=means3, stds0_f=stds0, stds1_f=stds1,
                    stds2_f=stds2, stds3_f=stds3)


class _NPUNmsWithMaskOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_nms_with_mask(*args, **kwargs)

    @staticmethod
    def symbolic(g, inputs: Tensor, iou_threshold: float):
        return g.op("npu::NPUNmsWithMask", inputs, iou_threshold_f=iou_threshold, outputs=3)


class _NPURotatedIouOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_rotated_iou(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, query_boxes: Tensor, trans: bool = False, mode: int = 0,
                 is_cross: bool = True, v_threshold: float = 0.0, e_threshold: float = 0.0):
        return g.op("npu::NPURotatedIou", self, query_boxes, trans_i=trans, mode_i=mode,
                    is_cross_i=is_cross, v_threshold_f=v_threshold, e_threshold_f=e_threshold)


class _NPURotatedOverlapsOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_rotated_overlaps(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, query_boxes: Tensor, trans: bool = False):
        return g.op("npu::NPURotatedOverlaps", self, query_boxes, trans_i=trans)


class _NPURotatedBoxDecodeOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_rotated_box_decode(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, deltas: Tensor, weight: Tensor):
        return g.op("npu::NPURotatedBoxDecode", self, deltas, weight)


class _NPURotatedBoxEncodeOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_rotated_box_encode(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, gt_bboxes: Tensor, weight: Tensor):
        return g.op("npu::NPURotatedBoxEncode", self, gt_bboxes, weight)


class _NPUYoloBoxesEncodeOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_yolo_boxes_encode(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, gt_bboxes: Tensor, stride: Tensor, performance_mode: bool = False):
        return g.op("npu::NPUYoloBoxesEncode", self, gt_bboxes, stride,
                    performance_mode_i=performance_mode)


class _NPUMaskedFillRangeOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_masked_fill_range(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, start: Tensor, end: Tensor, value: Tensor, axis: int = -1):
        return g.op("npu::NPUMaskedFillRange", self, start, end, value, axis_i=axis)


class _NPUAnchorResponseFlagsOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_anchor_response_flags(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, featmap_size: List[int], stride: List[int], num_base_anchors: int):
        return g.op("npu::NPUAnchorResponseFlags", self, featmap_sizes_i=featmap_size,
                    strides_i=stride, num_base_anchors_i=num_base_anchors)


class _NPUIndexingOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_indexing(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, begin: List[int], end: List[int], stride: List[int],
                 begin_mask: int = 0, end_mask: int = 0, ellipsis_mask: int = 0,
                 new_axis_mask: int = 0, shrink_axis_mask: int = 0):
        return g.op("npu::NPUIndexing", self, begins_i=begin, ends_i=end,
                    strides_i=stride, begin_mask_i=begin_mask, end_mask_i=end_mask,
                    ellipsis_mask_i=ellipsis_mask, new_axis_mask_i=new_axis_mask,
                    shrink_axis_mask_i=shrink_axis_mask)


class _NPUSignBitsPackOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_sign_bits_pack(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, size: int):
        return g.op("npu::NPUSignBitsPack", self, size_i=size)


class _NPUStrideAddOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_stride_add(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, other: Tensor, offset1: float, offset2: float, c1_len: int):
        return g.op("npu::NPUStrideAdd", self, other, offset1_f=offset1, offset2_f=offset2,
                    c1_len_i=c1_len)


class _NPUScatterOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_scatter(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, indices: Tensor, updates: Tensor, dim: int):
        return g.op("npu::NPUScatter", self, indices, updates, dim_i=dim)


class _NPUScatterNdUpdateOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_scatter_nd_update(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, indices: Tensor, updates: Tensor):
        return g.op("npu::NPUScatterNdUpdate", self, indices, updates)


class _NPULstmOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return tuple(torch.ops.npu.npu_lstm(*args, **kwargs))

    @staticmethod
    def symbolic(g, inputs: Tensor, weight: Tensor, bias: Tensor, seqMask: Tensor, h: Tensor,
                 c: Tensor, has_biases: bool, num_layers: int, dropout: float, train: bool,
                 bidirectional: bool, batch_first: bool, flagSeq: bool, direction: bool):
        if train:
            raise ValueError("Value of param 'train' must be False." + pta_error(ErrCode.VALUE))
        return g.op("npu::NPULstm", inputs, weight, bias, seqMask, h, c, has_biases_i=has_biases,
                    num_layers_i=num_layers, dropout_f=dropout, train_i=train, bidirectional_i=bidirectional,
                    batch_first_i=batch_first, flagSeq_i=flagSeq, direction_i=direction, outputs=8)


class _NPULstmCellOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return tuple(torch.ops.npu.npu_lstm_cell(*args, **kwargs))

    @staticmethod
    def symbolic(g, inputs: Tensor, w_ih: Tensor, w_hh: Tensor, h: Tensor, c: Tensor,
                 b_ih: Tensor = None, b_hh: Tensor = None):
        dtype = torch.float
        if b_ih is None:
            b_ih = g.op("Constant", value_t=torch.tensor([]).to(dtype))
        if b_hh is None:
            b_hh = g.op("Constant", value_t=torch.tensor([]).to(dtype))
        return g.op("npu::NPULstmCell", inputs, w_ih, w_hh, h, c, b_ih, b_hh, outputs=8)


class _NPUGruOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return tuple(torch.ops.npu.npu_gru(*args, **kwargs))

    @staticmethod
    def symbolic(g, inputs: Tensor, hx: Tensor, weight_input: Tensor, weight_hidden: Tensor,
                 bias_input: Tensor, bias_hidden: Tensor, seq_length: Tensor, has_biases: bool,
                 num_layers: int, dropout: float, train: bool, bidirectional: bool,
                 batch_first: bool):
        return g.op("npu::NPUGru", inputs, hx, weight_input, weight_hidden, bias_input,
                    bias_hidden, seq_length, has_biases_i=has_biases, num_layers_i=num_layers,
                    dropout_f=dropout, train_i=train, bidirectional_i=bidirectional,
                    batch_first_i=batch_first, outputs=6)


class _NPUDropoutWithAddSoftmaxOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_dropout_with_add_softmax(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor, x1: Tensor, alpha: float, prob: float, dim: int):
        return g.op("npu::NPUDropoutWithAddSoftmax", self, x1, alpha_f=alpha, prob_f=prob,
                    dim_i=dim, outputs=3)


class _NPUScaledMaskedSoftmaxOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_scaled_masked_softmax(*args, **kwargs)

    @staticmethod
    def symbolic(g, x: Tensor, mask: Tensor, scale: float = 1, fixed_triu_mask: bool = False):
        return g.op("npu::NPUScaledMaskedSoftmax", x, mask, scale_f=scale, fixed_triu_mask_i=fixed_triu_mask)


class _NPUMishOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_mish(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: Tensor):
        return g.op("npu::NPUMish", self)


class _NPURotaryMulOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_rotary_mul(*args, **kwargs)

    @staticmethod
    def symbolic(g, x: Tensor, r1: Tensor, r2: Tensor):
        return g.op("npu::NPURotaryMul", x, r1, r2)


class _NPUPromptFlashAttentionOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.prompt_flash_attention(*args, **kwargs)

    @staticmethod
    def symbolic(g, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                 padding_mask: Optional[Tensor], atten_mask: Optional[Tensor], pse_shift: Optional[Tensor],
                 actual_seq_lengths: Optional[Tensor], num_heads: int = 1,
                 scale_value: float = 1.0, pre_tokens: int = 2147473647, next_tokens: int = 0,
                 input_layout: str = "BSH", num_key_value_heads: int = 0):
        return g.op("npu::NPUPromptFlashAttention", self, query, key, value,
                    pse_shift, atten_mask, actual_seq_lengths,
                    num_heads, scale_value, pre_tokens, next_tokens,
                    input_layout, num_key_value_heads)


class _NPUIncreFlashAttentionOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.incre_flash_attention(*args, **kwargs)

    @staticmethod
    def symbolic(g, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                 padding_mask: Optional[Tensor], atten_mask: Optional[Tensor], 
                 pse_shift: Optional[Tensor],
                 actual_seq_lengths: Optional[Tensor],
                 num_heads: int = 1, scale_value: float = 1.0,
                 input_layout: str = "BSH", num_key_value_heads: int = 0):
        return g.op("npu::NPUIncreFlashAttention", self, query, key, value,
                    pse_shift, atten_mask, actual_seq_lengths,
                    num_heads, scale_value, input_layout, num_key_value_heads)


class _NPUMaskedSoftmaxWithRelPosBiasOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_masked_softmax_with_rel_pos_bias(*args, **kwargs)

    @staticmethod
    def symbolic(g, x: Tensor, atten_mask: Optional[Tensor], relative_pos_bias: Tensor, scale_value: float = 1.0,
                 inner_precision_mode: int = 0):
        return g.op("npu::NPUMaskedSoftmaxWithRelPosBias", x, atten_mask, relative_pos_bias, scale_value_f=scale_value,
                    inner_precision_mode_i=inner_precision_mode)
    

class _NPUMmAllReduceBaseOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_mm_all_reduce_base(*args, **kwargs)

    @staticmethod
    def symbolic(g, x1: torch.Tensor, x2: torch.Tensor, hcom: str,
                 reduce_op: str = 'sum', bias: Optional[Tensor] = None, antiquant_scale: Optional[Tensor] = None,
                 antiquant_offset: Optional[Tensor] = None, x3: Optional[Tensor] = None,
                 dequant_scale: Optional[Tensor] = None, pertoken_scale: Optional[Tensor] = None,
                 comm_quant_scale_1: Optional[Tensor] = None, comm_quant_scale_2: Optional[Tensor] = None,
                 antiquant_group_size: int = 0, comm_turn: int = 0):
        return g.op("npu::NPUMmAllReduceBase", x1, x2, hcom, reduce_op, bias, antiquant_scale, antiquant_offset, x3,
                    dequant_scale, pertoken_scale, comm_quant_scale_1, comm_quant_scale_2, antiquant_group_size, comm_turn)


class _NPUDynamicQuantOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_dummy, smooth_scales):
        return torch.ops.npu.npu_dynamic_quant(input_dummy, smooth_scales=smooth_scales)

    @staticmethod
    def symbolic(g, input_dummy: Tensor, smooth_scales: Optional[Tensor] = None):
        if smooth_scales is None:
            smooth_scales = g.op("Constant", value_t=torch.tensor([]).to(input_dummy.type().dtype()))
        return g.op("npu::NPUDynamicQuant", input_dummy, smooth_scales, outputs=2)


class _NPUDynamicQuantV2Op(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_dummy, smooth_scales, group_index, dst_type):
        return torch.ops.npu.npu_dynamic_quant_asymmetric(input_dummy, smooth_scales=smooth_scales,
                                                            group_index=group_index, dst_type=dst_type)

    @staticmethod
    def symbolic(g, input_dummy: Tensor, smooth_scales: Optional[Tensor] = None,
                 group_index: Optional[Tensor] = None, dst_type: torch.dtype = torch.int8):
        if smooth_scales is None:
            smooth_scales = g.op("Constant", value_t=torch.tensor([]).to(input_dummy.type().dtype()))
        if group_index is None:
            group_index = g.op("Constant", value_t=torch.tensor([]).to(torch.int32))
        dst_type_i = 2 # 当前仅支持int8
        return g.op("npu::NPUDynamicQuantV2", input_dummy, smooth_scales,
                    group_index, dst_type_i=dst_type_i, outputs=3)


class _NPUWeightQuantBatchMatmulOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_weight_quant_batchmatmul(*args, **kwargs)

    @staticmethod
    def symbolic(g, 
                 x: torch.Tensor, 
                 weight: torch.Tensor, 
                 antiquant_scale: torch.Tensor,
                 antiquant_offset: Optional[Tensor], 
                 quant_scale: Optional[Tensor],
                 quant_offset: Optional[Tensor],
                 bias: Optional[Tensor],
                 antiquant_group_size: int = 0):
        dtype = -1
        if antiquant_offset is None:
            antiquant_offset = g.op("Constant", value_t=torch.tensor([]).to(torch.float))
        if quant_scale is None:
            quant_scale = g.op("Constant", value_t=torch.tensor([]).to(torch.float))
            dtype = 1 # ge DataType of float16
        else:
            dtype = 2 # ge DataType of int8
        if quant_offset is None:
            quant_offset = g.op("Constant", value_t=torch.tensor([]).to(torch.float))
        if bias is None:
            bias = g.op("Constant", value_t=torch.tensor([]).to(torch.float))
        return g.op("npu::NPUWeightQuantBatchMatmulV2", 
                    x, 
                    weight, 
                    antiquant_scale, 
                    antiquant_offset, 
                    quant_scale, 
                    quant_offset, 
                    bias,
                    antiquant_group_size_i=antiquant_group_size,
                    dtype_i=dtype)
    

class _NPUAntiQuantOP(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, scale, offset, dst_dtype, src_dtype):
        return torch.ops.npu.npu_anti_quant(x, scale, offset=offset, dst_dtype=dst_dtype, src_dtype=src_dtype)

    @staticmethod
    def symbolic(g,
                 x: torch.Tensor,
                 scale: torch.Tensor,
                 offset: Optional[Tensor],
                 dst_dtype: Optional[int],
                 src_dtype: Optional[int]
                 ):
        if dst_dtype is None or dst_dtype == torch.float16:
            dst_dtype = 1
        elif dst_dtype == torch.bfloat16:
            dst_dtype = 27
        else:
            raise TypeError("The argument 'dst_dtype' must be torch.float16 or torch.bfloat16." +
                            pta_error(ErrCode.TYPE))
        
        if src_dtype is None or src_dtype == torch.int8:
            src_dtype = 2
        else:
            raise TypeError("The argument 'src_dtype' must be torch.int8." + pta_error(ErrCode.TYPE))
        
        return g.op("npu::NPUAntiQuant", x, scale, offset, dst_dtype_i=dst_dtype, src_dtype_i=src_dtype)

    
class _NPUQuantizeOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_quantize(*args, **kwargs)

    @staticmethod
    def symbolic(g,
                 inputs: torch.Tensor,
                 scales: torch.Tensor,
                 zero_points: torch.Tensor,
                 dtype: torch.dtype,
                 axis: int = 0,
                 div_mode: bool = True):
        acl_dtype = 2
        if dtype == torch.quint8:
            acl_dtype = 4
        elif dtype == torch.qint8:
            acl_dtype = 2
        elif dtype == torch.qint32:
            acl_dtype = 3
        else:
            raise ValueError("The argument 'dtype' must be torch.quint8, torch.qint8 or torch.qint32")
        return g.op("npu::NPUQuantize", inputs, scales, zero_points, dtype_i=acl_dtype, axis_i=axis, div_mode_i=div_mode)
    

class _NPUMoeFinalizeRoutingOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch.ops.npu.npu_moe_finalize_routing(*args, **kwargs)

    @staticmethod
    def symbolic(g, expanded_permuted_rows: Tensor, skip1: Tensor, skip2: Optional[Tensor], bias: Tensor,
                 scales: Tensor, expanded_src_to_dst_row: Tensor, export_for_source_row: Tensor):
        if skip2 is None:
            skip2 = g.op("Constant", value_t=torch.tensor([]).to(torch.float))
        return g.op("npu::NPUMoeFinalizeRouting", expanded_permuted_rows, skip1, skip2, bias,
                 scales, expanded_src_to_dst_row, export_for_source_row)


def _wrapper_npu_masked_softmax_with_rel_pos_bias(x, atten_mask, relative_pos_bias, scale_value=1.0, inner_precision_mode=0):
    return _NPUMaskedSoftmaxWithRelPosBiasOP.apply(x, atten_mask, relative_pos_bias, scale_value, inner_precision_mode)


def _wrapper_npu_one_hot(self, num_classes=-1, depth=1, on_value=1, off_value=0):
    return _NPUOneHotOP.apply(self, num_classes, depth, on_value, off_value)


def _wrapper_npu_slice(self, offsets, size):
    return _NPUSliceOP.apply(self, offsets, size)


def _wrapper_npu_roi_align(self, rois, spatial_scale, pooled_height, pooled_width,
                          sample_num, roi_end_mode):
    return _NPURoiAlignOP.apply(self, rois, spatial_scale, pooled_height, pooled_width,
                               sample_num, roi_end_mode)


def _wrapper_npu_iou(bboxes, gtboxes, mode=0):
    return _NPUIouOP.apply(bboxes, gtboxes, mode)


def _wrapper_npu_batch_nms(self, scores, score_threshold, iou_threshold,
                          max_size_per_class, max_total_size,
                          change_coordinate_frame=False, transpose_box=False):
    return _NPUBatchNmsOP.apply(self, scores, score_threshold,
                               iou_threshold, max_size_per_class, max_total_size,
                               change_coordinate_frame, transpose_box)


def _wrapper_npu_fast_gelu(self):
    return _NPUFastGeluOP.apply(self)


def _wrapper_npu_geglu(self, dim=-1, approximate=1, activate_left=False):
    return _NPUGeGluOP.apply(self, dim, approximate, activate_left)


def _wrapper_npu_fused_attention_score(query_layer, key_layer, value_layer, attention_mask,
                                      scale, keep_prob, query_transpose=False, key_transpose=False,
                                      bmm_score_transpose_a=False, bmm_score_transpose_b=False,
                                      value_transpose=False, dx_transpose=False):
    return _NPUFusedAttentionScoreOP.apply(query_layer, key_layer, value_layer, attention_mask,
                                          scale, keep_prob, query_transpose, key_transpose,
                                          bmm_score_transpose_a, bmm_score_transpose_b,
                                          value_transpose, dx_transpose)


def _wrapper_npu_ciou(self, gtboxes, trans=False, is_cross=True, mode=0, atan_sub_flag=False):
    return _NPUCiouOP.apply(self, gtboxes, trans, is_cross, mode, atan_sub_flag)


def _wrapper_npu_multi_head_attention(query, key, value, query_weight, key_weight, value_weight,
                                     attn_mask, out_proj_weight, query_bias, key_bias, value_bias,
                                     out_proj_bias, dropout_mask, attn_head_num, attn_dim_per_head,
                                     src_len, tgt_len, dropout_prob, softmax_use_float):
    return _NPUMultiHeadAttentionOP.apply(query, key, value, query_weight, key_weight, value_weight,
                                         attn_mask, out_proj_weight, query_bias, key_bias, value_bias,
                                         out_proj_bias, dropout_mask, attn_head_num, attn_dim_per_head,
                                         src_len, tgt_len, dropout_prob, softmax_use_float)


def _wrapper_npu_diou(self, gtboxes, trans=False, is_cross=False, mode=0):
    return _NPUDiouOP.apply(self, gtboxes, trans, is_cross, mode)


def _wrapper_npu_giou(self, gtboxes, trans=False, is_cross=False, mode=0):
    return _NPUGiouOP.apply(self, gtboxes, trans, is_cross, mode)


def _wrapper_npu_deformable_conv2d(inputs, weight, offset, bias, kernel_size, stride, padding,
                                  dilation=[1, 1, 1, 1], groups=1, deformable_groups=1, modulated=True):
    return _NPUDeformableConv2dOP.apply(inputs, weight, offset, bias, kernel_size, stride,
                                       padding, dilation, groups, deformable_groups, modulated)


def _wrapper_npu_format_cast(self, acl_format):
    return _NPUFormatCastOP.apply(self, acl_format)


def _wrapper_npu_softmax_cross_entropy_with_logits(self, labels):
    return _NPUSoftmaxCrossEntropyWithLogitsOP.apply(self, labels)


def _wrapper_npu_ps_roi_pooling(self, rois, spatial_scale, group_size, output_dim):
    return _NPUPsRoiPoolingOP.apply(self, rois, spatial_scale, group_size, output_dim)


def _wrapper_npu_grid_assign_positive(self, overlaps, box_responsible_flags, max_overlaps,
                                     argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps,
                                     num_gts, pos_iou_thr, min_pos_iou, gt_max_assign_all):
    return _NPUGridAssignPositiveOP.apply(self, overlaps, box_responsible_flags, max_overlaps,
                                         argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps,
                                         num_gts, pos_iou_thr, min_pos_iou, gt_max_assign_all)


def _wrapper_npu_deep_norm(self, gx, beta, gamma, alpha=0.3, epsilon=1e-6):
    return _NPUDeepNormOP.apply(self, gx, beta, gamma, alpha, epsilon)


def _wrapper_npu_group_norm_silu(x, gamma, beta, group, eps=0.00001):
    return _NPUGroupNormSiluOP.apply(x, gamma, beta, group, eps)


def _wrapper_npu_ifmr(data, data_min, data_max, cumsum, min_percentile, max_percentile,
                     search_start, search_end, search_step, with_offset):
    return _NPUIfmrOP.apply(data, data_min, data_max, cumsum, min_percentile, max_percentile,
                           search_start, search_end, search_step, with_offset)


def _wrapper_npu_fused_attention_score_fwd(query_layer, key_layer, value_layer, attention_mask,
                                          scale, keep_prob, query_transpose=False, key_transpose=False,
                                          bmm_score_transpose_a=False, bmm_score_transpose_b=False,
                                          value_transpose=False, dx_transpose=False):
    return _NPUFusedAttentionScoreFwdOP.apply(query_layer, key_layer, value_layer, attention_mask,
                                             scale, keep_prob, query_transpose, key_transpose,
                                             bmm_score_transpose_a, bmm_score_transpose_b,
                                             value_transpose, dx_transpose)


def _wrapper_npu_sign_bits_unpack(inputs, size, dtype):
    return _NPUSignBitsUnpackOP.apply(inputs, size, dtype)


def _wrapper_npu_ptiou(bboxes, gtboxes, mode=0):
    return _NPUPtiouOP.apply(bboxes, gtboxes, mode)


def _wrapper_npu_normalize_batch(self, seq_len, normalize_type=0):
    return _NPUNormalizeBatchOP.apply(self, seq_len, normalize_type)


def _wrapper_npu_rms_norm(self, gamma, epsilon=1e-6):
    return _NPURmsNormOP.apply(self, gamma, epsilon)


def _wrapper_npu_add_rms_norm(x1, x2, gamma, epsilon=1e-6):
    return _NPUAddRmsNormOP.apply(x1, x2, gamma, epsilon)


def _wrapper_npu_nms_v4(self, scores, max_output_size, iou_threshold, scores_threshold,
                       pad_to_max_output_size=False):
    return _NPUNmsV4OP.apply(self, scores, max_output_size,
                            iou_threshold, scores_threshold, pad_to_max_output_size)


def _wrapper_npu_bounding_box_decode(rois, deltas, means0, means1, means2, means3, stds0,
                                    stds1, stds2, stds3, max_shape, wh_ratio_clip):
    return _NPUBoundingBoxDecodeOP.apply(rois, deltas, means0, means1, means2, means3,
                                        stds0, stds1, stds2, stds3, max_shape, wh_ratio_clip)


def _wrapper_npu_bounding_box_encode(anchor_box, ground_truth_box, means0, means1, means2,
                                    means3, stds0, stds1, stds2, stds3):
    return _NPUBoundingBoxEncodeOP.apply(anchor_box, ground_truth_box, means0, means1,
                                        means2, means3, stds0, stds1, stds2, stds3)


def _wrapper_npu_nms_with_mask(inputs, iou_threshold):
    return _NPUNmsWithMaskOP.apply(inputs, iou_threshold)


def _wrapper_npu_rotated_iou(self, query_boxes, trans=False, mode=0, is_cross=True,
                            v_threshold=0.0, e_threshold=0.0):
    return _NPURotatedIouOP.apply(self, query_boxes, trans, mode, is_cross, v_threshold,
                                 e_threshold)


def _wrapper_npu_rotated_overlaps(self, query_boxes, trans=False):
    return _NPURotatedOverlapsOP.apply(self, query_boxes, trans)


def _wrapper_npu_rotated_box_decode(self, deltas, weight):
    return _NPURotatedBoxDecodeOP.apply(self, deltas, weight)


def _wrapper_npu_rotated_box_encode(self, gt_bboxes, weight):
    return _NPURotatedBoxEncodeOP.apply(self, gt_bboxes, weight)


def _wrapper_npu_yolo_boxes_encode(self, gt_bboxes, weight):
    return _NPUYoloBoxesEncodeOP.apply(self, gt_bboxes, weight)


def _wrapper_npu_masked_fill_range(self, start, end, value, axis=-1):
    return _NPUMaskedFillRangeOP.apply(self, start, end, value, axis)


def _wrapper_npu_anchor_response_flags(self, featmap_size, stride, num_base_anchors):
    return _NPUAnchorResponseFlagsOP.apply(self, featmap_size, stride, num_base_anchors)


def _wrapper_npu_indexing(self, begin, end, strides, begin_mask=0, end_mask=0,
                         ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
    return _NPUIndexingOP.apply(self, begin, end, strides,
                               begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)


def _wrapper_npu_sign_bits_pack(self, size):
    return _NPUSignBitsPackOP.apply(self, size)


def _wrapper_npu_lstm_cell(inputs, w_ih, w_hh, h, c, b_ih=None, b_hh=None):
    return _NPULstmCellOP.apply(inputs, w_ih, w_hh, h, c, b_ih, b_hh)


def _wrapper_npu_lstm(inputs, weight, bias, seqMask, h, c, has_biases, num_layers,
                     dropout, train, bidirectional, batch_first, flagSeq, direction):
    return _NPULstmOP.apply(inputs, weight, bias, seqMask, h, c, has_biases, num_layers,
                           dropout, train, bidirectional, batch_first, flagSeq, direction)


def _wrapper_npu_scatter(self, indices, updates, dim):
    return _NPUScatterOP.apply(self, indices, updates, dim)


def _wrapper_npu_scatter_nd_update(self, indices, updates):
    return _NPUScatterNdUpdateOP.apply(self, indices, updates)


def _wrapper_npu_stride_add(self, other, offset1, offset2, c1_len):
    return _NPUStrideAddOP.apply(self, other, offset1, offset2, c1_len)


def _wrapper_npu_dynamic_quant(input_dummy, smooth_scales=None):
    return _NPUDynamicQuantOp.apply(input_dummy, smooth_scales)


def _wrapper_npu_dynamic_quant_asymmetric(input_dummy, smooth_scales=None, group_index=None, dst_type=torch.int8):
    return _NPUDynamicQuantV2Op.apply(input_dummy, smooth_scales, group_index, dst_type)


def _wrapper_npu_gru(inputs, hx, weight_input, weight_hidden, bias_input, bias_hidden,
                    seq_length, has_biases, num_layers, dropout, train, bidirectional, batch_first):
    return _NPUGruOP.apply(inputs, hx, weight_input, weight_hidden, bias_input, bias_hidden,
                          seq_length, has_biases, num_layers, dropout, train, bidirectional, batch_first)


def _wrapper_npu_dropout_with_add_softmax(self, x1, alpha, prob, dim):
    return _NPUDropoutWithAddSoftmaxOP.apply(self, x1, alpha, prob, dim)


def _wrapper_npu_scaled_masked_softmax(x, mask, scale=1, fixed_triu_mask=False):
    return _NPUScaledMaskedSoftmaxOP.apply(x, mask, scale, fixed_triu_mask)


def _wrapper_npu_mish(self):
    return _NPUMishOP.apply(self)


def _wrapper_npu_rotary_mul(x, r1, r2):
    return _NPURotaryMulOP.apply(x, r1, r2)


def _wrapper_npu_prompt_flash_attention(self, query, key, value, padding_mask, atten_mask, pse_shift, actual_seq_lengths,
                                       num_heads, scale_value, pre_tokens, next_tokens, input_layout, num_key_value_heads):
    return _NPUPromptFlashAttentionOP.apply(self, query, key, value, padding_mask, atten_mask, pse_shift, actual_seq_lengths,
                                           num_heads, scale_value, pre_tokens, next_tokens, input_layout, num_key_value_heads)


def _wrapper_npu_incre_flash_attention(self, query, key, value, padding_mask, atten_mask, pse_shift, actual_seq_lengths,
                                      num_heads, scale_value, input_layout, num_key_value_heads):
    return _NPUIncreFlashAttentionOP.apply(self, query, key, value, padding_mask, atten_mask, pse_shift, actual_seq_lengths,
                                          num_heads, scale_value, input_layout, num_key_value_heads)


def _wrapper_npu_mm_all_reduce_base(x1, x2, hcom, reduce_op, bias, antiquant_scale, antiquant_offset, x3,
                                   dequant_scale, pertoken_scale, comm_quant_scale_1, comm_quant_scale_2,
                                   antiquant_group_size, comm_turn):
    return _NPUMmAllReduceBaseOP.apply(x1, x2, hcom, reduce_op, bias, antiquant_scale, antiquant_offset, x3,
                                      dequant_scale, pertoken_scale, comm_quant_scale_1, comm_quant_scale_2,
                                      antiquant_group_size, comm_turn)



def _wrapper_npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, 
                                           quant_scale, quant_offset, bias, antiquant_group_size):
    return _NPUWeightQuantBatchMatmulOP.apply(x, weight, antiquant_scale, antiquant_offset, 
                                               quant_scale, quant_offset, bias, antiquant_group_size)


def _wrapper_npu_anti_quant(x, scale, offset=None, dst_dtype=None, src_dtype=None):
    return _NPUAntiQuantOP.apply(x, scale, offset, dst_dtype, src_dtype)


def _wrapper_npu_quantize(inputs, scales, zero_points, dtype, axis, div_mode=True):
    return _NPUQuantizeOP.apply(inputs, scales, zero_points, dtype, axis, div_mode)


def _wrapper_npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2, bias,
                                      scales, expanded_src_to_dst_row, export_for_source_row):
    return _NPUMoeFinalizeRoutingOP.apply(expanded_permuted_rows, skip1, skip2, bias,
                                          scales, expanded_src_to_dst_row, export_for_source_row)


def _add_onnx_ops():
    torch_npu.npu_one_hot = _wrapper_npu_one_hot
    torch_npu.npu_slice = _wrapper_npu_slice
    torch_npu.npu_roi_align = _wrapper_npu_roi_align
    torch_npu.npu_group_norm_silu = _wrapper_npu_group_norm_silu
    torch_npu.npu_iou = _wrapper_npu_iou
    torch_npu.npu_batch_nms = _wrapper_npu_batch_nms
    torch_npu.fast_gelu = _wrapper_npu_fast_gelu
    torch_npu.npu_fast_gelu = _wrapper_npu_fast_gelu
    torch_npu.npu_geglu = _wrapper_npu_geglu
    torch_npu.npu_fused_attention_score = _wrapper_npu_fused_attention_score
    torch_npu.npu_ciou = _wrapper_npu_ciou
    torch_npu.npu_multi_head_attention = _wrapper_npu_multi_head_attention
    torch_npu.npu_diou = _wrapper_npu_diou
    torch_npu.npu_giou = _wrapper_npu_giou
    torch_npu.npu_deformable_conv2d = _wrapper_npu_deformable_conv2d
    torch_npu.npu_format_cast = _wrapper_npu_format_cast
    torch_npu.npu_softmax_cross_entropy_with_logits = _wrapper_npu_softmax_cross_entropy_with_logits
    torch_npu.npu_ps_roi_pooling = _wrapper_npu_ps_roi_pooling
    torch_npu.npu_grid_assign_positive = _wrapper_npu_grid_assign_positive
    torch_npu.npu_ifmr = _wrapper_npu_ifmr
    torch_npu.npu_fused_attention_score_fwd = _wrapper_npu_fused_attention_score_fwd
    torch_npu.npu_sign_bits_unpack = _wrapper_npu_sign_bits_unpack
    torch_npu.npu_ptiou = _wrapper_npu_ptiou
    torch_npu.npu_normalize_batch = _wrapper_npu_normalize_batch
    torch_npu.npu_nms_v4 = _wrapper_npu_nms_v4
    torch_npu.npu_bounding_box_decode = _wrapper_npu_bounding_box_decode
    torch_npu.npu_bounding_box_encode = _wrapper_npu_bounding_box_encode
    torch_npu.npu_nms_with_mask = _wrapper_npu_nms_with_mask
    torch_npu.npu_rotated_iou = _wrapper_npu_rotated_iou
    torch_npu.npu_rotated_overlaps = _wrapper_npu_rotated_overlaps
    torch_npu.npu_rotated_box_decode = _wrapper_npu_rotated_box_decode
    torch_npu.npu_rotated_box_encode = _wrapper_npu_rotated_box_encode
    torch_npu.npu_yolo_boxes_encode = _wrapper_npu_yolo_boxes_encode
    torch_npu.npu_masked_fill_range = _wrapper_npu_masked_fill_range
    torch_npu.npu_anchor_response_flags = _wrapper_npu_anchor_response_flags
    torch_npu.npu_indexing = _wrapper_npu_indexing
    torch_npu.npu_sign_bits_pack = _wrapper_npu_sign_bits_pack
    torch_npu.npu_stride_add = _wrapper_npu_stride_add
    torch_npu.npu_deep_norm = _wrapper_npu_deep_norm
    torch_npu.npu_scatter = _wrapper_npu_scatter
    torch_npu.npu_scatter_nd_update = _wrapper_npu_scatter_nd_update
    torch_npu.npu_lstm = _wrapper_npu_lstm
    torch_npu.npu_dynamic_quant = _wrapper_npu_dynamic_quant
    torch_npu.npu_dynamic_quant_asymmetric = _wrapper_npu_dynamic_quant_asymmetric
    torch_npu.npu_rms_norm = _wrapper_npu_rms_norm
    torch_npu.npu_add_rms_norm = _wrapper_npu_add_rms_norm
    torch_npu.npu_lstm_cell = _wrapper_npu_lstm_cell
    torch_npu.npu_gru = _wrapper_npu_gru
    torch_npu.npu_dropout_with_add_softmax = _wrapper_npu_dropout_with_add_softmax
    torch_npu.npu_scaled_masked_softmax = _wrapper_npu_scaled_masked_softmax
    torch_npu.npu_mish = _wrapper_npu_mish
    torch_npu.npu_rotary_mul = _wrapper_npu_rotary_mul
    torch_npu.npu_prompt_flash_attention = _wrapper_npu_prompt_flash_attention
    torch_npu.npu_incre_flash_attention = _wrapper_npu_incre_flash_attention
    torch_npu.npu_masked_softmax_with_rel_pos_bias = _wrapper_npu_masked_softmax_with_rel_pos_bias
    torch_npu.npu_mm_all_reduce_base = _wrapper_npu_mm_all_reduce_base
    torch_npu.npu_weight_quant_batchmatmul = _wrapper_npu_weight_quant_batchmatmul
    torch_npu.npu_anti_quant = _wrapper_npu_anti_quant
    torch_npu.npu_quantize = _wrapper_npu_quantize
    torch_npu.npu_moe_finalize_routing = _wrapper_npu_moe_finalize_routing
