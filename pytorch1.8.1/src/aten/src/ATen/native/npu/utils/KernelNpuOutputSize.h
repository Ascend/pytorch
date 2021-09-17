// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION. 
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __NATIVE_NPU_UTILS_KERNEL_NPU_OUTPUT_SIZE__
#define __NATIVE_NPU_UTILS_KERNEL_NPU_OUTPUT_SIZE__

#include <stdint.h>
#include <bitset>
#include <string>
#include <tuple>
#include <vector>
#include "ATen/ATen.h"

using std::bitset;
using std::string;
using std::tuple;
using std::vector;

namespace at {
namespace native {
namespace npu {

const int N_SIZE = 32;
// npu tensor max size
const int SIZE = 8;
SmallVector<int64_t, SIZE> glu_npu_output_size(const Tensor& self, int64_t dim);

int64_t CeilDiv(int64_t value, int64_t factor);

int64_t make_wrap_dim(int64_t dim, int64_t dim_post_expr);

bitset<64> make_dim_mask(IntArrayRef dims, int64_t ndim);

SmallVector<int64_t, SIZE> array_to_small_vector(IntArrayRef shape);

IntArrayRef input_same_output_size(const Tensor& input);

SmallVector<int64_t, SIZE> broadcast_ops_npu_output_size(
    IntArrayRef shape1_,
    IntArrayRef shape2_);

SmallVector<int64_t, SIZE> broadcast_ops_npu_output_size(
    const Tensor& self,
    const Tensor& other);

SmallVector<int64_t, SIZE> reduce_ops_npu_output_size(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim);

SmallVector<int64_t, SIZE> adaptive_avg_pool3d_npu_output_size(
    const Tensor& self,
    IntArrayRef output_size);

SmallVector<int64_t, SIZE> addmm_npu_output_size(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha);

SmallVector<int64_t, SIZE> addbmm_npu_output_size(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha);

SmallVector<int64_t, SIZE> addmv_npu_output_size(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    Scalar beta,
    Scalar alpha);

SmallVector<int64_t, SIZE> addr_npu_output_size(
    const Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    Scalar beta,
    Scalar alpha);

SmallVector<int64_t, SIZE> avg_pool2d_npu_output_size(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);

SmallVector<int64_t, SIZE> baddbmm_npu_output_size(
    const Tensor& self,
    const Tensor& mat2);

SmallVector<int64_t, SIZE> cdist_npu_output_size(
    const Tensor& x1,
    const Tensor& x2);

tuple<IntArrayRef, IntArrayRef, SmallVector<int64_t, SIZE>>
conv2d_backward_npu_output_size(
    const Tensor& input,
    const Tensor& grad,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups);

SmallVector<int64_t, SIZE> cosine_similarity_npu_output_size(
	const Tensor& x1,
	int64_t dim,
	bool keepdim
);

tuple<IntArrayRef, IntArrayRef, SmallVector<int64_t, SIZE>> 
conv_transpose2d_backward_npu_output_size(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups);

SmallVector<int64_t, SIZE> conv_transpose2d_npu_output_size(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups);

SmallVector<int64_t, SIZE> deformable_conv2d_npu_output_size(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& offset,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t deformable_groups,
    bool modulated);

SmallVector<int64_t, SIZE> det_npu_output_size(const Tensor& self);

tuple<SmallVector<int64_t, SIZE>, SmallVector<int64_t, SIZE>>
ctc_loss_npu_output_size(
    const Tensor& logProbs,
    const Tensor& targets,
    IntArrayRef targetLengths);

SmallVector<int64_t, SIZE> dot_npu_output_size(const Tensor& self, const Tensor& other);

tuple<SmallVector<int64_t, SIZE>, SmallVector<int64_t, SIZE>>
nms_v4_npu_output_size(Scalar max_output_size);

SmallVector<int64_t, SIZE> equal_npu_output_size(void);

SmallVector<int64_t, SIZE> embedding_dense_backward_npu_output_size(
    const Tensor& grad_output, 
    const Tensor& indices, 
    int64_t num_weights, 
    int64_t padding_idx, 
    bool scale_grad_by_freq);

SmallVector<int64_t, SIZE> index_npu_output_size(
  const Tensor& self, 
  TensorList indices);

SmallVector<int64_t, SIZE> index_select_npu_output_size(
    const Tensor& self,
    int64_t dim,
    const Tensor& index);

SmallVector<int64_t, SIZE> iou_npu_output_size(
    const Tensor& bboxes,
    const Tensor& gtboxes);

tuple<IntArrayRef, IntArrayRef, IntArrayRef>
layer_norm_backward_npu_output_size(
  const Tensor& dY,
  const Tensor& X,
  const Tensor& mean,
  const Tensor& rstd,
  const Tensor& gamma,
  int64_t M,
  int64_t N);

SmallVector<int64_t, SIZE> lstm_npu_output_size(
    const Tensor& input,
	const Tensor& weight,
	const Tensor& bias,
	const Tensor& h,
	const Tensor& c,
	bool has_biases,
	int64_t num_layers,
	double dropout,
	bool train,
	bool bidirectional,
	bool batch_first);

SmallVector<int64_t, SIZE> mm_npu_output_size(
    const Tensor& self,
    const Tensor& mat2);

SmallVector<int64_t, SIZE> nnpack_spatial_convolution_npu_output_size(
  const Tensor& input,
  const Tensor& weight,
  IntArrayRef padding,
  IntArrayRef stride);

tuple<
    SmallVector<int64_t, SIZE>,
    SmallVector<int64_t, SIZE>,
    SmallVector<int64_t, SIZE>>
nms_with_mask_npu_output_size(const Tensor& self);

SmallVector<int64_t, SIZE> nonzero_npu_output_size(const Tensor& self);

SmallVector<int64_t, SIZE> pad_npu_output_size(const Tensor& input, IntArrayRef paddings);

SmallVector<int64_t, SIZE> pdist_npu_output_size(const Tensor& self, float p);

SmallVector<int64_t, SIZE> prod_npu_output_size(const Tensor & self, int64_t dim, bool keepdim);

SmallVector<int64_t, SIZE> prod_npu_output_size(
    const Tensor& self,
    int64_t dim,
    bool keepdim);

SmallVector<int64_t, SIZE> prod_npu_output_size(
    const Tensor& self,
    bool keepdim);

SmallVector<int64_t, SIZE> quantized_max_pool2d_npu_output_size(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode);

SmallVector<int64_t, SIZE> range_npu_output_size(
    float start,
    float end,
    float step);

IntArrayRef renorm_npu_output_size(
    const Tensor& self,
    Scalar p, 
    int dim, 
    Scalar maxnorm);

SmallVector<int64_t, SIZE> repeat_interleave_npu_output_size(
    const Tensor& self,
    int64_t repeats,
    int64_t dim);

SmallVector<int64_t, SIZE> replication_pad2d_npu_output_size(const Tensor& self,IntArrayRef padding);

SmallVector<int64_t, SIZE> repeat_npu_output_size(
    const Tensor& self,
    IntArrayRef repeats);

SmallVector<int64_t, SIZE> soft_margin_loss_npu_output_size(
    const Tensor &self,
    const Tensor &target, 
    int64_t reduction
);

SmallVector<int64_t, SIZE> slow_conv_dilated2d_npu_output_size(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

tuple<IntArrayRef, IntArrayRef, IntArrayRef> slow_conv_dilated2d_backward_npu_output_size(
    const Tensor& grad_output, 
    const Tensor& self, 
    const Tensor& weight, 
    IntArrayRef kernel_size, 
    IntArrayRef stride, 
    IntArrayRef padding, 
    IntArrayRef dilation);
    
tuple<IntArrayRef, IntArrayRef,IntArrayRef> slow_conv_transpose2d_backward_npu_output_size(
    const Tensor& grad_output,
    const Tensor& self,  
    const Tensor& weight, 
    IntArrayRef kernel_size, 
    IntArrayRef stride, 
    IntArrayRef padding, 
    IntArrayRef output_padding,
    IntArrayRef dilation, 
    const Tensor& columns,
    const Tensor& ones);

IntArrayRef smooth_l1_loss_npu_output_size(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction);

SmallVector<int64_t, SIZE> transpose_npu_output_size(
    const Tensor& self,
    IntArrayRef perm);

tuple<SmallVector<int64_t, SIZE>, SmallVector<int64_t, SIZE>>
softmax_cross_entropy_with_logits_impl_npu_output_size(const Tensor& self);

SmallVector<int64_t, SIZE> sum_npu_output_size(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim);

SmallVector<int64_t, SIZE> topk_npu_output_size(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted);

SmallVector<int64_t, SIZE> trace_npu_output_size(const Tensor& self);

IntArrayRef upsample_bicubic2d_backward_npu_output_size(IntArrayRef input_size);

SmallVector<int64_t, SIZE> upsample_bilinear2d_npu_output_size(
    const Tensor& self,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

IntArrayRef upsample_bilinear2d_backward_npu_output_size(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

SmallVector<int64_t, SIZE> upsample_linear1d_npu_output_size(
    const Tensor& self,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales);

SmallVector<int64_t, SIZE> var_npu_output_size(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim);

} // namespace npu
} // namespace native
} // namespace at

#endif
