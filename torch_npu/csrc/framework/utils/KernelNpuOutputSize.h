#ifndef __PLUGIN_NATIVE_NPU_UTILS_KERNEL_NPU_OUTPUT_SIZE__
#define __PLUGIN_NATIVE_NPU_UTILS_KERNEL_NPU_OUTPUT_SIZE__

#include <stdint.h>
#include <bitset>
#include <string>
#include <tuple>
#include <vector>
#include <ATen/ATen.h>

using std::bitset;
using std::string;
using std::tuple;
using std::vector;

namespace at_npu {
namespace native {

const int N_SIZE = 32;
// npu tensor max size
const int SIZE = 8;
c10::SmallVector<int64_t, SIZE> glu_npu_output_size(const at::Tensor& self, int64_t dim);

int64_t CeilDiv(int64_t value, int64_t factor);

int64_t make_wrap_dim(int64_t dim, int64_t dim_post_expr);

bitset<64> make_dim_mask(c10::IntArrayRef dims, int64_t ndim);

c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape);

c10::IntArrayRef input_same_output_size(const at::Tensor& input);

c10::SmallVector<int64_t, SIZE> broadcast_ops_npu_output_size(
    c10::IntArrayRef shape1_,
    c10::IntArrayRef shape2_);

c10::SmallVector<int64_t, SIZE> broadcast_ops_npu_output_size(
    const at::Tensor& self,
    const at::Tensor& other);

c10::SmallVector<int64_t, SIZE> reduce_ops_npu_output_size(
    const at::Tensor& self,
    c10::IntArrayRef dim,
    bool keepdim);

c10::SmallVector<int64_t, SIZE> adaptive_avg_pool3d_npu_output_size(
    const at::Tensor& self,
    c10::IntArrayRef output_size);

c10::SmallVector<int64_t, SIZE> addmm_npu_output_size(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    c10::Scalar beta,
    c10::Scalar alpha);

c10::SmallVector<int64_t, SIZE> addbmm_npu_output_size(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    c10::Scalar beta,
    c10::Scalar alpha);

c10::SmallVector<int64_t, SIZE> addmv_npu_output_size(
    const at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    c10::Scalar beta,
    c10::Scalar alpha);

c10::SmallVector<int64_t, SIZE> addr_npu_output_size(
    const at::Tensor& self,
    const at::Tensor& vec1,
    const at::Tensor& vec2,
    c10::Scalar beta,
    c10::Scalar alpha);

c10::SmallVector<int64_t, SIZE> avg_pool2d_npu_output_size(
    const at::Tensor& self,
    c10::IntArrayRef kernel_size,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);

c10::SmallVector<int64_t, SIZE> baddbmm_npu_output_size(
    const at::Tensor& self,
    const at::Tensor& mat2);

c10::SmallVector<int64_t, SIZE> cdist_npu_output_size(
    const at::Tensor& x1,
    const at::Tensor& x2);

tuple<c10::IntArrayRef, c10::IntArrayRef, c10::SmallVector<int64_t, SIZE>> conv2d_backward_npu_output_size(
    const at::Tensor& input,
    const at::Tensor& grad,
    const at::Tensor& weight,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation,
    int64_t groups);

c10::SmallVector<int64_t, SIZE> cosine_similarity_npu_output_size(
    const at::Tensor& x1,
    int64_t dim,
    bool keepdim);

tuple<c10::IntArrayRef, c10::IntArrayRef, c10::SmallVector<int64_t, SIZE>> conv_transpose2d_backward_npu_output_size(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    c10::IntArrayRef padding,
    c10::IntArrayRef output_padding,
    c10::IntArrayRef stride,
    c10::IntArrayRef dilation,
    int64_t groups);

c10::SmallVector<int64_t, SIZE> conv_transpose2d_npu_output_size(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    c10::IntArrayRef padding,
    c10::IntArrayRef output_padding,
    c10::IntArrayRef stride,
    c10::IntArrayRef dilation,
    int64_t groups);

c10::SmallVector<int64_t, SIZE> deformable_conv2d_npu_output_size(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& bias,
    c10::IntArrayRef kernel_size,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation,
    int64_t groups,
    int64_t deformable_groups,
    bool modulated);

c10::SmallVector<int64_t, SIZE> det_npu_output_size(const at::Tensor& self);

tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> ctc_loss_npu_output_size(
    const at::Tensor& logProbs,
    int64_t maxLength);

c10::SmallVector<int64_t, SIZE> dot_npu_output_size(const at::Tensor& self, const at::Tensor& other);

tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> nms_v4_npu_output_size(
    c10::Scalar max_output_size);

c10::SmallVector<int64_t, SIZE> equal_npu_output_size(void);

c10::SmallVector<int64_t, SIZE> embedding_dense_backward_npu_output_size(
    const at::Tensor& grad_output,
    const at::Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq);

c10::SmallVector<int64_t, SIZE> index_npu_output_size(
    const at::Tensor& self,
    at::TensorList indices);

c10::SmallVector<int64_t, SIZE> index_select_npu_output_size(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index);

c10::SmallVector<int64_t, SIZE> iou_npu_output_size(
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes);

tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef> layer_norm_backward_npu_output_size(
    const at::Tensor& dY,
    const at::Tensor& X,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& gamma,
    int64_t M,
    int64_t N);

c10::SmallVector<int64_t, SIZE> lstm_npu_output_size(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& h,
    const at::Tensor& c,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first);

c10::SmallVector<int64_t, SIZE> nnpack_spatial_convolution_npu_output_size(
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::IntArrayRef padding,
    c10::IntArrayRef stride);

tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> nms_with_mask_npu_output_size(
    const at::Tensor& self);

c10::SmallVector<int64_t, SIZE> nonzero_npu_output_size(const at::Tensor& self);

c10::SmallVector<int64_t, SIZE> nonzero_npu_max_output_size(const at::Tensor& self);

c10::SmallVector<int64_t, SIZE> pad_npu_output_size(const at::Tensor& input, c10::IntArrayRef paddings);

c10::SmallVector<int64_t, SIZE> pdist_npu_output_size(const at::Tensor& self, float p);

c10::SmallVector<int64_t, SIZE> prod_npu_output_size(const at::Tensor & self, int64_t dim, bool keepdim);

c10::SmallVector<int64_t, SIZE> prod_npu_output_size(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim);

c10::SmallVector<int64_t, SIZE> prod_npu_output_size(
    const at::Tensor& self,
    bool keepdim);

c10::SmallVector<int64_t, SIZE> quantized_max_pool2d_npu_output_size(
    const at::Tensor& self,
    c10::IntArrayRef kernel_size,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation,
    bool ceil_mode);

c10::SmallVector<int64_t, SIZE> range_npu_output_size(
    float start,
    float end,
    float step);

c10::IntArrayRef renorm_npu_output_size(
    const at::Tensor& self,
    c10::Scalar p,
    int dim,
    c10::Scalar maxnorm);

c10::SmallVector<int64_t, SIZE> repeat_interleave_npu_output_size(
    const at::Tensor& self,
    int64_t repeats,
    int64_t dim);

c10::SmallVector<int64_t, SIZE> repeat_interleave_npu_output_size(
    const at::Tensor& self,
    const at::Tensor& repeats,
    int64_t dim);

c10::SmallVector<int64_t, SIZE> replication_pad2d_npu_output_size(const at::Tensor& self,c10::IntArrayRef padding);

c10::SmallVector<int64_t, SIZE> roi_align_backward_npu_output_size(
    c10::IntArrayRef xdiff_shape);

c10::SmallVector<int64_t, SIZE> repeat_npu_output_size(
    const at::Tensor& self,
    c10::IntArrayRef repeats);

c10::SmallVector<int64_t, SIZE> soft_margin_loss_npu_output_size(
    const at::Tensor &self,
    const at::Tensor &target,
    int64_t reduction
);

c10::SmallVector<int64_t, SIZE> slow_conv_dilated2d_npu_output_size(
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation);

tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef> slow_conv_dilated2d_backward_npu_output_size(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    c10::IntArrayRef kernel_size,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation);

tuple<c10::IntArrayRef, c10::IntArrayRef,c10::IntArrayRef> slow_conv_transpose2d_backward_npu_output_size(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    c10::IntArrayRef kernel_size,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef output_padding,
    c10::IntArrayRef dilation);

c10::IntArrayRef smooth_l1_loss_npu_output_size(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction);

c10::SmallVector<int64_t, SIZE> transpose_npu_output_size(
    const at::Tensor& self,
    c10::IntArrayRef perm);

tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>softmax_cross_entropy_with_logits_impl_npu_output_size(
    const at::Tensor& self);

c10::SmallVector<int64_t, SIZE> sum_npu_output_size(
    const at::Tensor& self,
    c10::IntArrayRef dim,
    bool keepdim);

c10::SmallVector<int64_t, SIZE> topk_npu_output_size(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted);

c10::SmallVector<int64_t, SIZE> trace_npu_output_size(const at::Tensor& self);

c10::IntArrayRef upsample_bicubic2d_backward_npu_output_size(c10::IntArrayRef input_size);

c10::SmallVector<int64_t, SIZE> upsample_bilinear2d_npu_output_size(
    const at::Tensor& self,
    c10::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

c10::IntArrayRef upsample_bilinear2d_backward_npu_output_size(
    const at::Tensor& grad_output,
    c10::IntArrayRef output_size,
    c10::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

c10::SmallVector<int64_t, SIZE> upsample_linear1d_npu_output_size(
    const at::Tensor& self,
    c10::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales);

c10::SmallVector<int64_t, SIZE> var_npu_output_size(
    const at::Tensor& self,
    c10::IntArrayRef dim,
    bool keepdim);

c10::SmallVector<int64_t, SIZE> crop_and_resize_npu_output_size(
    const at::Tensor &self,
    at::IntArrayRef box_index,
    at::IntArrayRef crop_size);

c10::SmallVector<int64_t, SIZE> decode_jpeg_npu_output_size(
    at::IntArrayRef image_shape,
    int64_t channels);

} // namespace native
} // namespace at_npu

#endif
