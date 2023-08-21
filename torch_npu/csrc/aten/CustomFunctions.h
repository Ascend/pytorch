#ifndef TORCH_NPU_CSRC_ATEN_CUSTOM_FUNCTIONS
#define TORCH_NPU_CSRC_ATEN_CUSTOM_FUNCTIONS

#include <ATen/ATen.h>

namespace at_npu {
namespace native {
namespace custom_ops {
at::Tensor npu_dtype_cast(const at::Tensor &self, at::ScalarType dtype);
at::Tensor &npu_dtype_cast_(at::Tensor &self, const at::Tensor &src);
at::Tensor npu_bmmV2(const at::Tensor &self, const at::Tensor &mat2, at::IntArrayRef output_sizes);
at::Tensor npu_broadcast(const at::Tensor &self, at::IntArrayRef size);
at::Tensor &npu_broadcast_out(const at::Tensor &self, at::IntArrayRef size, at::Tensor &result);
at::Tensor &npu_indexing_out(const at::Tensor &self, c10::IntArrayRef begin, c10::IntArrayRef end,
                             c10::IntArrayRef strides, int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask,
                             int64_t new_axis_mask, int64_t shrink_axis_mask, at::Tensor &result);
at::Tensor &npu_reshape_out(const at::Tensor &src, at::IntArrayRef shape, bool can_refresh, at::Tensor &result);
at::Tensor &npu_slice_out(const at::Tensor &self, c10::IntArrayRef offsets, c10::IntArrayRef size, at::Tensor &result);
std::tuple<at::Tensor, at::Tensor> _npu_ciou(const at::Tensor &self, const at::Tensor &gtboxes, bool trans,
                                             bool is_cross, int64_t mode, bool atan_sub_flag);
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_fused_attention_score_fwd(
    const at::Tensor &query_layer, const at::Tensor &key_layer, const at::Tensor &value_layer,
    const at::Tensor &attention_mask, const at::Scalar &scale, double keep_prob, bool query_transpose,
    bool key_transpose, bool bmm_score_transpose_a, bool bmm_score_transpose_b, bool value_transpose,
    bool dx_transpose);
std::tuple<at::Tensor, at::Tensor> _dropout_with_byte_mask(const at::Tensor &self, double p);
std::tuple<at::Tensor, at::Tensor> _npu_dropout(const at::Tensor &self, double p);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias, const at::Tensor &seq_mask,
    const at::Tensor &h, const at::Tensor &c, bool has_biases, int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first, bool flag_seq, bool direction);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_lstm_data(const at::Tensor &input, const at::Tensor &batch_sizes, const at::Tensor &weight, const at::Tensor &bias,
              const at::Tensor &seq_mask, const at::Tensor &h, const at::Tensor &c, bool has_biases,
              int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq,
              bool flag_direction);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_lstm_cell(const at::Tensor &input, const at::Tensor &w_ih, const at::Tensor &w_hh, const at::Tensor &h,
              const at::Tensor &c, const c10::optional<at::Tensor> &bias);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_gru(
    const at::Tensor &input, const at::Tensor &hx, const at::Tensor &weight_input, const at::Tensor &weight_hidden,
    const at::Tensor &bias_input, const at::Tensor &bias_hidden, const at::Tensor &seq_length, bool has_biases,
    int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first);
}  // namespace custom_ops
}  // namespace native
}  // namespace at_npu

#endif  // TORCH_NPU_CSRC_ATEN_CUSTOM_FUNCTIONS
