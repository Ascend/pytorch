#include "torch_npu/csrc/aten/CustomFunctions.h"
#include <ATen/core/dispatch/Dispatcher.h>

#define GET_OP_HANDLE(BASE_NAME, OVERLOAD_NAME, FUNC) \
  c10::Dispatcher::singleton().findSchemaOrThrow(BASE_NAME, OVERLOAD_NAME).typed<decltype(FUNC)>()

namespace at_npu {
namespace native {
namespace custom_ops {
at::Tensor npu_dtype_cast(const at::Tensor &self, at::ScalarType dtype) {
  static auto op = GET_OP_HANDLE("npu::npu_dtype_cast", "", npu_dtype_cast);
  return op.call(self, dtype);
}

at::Tensor &npu_dtype_cast_(at::Tensor &self, const at::Tensor &src) {
  static auto op = GET_OP_HANDLE("npu::npu_dtype_cast_", "", npu_dtype_cast_);
  return op.call(self, src);
}

at::Tensor npu_bmmV2(const at::Tensor &self, const at::Tensor &mat2, at::IntArrayRef output_sizes) {
  static auto op = GET_OP_HANDLE("npu::npu_bmmV2", "", npu_bmmV2);
  return op.call(self, mat2, output_sizes);
}

at::Tensor npu_broadcast(const at::Tensor &self, at::IntArrayRef size) {
  static auto op = GET_OP_HANDLE("npu::npu_broadcast", "", npu_broadcast);
  return op.call(self, size);
}

at::Tensor &npu_broadcast_out(const at::Tensor &self, at::IntArrayRef size, at::Tensor &result) {
  static auto op = GET_OP_HANDLE("npu::npu_broadcast", "out", npu_broadcast_out);
  return op.call(self, size, result);
}

at::Tensor &npu_indexing_out(const at::Tensor &self, c10::IntArrayRef begin, c10::IntArrayRef end,
                             c10::IntArrayRef strides, int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask,
                             int64_t new_axis_mask, int64_t shrink_axis_mask, at::Tensor &result) {
  static auto op = GET_OP_HANDLE("npu::npu_indexing", "out", npu_indexing_out);
  return op.call(self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask,
                 result);
}

at::Tensor &npu_reshape_out(const at::Tensor &src, at::IntArrayRef shape, bool can_refresh, at::Tensor &result) {
  static auto op = GET_OP_HANDLE("npu::npu_reshape", "out", npu_reshape_out);
  return op.call(src, shape, can_refresh, result);
}

at::Tensor &npu_slice_out(const at::Tensor &self, c10::IntArrayRef offsets, c10::IntArrayRef size,
                          at::Tensor &result) {
  static auto op = GET_OP_HANDLE("npu::npu_slice", "out", npu_slice_out);
  return op.call(self, offsets, size, result);
}

std::tuple<at::Tensor, at::Tensor> _npu_ciou(const at::Tensor &self, const at::Tensor &gtboxes, bool trans,
                                             bool is_cross, int64_t mode, bool atan_sub_flag) {
  static auto op = GET_OP_HANDLE("npu::_npu_ciou", "", _npu_ciou);
  return op.call(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_fused_attention_score_fwd (
    const at::Tensor &query_layer, const at::Tensor &key_layer, const at::Tensor &value_layer,
    const at::Tensor &attention_mask, const at::Scalar &scale, double keep_prob, bool query_transpose,
    bool key_transpose, bool bmm_score_transpose_a, bool bmm_score_transpose_b, bool value_transpose,
    bool dx_transpose) {
  static auto op = GET_OP_HANDLE("npu::npu_fused_attention_score_fwd", "", npu_fused_attention_score_fwd);
  return op.call(query_layer, key_layer, value_layer, attention_mask, scale, keep_prob, query_transpose,
      key_transpose, bmm_score_transpose_a, bmm_score_transpose_b, value_transpose, dx_transpose);
}

std::tuple<at::Tensor, at::Tensor> _dropout_with_byte_mask(const at::Tensor &self, double p) {
  static auto op = GET_OP_HANDLE("npu::_dropout_with_byte_mask", "", _dropout_with_byte_mask);
  return op.call(self, p);
}

std::tuple<at::Tensor, at::Tensor> _npu_dropout(const at::Tensor &self, double p) {
  static auto op = GET_OP_HANDLE("npu::_npu_dropout", "", _npu_dropout);
  return op.call(self, p);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias, const at::Tensor &seq_mask,
    const at::Tensor &h, const at::Tensor &c, bool has_biases, int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first, bool flag_seq, bool direction) {
  static auto op = GET_OP_HANDLE("npu::npu_lstm", "", npu_lstm);
  return op.call(input, weight, bias, seq_mask, h, c, has_biases, num_layers,
      dropout, train, bidirectional, batch_first, flag_seq, direction);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_lstm_data(const at::Tensor &input, const at::Tensor &batch_sizes, const at::Tensor &weight, const at::Tensor &bias,
              const at::Tensor &seq_mask, const at::Tensor &h, const at::Tensor &c, bool has_biases,
              int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq,
              bool direction) {
  static auto op = GET_OP_HANDLE("npu::npu_lstm_data", "", npu_lstm_data);
  return op.call(input, batch_sizes, weight, bias, seq_mask, h, c, has_biases, num_layers, dropout, train,
                 bidirectional, batch_first, flag_seq, direction);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_lstm_cell(const at::Tensor &input, const at::Tensor &w_ih, const at::Tensor &w_hh, const at::Tensor &h,
              const at::Tensor &c, const c10::optional<at::Tensor> &bias) {
  static auto op = GET_OP_HANDLE("npu::npu_lstm_cell", "", npu_lstm_cell);
  return op.call(input, w_ih, w_hh, h, c, bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_gru(
    const at::Tensor &input, const at::Tensor &hx, const at::Tensor &weight_input, const at::Tensor &weight_hidden,
    const at::Tensor &bias_input, const at::Tensor &bias_hidden, const at::Tensor &seq_length, bool has_biases,
    int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  static auto op = GET_OP_HANDLE("npu::npu_gru", "", npu_gru);
  return op.call(input, hx, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, has_biases,
      num_layers, dropout, train, bidirectional, batch_first);
}

}  // namespace custom_ops
}  // namespace native
}  // namespace at_npu
