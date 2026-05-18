#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include "torch_npu/csrc/inductor/aoti_torch/generated/c_shim_npu.h"

using torch::stable::Tensor;

Tensor my__adaptive_avg_pool2d(Tensor self, std::vector<int64_t> output_size)
{
    AtenTensorHandle ret0;
    aoti_torch_npu__adaptive_avg_pool2d(self.get(), output_size.data(), output_size.size(), &ret0);
    return Tensor(ret0);
}

Tensor my__adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self)
{
    AtenTensorHandle ret0;
    aoti_torch_npu__adaptive_avg_pool2d_backward(grad_output.get(), self.get(), &ret0);
    return Tensor(ret0);
}

Tensor my__adaptive_avg_pool3d(Tensor self, std::vector<int64_t> output_size)
{
    AtenTensorHandle ret0;
    aoti_torch_npu__adaptive_avg_pool3d(self.get(), output_size.data(), output_size.size(), &ret0);
    return Tensor(ret0);
}

Tensor my__adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self)
{
    AtenTensorHandle ret0;
    aoti_torch_npu__adaptive_avg_pool3d_backward(grad_output.get(), self.get(), &ret0);
    return Tensor(ret0);
}

Tensor my__cdist_backward(Tensor grad, Tensor x1, Tensor x2, double p, Tensor cdist)
{
    AtenTensorHandle ret0;
    aoti_torch_npu__cdist_backward(grad.get(), x1.get(), x2.get(), p, cdist.get(), &ret0);
    return Tensor(ret0);
}

Tensor my__cdist_forward(Tensor x1, Tensor x2, double p, int64_t* compute_mode)
{
    AtenTensorHandle ret0;
    aoti_torch_npu__cdist_forward(x1.get(), x2.get(), p, compute_mode, &ret0);
    return Tensor(ret0);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> my__embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, std::optional<Tensor> per_sample_weights, bool include_last_offset, int64_t padding_idx)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    AtenTensorHandle ret2;
    AtenTensorHandle ret3;
    if (per_sample_weights.has_value()) {
        AtenTensorHandle per_sample_weights_handle = per_sample_weights.value().get();
        aoti_torch_npu__embedding_bag(weight.get(), indices.get(), offsets.get(), static_cast<int32_t>(scale_grad_by_freq), mode, static_cast<int32_t>(sparse), &per_sample_weights_handle, static_cast<int32_t>(include_last_offset), padding_idx, &ret0, &ret1, &ret2, &ret3);
    } else {
        aoti_torch_npu__embedding_bag(weight.get(), indices.get(), offsets.get(), static_cast<int32_t>(scale_grad_by_freq), mode, static_cast<int32_t>(sparse), nullptr, static_cast<int32_t>(include_last_offset), padding_idx, &ret0, &ret1, &ret2, &ret3);
    }
    return std::make_tuple(Tensor(ret0), Tensor(ret1), Tensor(ret2), Tensor(ret3));
}

std::tuple<Tensor, Tensor, Tensor, Tensor> my__embedding_bag_forward_only(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, std::optional<Tensor> per_sample_weights, bool include_last_offset, int64_t padding_idx)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    AtenTensorHandle ret2;
    AtenTensorHandle ret3;
    if (per_sample_weights.has_value()) {
        AtenTensorHandle per_sample_weights_handle = per_sample_weights.value().get();
        aoti_torch_npu__embedding_bag_forward_only(weight.get(), indices.get(), offsets.get(), static_cast<int32_t>(scale_grad_by_freq), mode, static_cast<int32_t>(sparse), &per_sample_weights_handle, static_cast<int32_t>(include_last_offset), padding_idx, &ret0, &ret1, &ret2, &ret3);
    } else {
        aoti_torch_npu__embedding_bag_forward_only(weight.get(), indices.get(), offsets.get(), static_cast<int32_t>(scale_grad_by_freq), mode, static_cast<int32_t>(sparse), nullptr, static_cast<int32_t>(include_last_offset), padding_idx, &ret0, &ret1, &ret2, &ret3);
    }
    return std::make_tuple(Tensor(ret0), Tensor(ret1), Tensor(ret2), Tensor(ret3));
}

Tensor my__embedding_bag_per_sample_weights_backward(Tensor grad, Tensor weight, Tensor indices, Tensor offsets, Tensor offset2bag, int64_t mode, int64_t padding_idx)
{
    AtenTensorHandle ret0;
    aoti_torch_npu__embedding_bag_per_sample_weights_backward(grad.get(), weight.get(), indices.get(), offsets.get(), offset2bag.get(), mode, padding_idx, &ret0);
    return Tensor(ret0);
}

Tensor my__fft_c2c(Tensor self, std::vector<int64_t> dim, int64_t normalization, bool forward)
{
    AtenTensorHandle ret0;
    aoti_torch_npu__fft_c2c(self.get(), dim.data(), dim.size(), normalization, static_cast<int32_t>(forward), &ret0);
    return Tensor(ret0);
}

Tensor my__fft_r2c(Tensor self, std::vector<int64_t> dim, int64_t normalization, bool onesided)
{
    AtenTensorHandle ret0;
    aoti_torch_npu__fft_r2c(self.get(), dim.data(), dim.size(), normalization, static_cast<int32_t>(onesided), &ret0);
    return Tensor(ret0);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> my__fused_moving_avg_obs_fq_helper_functional(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor running_min, Tensor running_max, Tensor scale, Tensor zero_point, double averaging_const, int64_t quant_min, int64_t quant_max, int64_t ch_axis, bool per_row_fake_quant, bool symmetric_quant)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    AtenTensorHandle ret2;
    AtenTensorHandle ret3;
    AtenTensorHandle ret4;
    AtenTensorHandle ret5;
    aoti_torch_npu__fused_moving_avg_obs_fq_helper_functional(self.get(), observer_on.get(), fake_quant_on.get(), running_min.get(), running_max.get(), scale.get(), zero_point.get(), averaging_const, quant_min, quant_max, ch_axis, static_cast<int32_t>(per_row_fake_quant), static_cast<int32_t>(symmetric_quant), &ret0, &ret1, &ret2, &ret3, &ret4, &ret5);
    return std::make_tuple(Tensor(ret0), Tensor(ret1), Tensor(ret2), Tensor(ret3), Tensor(ret4), Tensor(ret5));
}

std::tuple<Tensor, Tensor> my__fused_rms_norm(Tensor input, std::vector<int64_t> normalized_shape, std::optional<Tensor> weight, std::optional<double> eps)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    if (weight.has_value() && eps.has_value()) {
        AtenTensorHandle weight_handle = weight.value().get();
        double eps_val = eps.value();
        aoti_torch_npu__fused_rms_norm(input.get(), normalized_shape.data(), normalized_shape.size(), &weight_handle, &eps_val, &ret0, &ret1);
    } else if (weight.has_value()) {
        AtenTensorHandle weight_handle = weight.value().get();
        aoti_torch_npu__fused_rms_norm(input.get(), normalized_shape.data(), normalized_shape.size(), &weight_handle, nullptr, &ret0, &ret1);
    } else if (eps.has_value()) {
        double eps_val = eps.value();
        aoti_torch_npu__fused_rms_norm(input.get(), normalized_shape.data(), normalized_shape.size(), nullptr, &eps_val, &ret0, &ret1);
    } else {
        aoti_torch_npu__fused_rms_norm(input.get(), normalized_shape.data(), normalized_shape.size(), nullptr, nullptr, &ret0, &ret1);
    }
    return std::make_tuple(Tensor(ret0), Tensor(ret1));
}

Tensor my__pdist_forward(Tensor self, double p)
{
    AtenTensorHandle ret0;
    aoti_torch_npu__pdist_forward(self.get(), p, &ret0);
    return Tensor(ret0);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> my__scaled_dot_product_fused_attention_overrideable(Tensor query, Tensor key, Tensor value, std::optional<Tensor> attn_bias, double dropout_p, bool is_causal, bool return_debug_mask, std::optional<double> scale)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    AtenTensorHandle ret2;
    AtenTensorHandle ret3;
    int64_t ret4;
    int64_t ret5;
    AtenTensorHandle ret6;
    AtenTensorHandle ret7;
    AtenTensorHandle ret8;
    if (attn_bias.has_value() && scale.has_value()) {
        AtenTensorHandle attn_bias_handle = attn_bias.value().get();
        double scale_val = scale.value();
        aoti_torch_npu__scaled_dot_product_fused_attention_overrideable(query.get(), key.get(), value.get(), &attn_bias_handle, dropout_p, static_cast<int32_t>(is_causal), static_cast<int32_t>(return_debug_mask), &scale_val, &ret0, &ret1, &ret2, &ret3, &ret4, &ret5, &ret6, &ret7, &ret8);
    } else if (attn_bias.has_value()) {
        AtenTensorHandle attn_bias_handle = attn_bias.value().get();
        aoti_torch_npu__scaled_dot_product_fused_attention_overrideable(query.get(), key.get(), value.get(), &attn_bias_handle, dropout_p, static_cast<int32_t>(is_causal), static_cast<int32_t>(return_debug_mask), nullptr, &ret0, &ret1, &ret2, &ret3, &ret4, &ret5, &ret6, &ret7, &ret8);
    } else if (scale.has_value()) {
        double scale_val = scale.value();
        aoti_torch_npu__scaled_dot_product_fused_attention_overrideable(query.get(), key.get(), value.get(), nullptr, dropout_p, static_cast<int32_t>(is_causal), static_cast<int32_t>(return_debug_mask), &scale_val, &ret0, &ret1, &ret2, &ret3, &ret4, &ret5, &ret6, &ret7, &ret8);
    } else {
        aoti_torch_npu__scaled_dot_product_fused_attention_overrideable(query.get(), key.get(), value.get(), nullptr, dropout_p, static_cast<int32_t>(is_causal), static_cast<int32_t>(return_debug_mask), nullptr, &ret0, &ret1, &ret2, &ret3, &ret4, &ret5, &ret6, &ret7, &ret8);
    }
    AtenTensorHandle ret4_tensor;
    AtenTensorHandle ret5_tensor;
    aoti_torch_scalar_to_tensor_int64(ret4, &ret4_tensor);
    aoti_torch_scalar_to_tensor_int64(ret5, &ret5_tensor);
    return std::make_tuple(Tensor(ret0), Tensor(ret1), Tensor(ret2), Tensor(ret3), Tensor(ret4_tensor), Tensor(ret5_tensor), Tensor(ret6), Tensor(ret7), Tensor(ret8));
}

std::tuple<Tensor, Tensor, Tensor, Tensor> my__scaled_dot_product_fused_attention_overrideable_backward(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor attn_bias, std::vector<int32_t> grad_input_mask, Tensor out, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, int64_t max_q, int64_t max_k, double dropout_p, bool is_causal, Tensor philox_seed, Tensor philox_offset, std::optional<double> scale)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    AtenTensorHandle ret2;
    AtenTensorHandle ret3;
    std::vector<int32_t> grad_input_mask_vec(grad_input_mask.begin(), grad_input_mask.end());
    if (scale.has_value()) {
        double scale_val = scale.value();
        aoti_torch_npu__scaled_dot_product_fused_attention_overrideable_backward(grad_out.get(), query.get(), key.get(), value.get(), attn_bias.get(), grad_input_mask_vec.data(), grad_input_mask.size(), out.get(), logsumexp.get(), cum_seq_q.get(), cum_seq_k.get(), max_q, max_k, dropout_p, static_cast<int32_t>(is_causal), philox_seed.get(), philox_offset.get(), &scale_val, &ret0, &ret1, &ret2, &ret3);
    } else {
        aoti_torch_npu__scaled_dot_product_fused_attention_overrideable_backward(grad_out.get(), query.get(), key.get(), value.get(), attn_bias.get(), grad_input_mask_vec.data(), grad_input_mask.size(), out.get(), logsumexp.get(), cum_seq_q.get(), cum_seq_k.get(), max_q, max_k, dropout_p, static_cast<int32_t>(is_causal), philox_seed.get(), philox_offset.get(), nullptr, &ret0, &ret1, &ret2, &ret3);
    }
    return std::make_tuple(Tensor(ret0), Tensor(ret1), Tensor(ret2), Tensor(ret3));
}

std::tuple<Tensor, Tensor, Tensor> my__thnn_fused_lstm_cell(Tensor input_gates, Tensor hidden_gates, Tensor cx, std::optional<Tensor> input_bias, std::optional<Tensor> hidden_bias)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    AtenTensorHandle ret2;
    if (input_bias.has_value() && hidden_bias.has_value()) {
        AtenTensorHandle input_bias_handle = input_bias.value().get();
        AtenTensorHandle hidden_bias_handle = hidden_bias.value().get();
        aoti_torch_npu__thnn_fused_lstm_cell(input_gates.get(), hidden_gates.get(), cx.get(), &input_bias_handle, &hidden_bias_handle, &ret0, &ret1, &ret2);
    } else if (input_bias.has_value()) {
        AtenTensorHandle input_bias_handle = input_bias.value().get();
        aoti_torch_npu__thnn_fused_lstm_cell(input_gates.get(), hidden_gates.get(), cx.get(), &input_bias_handle, nullptr, &ret0, &ret1, &ret2);
    } else if (hidden_bias.has_value()) {
        AtenTensorHandle hidden_bias_handle = hidden_bias.value().get();
        aoti_torch_npu__thnn_fused_lstm_cell(input_gates.get(), hidden_gates.get(), cx.get(), nullptr, &hidden_bias_handle, &ret0, &ret1, &ret2);
    } else {
        aoti_torch_npu__thnn_fused_lstm_cell(input_gates.get(), hidden_gates.get(), cx.get(), nullptr, nullptr, &ret0, &ret1, &ret2);
    }
    return std::make_tuple(Tensor(ret0), Tensor(ret1), Tensor(ret2));
}

Tensor my__trilinear(Tensor i1, Tensor i2, Tensor i3, std::vector<int64_t> expand1, std::vector<int64_t> expand2, std::vector<int64_t> expand3, std::vector<int64_t> sumdim, int64_t unroll_dim)
{
    AtenTensorHandle ret0;
    aoti_torch_npu__trilinear(i1.get(), i2.get(), i3.get(), expand1.data(), expand1.size(), expand2.data(), expand2.size(), expand3.data(), expand3.size(), sumdim.data(), sumdim.size(), unroll_dim, &ret0);
    return Tensor(ret0);
}

Tensor my_abs(Tensor self)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_abs(self.get(), &ret0);
    return Tensor(ret0);
}

std::tuple<Tensor, Tensor> my_adaptive_max_pool2d(Tensor self, std::vector<int64_t> output_size)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    aoti_torch_npu_adaptive_max_pool2d(self.get(), output_size.data(), output_size.size(), &ret0, &ret1);
    return std::make_tuple(Tensor(ret0), Tensor(ret1));
}

Tensor my_adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_adaptive_max_pool2d_backward(grad_output.get(), self.get(), indices.get(), &ret0);
    return Tensor(ret0);
}

std::tuple<Tensor, Tensor> my_adaptive_max_pool3d(Tensor self, std::vector<int64_t> output_size)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    aoti_torch_npu_adaptive_max_pool3d(self.get(), output_size.data(), output_size.size(), &ret0, &ret1);
    return std::make_tuple(Tensor(ret0), Tensor(ret1));
}

Tensor my_adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_adaptive_max_pool3d_backward(grad_output.get(), self.get(), indices.get(), &ret0);
    return Tensor(ret0);
}

Tensor my_add_Scalar(Tensor self, double other, double alpha)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_add_Scalar(self.get(), other, alpha, &ret0);
    return Tensor(ret0);
}

Tensor my_add_Tensor(Tensor self, Tensor other, double alpha)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_add_Tensor(self.get(), other.get(), alpha, &ret0);
    return Tensor(ret0);
}

Tensor my_addbmm(Tensor self, Tensor batch1, Tensor batch2, double beta, double alpha)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_addbmm(self.get(), batch1.get(), batch2.get(), beta, alpha, &ret0);
    return Tensor(ret0);
}

void my_addmm_out(Tensor out, Tensor self, Tensor mat1, Tensor mat2, double beta, double alpha)
{
    aoti_torch_npu_addmm_out(out.get(), self.get(), mat1.get(), mat2.get(), beta, alpha);
}

Tensor my_addmv(Tensor self, Tensor mat, Tensor vec, double beta, double alpha)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_addmv(self.get(), mat.get(), vec.get(), beta, alpha, &ret0);
    return Tensor(ret0);
}

Tensor my_angle(Tensor self)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_angle(self.get(), &ret0);
    return Tensor(ret0);
}

Tensor my_avg_pool2d(Tensor self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, int64_t* divisor_override)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_avg_pool2d(self.get(), kernel_size.data(), kernel_size.size(), stride.data(), stride.size(), padding.data(), padding.size(), static_cast<int32_t>(ceil_mode), static_cast<int32_t>(count_include_pad), divisor_override, &ret0);
    return Tensor(ret0);
}

Tensor my_avg_pool2d_backward(Tensor grad_output, Tensor self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, int64_t* divisor_override)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_avg_pool2d_backward(grad_output.get(), self.get(), kernel_size.data(), kernel_size.size(), stride.data(), stride.size(), padding.data(), padding.size(), static_cast<int32_t>(ceil_mode), static_cast<int32_t>(count_include_pad), divisor_override, &ret0);
    return Tensor(ret0);
}

Tensor my_avg_pool3d(Tensor self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, int64_t* divisor_override)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_avg_pool3d(self.get(), kernel_size.data(), kernel_size.size(), stride.data(), stride.size(), padding.data(), padding.size(), static_cast<int32_t>(ceil_mode), static_cast<int32_t>(count_include_pad), divisor_override, &ret0);
    return Tensor(ret0);
}

Tensor my_avg_pool3d_backward(Tensor grad_output, Tensor self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode, bool count_include_pad, int64_t* divisor_override)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_avg_pool3d_backward(grad_output.get(), self.get(), kernel_size.data(), kernel_size.size(), stride.data(), stride.size(), padding.data(), padding.size(), static_cast<int32_t>(ceil_mode), static_cast<int32_t>(count_include_pad), divisor_override, &ret0);
    return Tensor(ret0);
}

void my_baddbmm_out(Tensor out, Tensor self, Tensor batch1, Tensor batch2, double beta, double alpha)
{
    aoti_torch_npu_baddbmm_out(out.get(), self.get(), batch1.get(), batch2.get(), beta, alpha);
}

void my_bernoulli__Tensor(Tensor self, Tensor p, std::optional<AtenGeneratorHandle> generator)
{
    AtenGeneratorHandle generator_handle = generator.value_or(nullptr);
    aoti_torch_npu_bernoulli__Tensor(self.get(), p.get(), generator_handle ? &generator_handle : nullptr);
}

void my_bernoulli__float(Tensor self, double p, std::optional<AtenGeneratorHandle> generator)
{
    AtenGeneratorHandle generator_handle = generator.value_or(nullptr);
    aoti_torch_npu_bernoulli__float(self.get(), p, generator_handle ? &generator_handle : nullptr);
}

void my_bmm_out(Tensor out, Tensor self, Tensor mat2)
{
    aoti_torch_npu_bmm_out(out.get(), self.get(), mat2.get());
}

Tensor my_bucketize_Tensor(Tensor self, Tensor boundaries, bool out_int32, bool right)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_bucketize_Tensor(self.get(), boundaries.get(), static_cast<int32_t>(out_int32), static_cast<int32_t>(right), &ret0);
    return Tensor(ret0);
}

Tensor my_cat(std::vector<Tensor> tensors, int64_t dim)
{
    AtenTensorHandle ret0;
    std::vector<AtenTensorHandle> tensors_handles(tensors.size());
    for (size_t i = 0; i < tensors.size(); i++) { tensors_handles[i] = tensors[i].get(); }
    aoti_torch_npu_cat(tensors_handles.data(), tensors.size(), dim, &ret0);
    return Tensor(ret0);
}

Tensor my_cholesky_solve(Tensor self, Tensor input2, bool upper)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_cholesky_solve(self.get(), input2.get(), static_cast<int32_t>(upper), &ret0);
    return Tensor(ret0);
}

Tensor my_convolution(Tensor input, Tensor weight, std::optional<Tensor> bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool transposed, std::vector<int64_t> output_padding, int64_t groups)
{
    AtenTensorHandle ret0;
    if (bias.has_value()) {
        AtenTensorHandle bias_handle = bias.value().get();
        aoti_torch_npu_convolution(input.get(), weight.get(), &bias_handle, stride.data(), stride.size(), padding.data(), padding.size(), dilation.data(), dilation.size(), static_cast<int32_t>(transposed), output_padding.data(), output_padding.size(), groups, &ret0);
    } else {
        aoti_torch_npu_convolution(input.get(), weight.get(), nullptr, stride.data(), stride.size(), padding.data(), padding.size(), dilation.data(), dilation.size(), static_cast<int32_t>(transposed), output_padding.data(), output_padding.size(), groups, &ret0);
    }
    return Tensor(ret0);
}

std::tuple<Tensor, Tensor, Tensor> my_convolution_backward(Tensor grad_output, Tensor input, Tensor weight, std::optional<std::vector<int64_t>> bias_sizes, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool transposed, std::vector<int64_t> output_padding, int64_t groups, std::vector<int32_t> output_mask)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    AtenTensorHandle ret2;
    std::vector<const int64_t*> bs_ptrs;
    if (bias_sizes.has_value()) {
        auto& bs = bias_sizes.value();
        bs_ptrs.reserve(bs.size());
        for (auto& v : bs) {
            bs_ptrs.push_back(&v);
        }
        aoti_torch_npu_convolution_backward(grad_output.get(), input.get(), weight.get(), bs_ptrs.data(), bs_ptrs.size(), stride.data(), stride.size(), padding.data(), padding.size(), dilation.data(), dilation.size(), static_cast<int32_t>(transposed), output_padding.data(), output_padding.size(), groups, output_mask.data(), output_mask.size(), &ret0, &ret1, &ret2);
    } else {
        aoti_torch_npu_convolution_backward(grad_output.get(), input.get(), weight.get(), nullptr, 0, stride.data(), stride.size(), padding.data(), padding.size(), dilation.data(), dilation.size(), static_cast<int32_t>(transposed), output_padding.data(), output_padding.size(), groups, output_mask.data(), output_mask.size(), &ret0, &ret1, &ret2);
    }
    return std::make_tuple(Tensor(ret0), Tensor(ret1), Tensor(ret2));
}

std::tuple<Tensor, Tensor> my_cummax(Tensor self, int64_t dim)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    aoti_torch_npu_cummax(self.get(), dim, &ret0, &ret1);
    return std::make_tuple(Tensor(ret0), Tensor(ret1));
}

std::tuple<Tensor, Tensor> my_cummin(Tensor self, int64_t dim)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    aoti_torch_npu_cummin(self.get(), dim, &ret0, &ret1);
    return std::make_tuple(Tensor(ret0), Tensor(ret1));
}

Tensor my_cumsum(Tensor self, int64_t dim, int32_t* dtype)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_cumsum(self.get(), dim, dtype, &ret0);
    return Tensor(ret0);
}

Tensor my_exponential(Tensor self, double lambd, std::optional<AtenGeneratorHandle> generator)
{
    AtenTensorHandle ret0;
    AtenGeneratorHandle generator_handle = generator.value_or(nullptr);
    aoti_torch_npu_exponential(self.get(), lambd, generator_handle ? &generator_handle : nullptr, &ret0);
    return Tensor(ret0);
}

void my_fill__Scalar(Tensor self, double value)
{
    aoti_torch_npu_fill__Scalar(self.get(), value);
}

std::tuple<Tensor, Tensor> my_grid_sampler_2d_backward(Tensor grad_output, Tensor input, Tensor grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, std::vector<int32_t> output_mask)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    std::vector<int32_t> output_mask_vec(output_mask.begin(), output_mask.end());
    aoti_torch_npu_grid_sampler_2d_backward(grad_output.get(), input.get(), grid.get(), interpolation_mode, padding_mode, static_cast<int32_t>(align_corners), output_mask_vec.data(), output_mask.size(), &ret0, &ret1);
    return std::make_tuple(Tensor(ret0), Tensor(ret1));
}

Tensor my_hann_window(int64_t window_length, int32_t* dtype, int32_t* layout, int32_t* device, int32_t* pin_memory)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_hann_window(window_length, dtype, layout, device, 0, pin_memory, &ret0);
    return Tensor(ret0);
}

Tensor my_histc(Tensor self, int64_t bins, double min, double max)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_histc(self.get(), bins, min, max, &ret0);
    return Tensor(ret0);
}

Tensor my_index_Tensor(Tensor self, std::vector<Tensor> indices)
{
    AtenTensorHandle ret0;
    std::vector<AtenTensorHandle> indices_handles_raw(indices.size());
    for (size_t i = 0; i < indices.size(); i++) { indices_handles_raw[i] = indices[i].get(); }
    std::vector<const AtenTensorHandle*> indices_handles(indices.size());
    for (size_t i = 0; i < indices.size(); i++) { indices_handles[i] = &indices_handles_raw[i]; }
    aoti_torch_npu_index_Tensor(self.get(), indices_handles.data(), indices.size(), &ret0);
    return Tensor(ret0);
}

Tensor my_index_put(Tensor self, std::vector<Tensor> indices, Tensor values, bool accumulate)
{
    AtenTensorHandle ret0;
    std::vector<AtenTensorHandle> indices_handles_raw(indices.size());
    for (size_t i = 0; i < indices.size(); i++) { indices_handles_raw[i] = indices[i].get(); }
    std::vector<const AtenTensorHandle*> indices_handles(indices.size());
    for (size_t i = 0; i < indices.size(); i++) { indices_handles[i] = &indices_handles_raw[i]; }
    aoti_torch_npu_index_put(self.get(), indices_handles.data(), indices.size(), values.get(), static_cast<int32_t>(accumulate), &ret0);
    return Tensor(ret0);
}

std::tuple<Tensor, Tensor> my_kthvalue(Tensor self, int64_t k, int64_t dim, bool keepdim)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    aoti_torch_npu_kthvalue(self.get(), k, dim, static_cast<int32_t>(keepdim), &ret0, &ret1);
    return std::make_tuple(Tensor(ret0), Tensor(ret1));
}

Tensor my_logcumsumexp(Tensor self, int64_t dim)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_logcumsumexp(self.get(), dim, &ret0);
    return Tensor(ret0);
}

Tensor my_masked_scatter(Tensor self, Tensor mask, Tensor source)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_masked_scatter(self.get(), mask.get(), source.get(), &ret0);
    return Tensor(ret0);
}

Tensor my_masked_scatter_backward(Tensor grad_output, Tensor mask, std::vector<int64_t> sizes)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_masked_scatter_backward(grad_output.get(), mask.get(), sizes.data(), sizes.size(), &ret0);
    return Tensor(ret0);
}

Tensor my_masked_select(Tensor self, Tensor mask)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_masked_select(self.get(), mask.get(), &ret0);
    return Tensor(ret0);
}

std::tuple<Tensor, Tensor> my_max_pool2d_with_indices(Tensor self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    aoti_torch_npu_max_pool2d_with_indices(self.get(), kernel_size.data(), kernel_size.size(), stride.data(), stride.size(), padding.data(), padding.size(), dilation.data(), dilation.size(), static_cast<int32_t>(ceil_mode), &ret0, &ret1);
    return std::make_tuple(Tensor(ret0), Tensor(ret1));
}

Tensor my_max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode, Tensor indices)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_max_pool2d_with_indices_backward(grad_output.get(), self.get(), kernel_size.data(), kernel_size.size(), stride.data(), stride.size(), padding.data(), padding.size(), dilation.data(), dilation.size(), static_cast<int32_t>(ceil_mode), indices.get(), &ret0);
    return Tensor(ret0);
}

std::tuple<Tensor, Tensor> my_max_pool3d_with_indices(Tensor self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    aoti_torch_npu_max_pool3d_with_indices(self.get(), kernel_size.data(), kernel_size.size(), stride.data(), stride.size(), padding.data(), padding.size(), dilation.data(), dilation.size(), static_cast<int32_t>(ceil_mode), &ret0, &ret1);
    return std::make_tuple(Tensor(ret0), Tensor(ret1));
}

Tensor my_max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode, Tensor indices)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_max_pool3d_with_indices_backward(grad_output.get(), self.get(), kernel_size.data(), kernel_size.size(), stride.data(), stride.size(), padding.data(), padding.size(), dilation.data(), dilation.size(), static_cast<int32_t>(ceil_mode), indices.get(), &ret0);
    return Tensor(ret0);
}

Tensor my_max_unpool2d(Tensor self, Tensor indices, std::vector<int64_t> output_size)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_max_unpool2d(self.get(), indices.get(), output_size.data(), output_size.size(), &ret0);
    return Tensor(ret0);
}

Tensor my_max_unpool3d(Tensor self, Tensor indices, std::vector<int64_t> output_size, std::vector<int64_t> stride, std::vector<int64_t> padding)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_max_unpool3d(self.get(), indices.get(), output_size.data(), output_size.size(), stride.data(), stride.size(), padding.data(), padding.size(), &ret0);
    return Tensor(ret0);
}

Tensor my_median(Tensor self)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_median(self.get(), &ret0);
    return Tensor(ret0);
}

void my_mm_out(Tensor out, Tensor self, Tensor mat2)
{
    aoti_torch_npu_mm_out(out.get(), self.get(), mat2.get());
}

Tensor my_mul_Scalar(Tensor self, double other)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_mul_Scalar(self.get(), other, &ret0);
    return Tensor(ret0);
}

Tensor my_mul_Tensor(Tensor self, Tensor other)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_mul_Tensor(self.get(), other.get(), &ret0);
    return Tensor(ret0);
}

Tensor my_nanmedian(Tensor self)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_nanmedian(self.get(), &ret0);
    return Tensor(ret0);
}

Tensor my_narrow(Tensor self, int64_t dim, int64_t start, int64_t length)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_narrow(self.get(), dim, start, length, &ret0);
    return Tensor(ret0);
}

std::tuple<Tensor, Tensor> my_native_dropout(Tensor input, double p, int32_t* train)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    aoti_torch_npu_native_dropout(input.get(), p, train, &ret0, &ret1);
    return std::make_tuple(Tensor(ret0), Tensor(ret1));
}

Tensor my_nonzero(Tensor self)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_nonzero(self.get(), &ret0);
    return Tensor(ret0);
}

Tensor my_normal_functional(Tensor self, double mean, double std, std::optional<AtenGeneratorHandle> generator)
{
    AtenTensorHandle ret0;
    AtenGeneratorHandle generator_handle = generator.value_or(nullptr);
    aoti_torch_npu_normal_functional(self.get(), mean, std, generator_handle ? &generator_handle : nullptr, &ret0);
    return Tensor(ret0);
}

Tensor my_pad(Tensor self, std::vector<int64_t> pad, std::string mode, double* value)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_pad(self.get(), pad.data(), pad.size(), mode.c_str(), value, &ret0);
    return Tensor(ret0);
}

Tensor my_permute(Tensor self, std::vector<int64_t> dims)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_permute(self.get(), dims.data(), dims.size(), &ret0);
    return Tensor(ret0);
}

Tensor my_polar(Tensor abs, Tensor angle)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_polar(abs.get(), angle.get(), &ret0);
    return Tensor(ret0);
}

Tensor my_pow_Scalar(double self, Tensor exponent)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_pow_Scalar(self, exponent.get(), &ret0);
    return Tensor(ret0);
}

Tensor my_pow_Tensor_Scalar(Tensor self, double exponent)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_pow_Tensor_Scalar(self.get(), exponent, &ret0);
    return Tensor(ret0);
}

Tensor my_pow_Tensor_Tensor(Tensor self, Tensor exponent)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_pow_Tensor_Tensor(self.get(), exponent.get(), &ret0);
    return Tensor(ret0);
}

Tensor my_rand(std::vector<int64_t> size, int32_t* dtype, int32_t* layout, int32_t* device, int32_t* pin_memory)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_rand(size.data(), size.size(), dtype, layout, device, 0, pin_memory, &ret0);
    return Tensor(ret0);
}

Tensor my_rand_generator(std::vector<int64_t> size, std::optional<AtenGeneratorHandle> generator, int32_t* dtype, int32_t* layout, int32_t* device, int32_t* pin_memory)
{
    AtenTensorHandle ret0;
    AtenGeneratorHandle generator_handle = generator.value_or(nullptr);
    aoti_torch_npu_rand_generator(size.data(), size.size(), generator_handle ? &generator_handle : nullptr, dtype, layout, device, 0, pin_memory, &ret0);
    return Tensor(ret0);
}

Tensor my_randint(int64_t high, std::vector<int64_t> size, int32_t* dtype, int32_t* layout, int32_t* device, int32_t* pin_memory)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_randint(high, size.data(), size.size(), dtype, layout, device, 0, pin_memory, &ret0);
    return Tensor(ret0);
}

Tensor my_randint_generator(int64_t high, std::vector<int64_t> size, std::optional<AtenGeneratorHandle> generator, int32_t* dtype, int32_t* layout, int32_t* device, int32_t* pin_memory)
{
    AtenTensorHandle ret0;
    AtenGeneratorHandle generator_handle = generator.value_or(nullptr);
    aoti_torch_npu_randint_generator(high, size.data(), size.size(), generator_handle ? &generator_handle : nullptr, dtype, layout, device, 0, pin_memory, &ret0);
    return Tensor(ret0);
}

Tensor my_randint_low(int64_t low, int64_t high, std::vector<int64_t> size, int32_t* dtype, int32_t* layout, int32_t* device, int32_t* pin_memory)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_randint_low(low, high, size.data(), size.size(), dtype, layout, device, 0, pin_memory, &ret0);
    return Tensor(ret0);
}

void my_randint_low_out(Tensor out, int64_t low, int64_t high, std::vector<int64_t> size)
{
    aoti_torch_npu_randint_low_out(out.get(), low, high, size.data(), size.size());
}

Tensor my_randn(std::vector<int64_t> size, int32_t* dtype, int32_t* layout, int32_t* device, int32_t* pin_memory)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_randn(size.data(), size.size(), dtype, layout, device, 0, pin_memory, &ret0);
    return Tensor(ret0);
}

Tensor my_randn_generator(std::vector<int64_t> size, std::optional<AtenGeneratorHandle> generator, int32_t* dtype, int32_t* layout, int32_t* device, int32_t* pin_memory)
{
    AtenTensorHandle ret0;
    AtenGeneratorHandle generator_handle = generator.value_or(nullptr);
    aoti_torch_npu_randn_generator(size.data(), size.size(), generator_handle ? &generator_handle : nullptr, dtype, layout, device, 0, pin_memory, &ret0);
    return Tensor(ret0);
}

Tensor my_randperm(int64_t n, int32_t* dtype, int32_t* layout, int32_t* device, int32_t* pin_memory)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_randperm(n, dtype, layout, device, 0, pin_memory, &ret0);
    return Tensor(ret0);
}

Tensor my_replication_pad1d_backward(Tensor grad_output, Tensor self, std::vector<int64_t> padding)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_replication_pad1d_backward(grad_output.get(), self.get(), padding.data(), padding.size(), &ret0);
    return Tensor(ret0);
}

Tensor my_replication_pad2d_backward(Tensor grad_output, Tensor self, std::vector<int64_t> padding)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_replication_pad2d_backward(grad_output.get(), self.get(), padding.data(), padding.size(), &ret0);
    return Tensor(ret0);
}

Tensor my_reshape(Tensor self, std::vector<int64_t> shape)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_reshape(self.get(), shape.data(), shape.size(), &ret0);
    return Tensor(ret0);
}

void my_resize_(Tensor self, std::vector<int64_t> size, int32_t* memory_format)
{
    aoti_torch_npu_resize_(self.get(), size.data(), size.size(), memory_format);
}

void my_resize_as_(Tensor self, Tensor the_template, int32_t* memory_format)
{
    aoti_torch_npu_resize_as_(self.get(), the_template.get(), memory_format);
}

void my_scatter_src_out(Tensor out, Tensor self, int64_t dim, Tensor index, Tensor src)
{
    aoti_torch_npu_scatter_src_out(out.get(), self.get(), dim, index.get(), src.get());
}

void my_scatter_value_out(Tensor out, Tensor self, int64_t dim, Tensor index, double value)
{
    aoti_torch_npu_scatter_value_out(out.get(), self.get(), dim, index.get(), value);
}

Tensor my_searchsorted_Scalar(Tensor sorted_sequence, double self, bool out_int32, bool right, std::optional<Tensor> sorter)
{
    AtenTensorHandle ret0;
    if (sorter.has_value()) {
        AtenTensorHandle sorter_handle = sorter.value().get();
        aoti_torch_npu_searchsorted_Scalar(sorted_sequence.get(), self, static_cast<int32_t>(out_int32), static_cast<int32_t>(right), nullptr, &sorter_handle, &ret0);
    } else {
        aoti_torch_npu_searchsorted_Scalar(sorted_sequence.get(), self, static_cast<int32_t>(out_int32), static_cast<int32_t>(right), nullptr, nullptr, &ret0);
    }
    return Tensor(ret0);
}

Tensor my_searchsorted_Tensor(Tensor sorted_sequence, Tensor self, bool out_int32, bool right, std::optional<Tensor> sorter)
{
    AtenTensorHandle ret0;
    if (sorter.has_value()) {
        AtenTensorHandle sorter_handle = sorter.value().get();
        aoti_torch_npu_searchsorted_Tensor(sorted_sequence.get(), self.get(), static_cast<int32_t>(out_int32), static_cast<int32_t>(right), nullptr, &sorter_handle, &ret0);
    } else {
        aoti_torch_npu_searchsorted_Tensor(sorted_sequence.get(), self.get(), static_cast<int32_t>(out_int32), static_cast<int32_t>(right), nullptr, nullptr, &ret0);
    }
    return Tensor(ret0);
}

void my_set__source_Tensor(Tensor self, Tensor source)
{
    aoti_torch_npu_set__source_Tensor(self.get(), source.get());
}

Tensor my_slice_Tensor(Tensor self, int64_t dim, int64_t* start, int64_t* end, int64_t step)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_slice_Tensor(self.get(), dim, start, end, step, &ret0);
    return Tensor(ret0);
}

Tensor my_soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int64_t reduction)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_soft_margin_loss_backward(grad_output.get(), self.get(), target.get(), reduction, &ret0);
    return Tensor(ret0);
}

std::tuple<Tensor, Tensor> my_sort(Tensor self, int64_t dim, bool descending)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    aoti_torch_npu_sort(self.get(), dim, static_cast<int32_t>(descending), &ret0, &ret1);
    return std::make_tuple(Tensor(ret0), Tensor(ret1));
}

std::tuple<Tensor, Tensor> my_sort_stable(Tensor self, int32_t* stable, int64_t dim, bool descending)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    aoti_torch_npu_sort_stable(self.get(), stable, dim, static_cast<int32_t>(descending), &ret0, &ret1);
    return std::make_tuple(Tensor(ret0), Tensor(ret1));
}

Tensor my_squeeze_dim(Tensor self, int64_t dim)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_squeeze_dim(self.get(), dim, &ret0);
    return Tensor(ret0);
}

Tensor my_to_sparse(Tensor self, std::optional<int32_t> layout, std::optional<std::vector<int64_t>> blocksize, std::optional<int64_t> dense_dim)
{
    AtenTensorHandle ret0;
    int32_t layout_val = 0;
    int32_t* layout_ptr = nullptr;
    if (layout.has_value()) {
        layout_val = layout.value();
        layout_ptr = &layout_val;
    }
    int64_t dense_dim_val = 0;
    int64_t* dense_dim_ptr = nullptr;
    if (dense_dim.has_value()) {
        dense_dim_val = dense_dim.value();
        dense_dim_ptr = &dense_dim_val;
    }
    std::vector<const int64_t*> bs_ptrs;
    if (blocksize.has_value()) {
        auto& bs = blocksize.value();
        bs_ptrs.reserve(bs.size());
        for (auto& v : bs) {
            bs_ptrs.push_back(&v);
        }
        aoti_torch_npu_to_sparse(self.get(), layout_ptr, bs_ptrs.data(), bs_ptrs.size(), dense_dim_ptr, &ret0);
    } else {
        aoti_torch_npu_to_sparse(self.get(), layout_ptr, nullptr, 0, dense_dim_ptr, &ret0);
    }
    return Tensor(ret0);
}

std::tuple<Tensor, Tensor> my_topk(Tensor self, int64_t k, int64_t dim, bool largest, bool sorted)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    aoti_torch_npu_topk(self.get(), k, dim, static_cast<int32_t>(largest), static_cast<int32_t>(sorted), &ret0, &ret1);
    return std::make_tuple(Tensor(ret0), Tensor(ret1));
}

Tensor my_uniform(Tensor self, double from, double to, std::optional<AtenGeneratorHandle> generator)
{
    AtenTensorHandle ret0;
    AtenGeneratorHandle generator_handle = generator.value_or(nullptr);
    aoti_torch_npu_uniform(self.get(), from, to, generator_handle ? &generator_handle : nullptr, &ret0);
    return Tensor(ret0);
}

Tensor my_upsample_bicubic2d_backward(Tensor grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners, double* scales_h, double* scales_w)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_upsample_bicubic2d_backward(grad_output.get(), output_size.data(), output_size.size(), input_size.data(), input_size.size(), static_cast<int32_t>(align_corners), scales_h, scales_w, &ret0);
    return Tensor(ret0);
}

Tensor my_upsample_linear1d_backward(Tensor grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners, double* scales)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_upsample_linear1d_backward(grad_output.get(), output_size.data(), output_size.size(), input_size.data(), input_size.size(), static_cast<int32_t>(align_corners), scales, &ret0);
    return Tensor(ret0);
}

Tensor my_upsample_trilinear3d_backward(Tensor grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners, double* scales_d, double* scales_h, double* scales_w)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_upsample_trilinear3d_backward(grad_output.get(), output_size.data(), output_size.size(), input_size.data(), input_size.size(), static_cast<int32_t>(align_corners), scales_d, scales_h, scales_w, &ret0);
    return Tensor(ret0);
}

Tensor my_view_dtype(Tensor self, int32_t dtype)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_view_dtype(self.get(), dtype, &ret0);
    return Tensor(ret0);
}

Tensor my_view_as_complex(Tensor self)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_view_as_complex(self.get(), &ret0);
    return Tensor(ret0);
}

Tensor my_view_as_real(Tensor self)
{
    AtenTensorHandle ret0;
    aoti_torch_npu_view_as_real(self.get(), &ret0);
    return Tensor(ret0);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_211, m) {
  m.def("my__adaptive_avg_pool2d(Tensor self, int[] output_size) -> Tensor");
  m.def("my__adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor");
  m.def("my__adaptive_avg_pool3d(Tensor self, int[] output_size) -> Tensor");
  m.def("my__adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self) -> Tensor");
  m.def("my__cdist_backward(Tensor grad, Tensor x1, Tensor x2, float p, Tensor cdist) -> Tensor");
  m.def("my__cdist_forward(Tensor x1, Tensor x2, float p, int? compute_mode) -> Tensor");
  m.def("my__embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, bool include_last_offset, int padding_idx) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("my__embedding_bag_forward_only(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, bool include_last_offset, int padding_idx) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("my__embedding_bag_per_sample_weights_backward(Tensor grad, Tensor weight, Tensor indices, Tensor offsets, Tensor offset2bag, int mode, int padding_idx) -> Tensor");
  m.def("my__fft_c2c(Tensor self, int[] dim, int normalization, bool forward) -> Tensor");
  m.def("my__fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> Tensor");
  m.def("my__fused_moving_avg_obs_fq_helper_functional(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor running_min, Tensor running_max, Tensor scale, Tensor zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant, bool symmetric_quant) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("my__fused_rms_norm(Tensor input, int[] normalized_shape, Tensor? weight, float? eps) -> (Tensor, Tensor)");
  m.def("my__pdist_forward(Tensor self, float p) -> Tensor");
  m.def("my__scaled_dot_product_fused_attention_overrideable(Tensor query, Tensor key, Tensor value, Tensor? attn_bias, float dropout_p, bool is_causal, bool return_debug_mask, float? scale) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("my__scaled_dot_product_fused_attention_overrideable_backward(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor attn_bias, int[] grad_input_mask, Tensor out, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, int max_q, int max_k, float dropout_p, bool is_causal, Tensor philox_seed, Tensor philox_offset, float? scale) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("my__thnn_fused_lstm_cell(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor? input_bias, Tensor? hidden_bias) -> (Tensor, Tensor, Tensor)");
  m.def("my__trilinear(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim) -> Tensor");
  m.def("my_abs(Tensor self) -> Tensor");
  m.def("my_adaptive_max_pool2d(Tensor self, int[] output_size) -> (Tensor, Tensor)");
  m.def("my_adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor");
  m.def("my_adaptive_max_pool3d(Tensor self, int[] output_size) -> (Tensor, Tensor)");
  m.def("my_adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor");
  m.def("my_add_Scalar(Tensor self, float other, float alpha) -> Tensor");
  m.def("my_add_Tensor(Tensor self, Tensor other, float alpha) -> Tensor");
  m.def("my_addbmm(Tensor self, Tensor batch1, Tensor batch2, float beta, float alpha) -> Tensor");
  m.def("my_addmm_out(Tensor out, Tensor self, Tensor mat1, Tensor mat2, float beta, float alpha) -> ()");
  m.def("my_addmv(Tensor self, Tensor mat, Tensor vec, float beta, float alpha) -> Tensor");
  m.def("my_angle(Tensor self) -> Tensor");
  m.def("my_avg_pool2d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor");
  m.def("my_avg_pool2d_backward(Tensor grad_output, Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor");
  m.def("my_avg_pool3d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor");
  m.def("my_avg_pool3d_backward(Tensor grad_output, Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor");
  m.def("my_baddbmm_out(Tensor out, Tensor self, Tensor batch1, Tensor batch2, float beta, float alpha) -> ()");
  m.def("my_bernoulli__Tensor(Tensor self, Tensor p, Generator? generator) -> ()");
  m.def("my_bernoulli__float(Tensor self, float p, Generator? generator) -> ()");
  m.def("my_bmm_out(Tensor out, Tensor self, Tensor mat2) -> ()");
  m.def("my_bucketize_Tensor(Tensor self, Tensor boundaries, bool out_int32, bool right) -> Tensor");
  m.def("my_cat(Tensor[] tensors, int dim) -> Tensor");
  m.def("my_cholesky_solve(Tensor self, Tensor input2, bool upper) -> Tensor");
  m.def("my_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor");
  m.def("my_convolution_backward(Tensor grad_output, Tensor input, Tensor weight, int[]? bias_sizes, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, int[] output_mask) -> (Tensor, Tensor, Tensor)");
  m.def("my_cummax(Tensor self, int dim) -> (Tensor, Tensor)");
  m.def("my_cummin(Tensor self, int dim) -> (Tensor, Tensor)");
  m.def("my_cumsum(Tensor self, int dim, ScalarType? dtype) -> Tensor");
  m.def("my_exponential(Tensor self, float lambd, Generator? generator) -> Tensor");
  m.def("my_fill__Scalar(Tensor self, float value) -> ()");
  m.def("my_grid_sampler_2d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, bool[] output_mask) -> (Tensor, Tensor)");
  m.def("my_hann_window(int window_length, ScalarType? dtype, Layout? layout, Device? device, bool? pin_memory) -> Tensor");
  m.def("my_histc(Tensor self, int bins, float min, float max) -> Tensor");
  m.def("my_index_Tensor(Tensor self, Tensor[] indices) -> Tensor");
  m.def("my_index_put(Tensor self, Tensor[] indices, Tensor values, bool accumulate) -> Tensor");
  m.def("my_kthvalue(Tensor self, int k, int dim, bool keepdim) -> (Tensor, Tensor)");
  m.def("my_logcumsumexp(Tensor self, int dim) -> Tensor");
  m.def("my_masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor");
  m.def("my_masked_scatter_backward(Tensor grad_output, Tensor mask, int[] sizes) -> Tensor");
  m.def("my_masked_select(Tensor self, Tensor mask) -> Tensor");
  m.def("my_max_pool2d_with_indices(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> (Tensor, Tensor)");
  m.def("my_max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode, Tensor indices) -> Tensor");
  m.def("my_max_pool3d_with_indices(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> (Tensor, Tensor)");
  m.def("my_max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode, Tensor indices) -> Tensor");
  m.def("my_max_unpool2d(Tensor self, Tensor indices, int[] output_size) -> Tensor");
  m.def("my_max_unpool3d(Tensor self, Tensor indices, int[] output_size, int[] stride, int[] padding) -> Tensor");
  m.def("my_median(Tensor self) -> Tensor");
  m.def("my_mm_out(Tensor out, Tensor self, Tensor mat2) -> ()");
  m.def("my_mul_Scalar(Tensor self, float other) -> Tensor");
  m.def("my_mul_Tensor(Tensor self, Tensor other) -> Tensor");
  m.def("my_nanmedian(Tensor self) -> Tensor");
  m.def("my_narrow(Tensor self, int dim, int start, int length) -> Tensor");
  m.def("my_native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)");
  m.def("my_nonzero(Tensor self) -> Tensor");
  m.def("my_normal_functional(Tensor self, float mean, float std, Generator? generator) -> Tensor");
  m.def("my_pad(Tensor self, int[] pad, str mode, float? value) -> Tensor");
  m.def("my_permute(Tensor self, int[] dims) -> Tensor");
  m.def("my_polar(Tensor abs, Tensor angle) -> Tensor");
  m.def("my_pow_Scalar(float self, Tensor exponent) -> Tensor");
  m.def("my_pow_Tensor_Scalar(Tensor self, float exponent) -> Tensor");
  m.def("my_pow_Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor");
  m.def("my_rand(int[] size, ScalarType? dtype, Layout? layout, Device? device, bool? pin_memory) -> Tensor");
  m.def("my_rand_generator(int[] size, Generator? generator, ScalarType? dtype, Layout? layout, Device? device, bool? pin_memory) -> Tensor");
  m.def("my_randint(int high, int[] size, ScalarType? dtype, Layout? layout, Device? device, bool? pin_memory) -> Tensor");
  m.def("my_randint_generator(int high, int[] size, Generator? generator, ScalarType? dtype, Layout? layout, Device? device, bool? pin_memory) -> Tensor");
  m.def("my_randint_low(int low, int high, int[] size, ScalarType? dtype, Layout? layout, Device? device, bool? pin_memory) -> Tensor");
  m.def("my_randint_low_out(Tensor out, int low, int high, int[] size) -> ()");
  m.def("my_randn(int[] size, ScalarType? dtype, Layout? layout, Device? device, bool? pin_memory) -> Tensor");
  m.def("my_randn_generator(int[] size, Generator? generator, ScalarType? dtype, Layout? layout, Device? device, bool? pin_memory) -> Tensor");
  m.def("my_randperm(int n, ScalarType? dtype, Layout? layout, Device? device, bool? pin_memory) -> Tensor");
  m.def("my_replication_pad1d_backward(Tensor grad_output, Tensor self, int[] padding) -> Tensor");
  m.def("my_replication_pad2d_backward(Tensor grad_output, Tensor self, int[] padding) -> Tensor");
  m.def("my_reshape(Tensor self, int[] shape) -> Tensor");
  m.def("my_resize_(Tensor self, int[] size, bool? memory_format) -> ()");
  m.def("my_resize_as_(Tensor self, Tensor the_template, bool? memory_format) -> ()");
  m.def("my_scatter_src_out(Tensor out, Tensor self, int dim, Tensor index, Tensor src) -> ()");
  m.def("my_scatter_value_out(Tensor out, Tensor self, int dim, Tensor index, float value) -> ()");
  m.def("my_searchsorted_Scalar(Tensor sorted_sequence, float self, bool out_int32, bool right, Tensor? sorter) -> Tensor");
  m.def("my_searchsorted_Tensor(Tensor sorted_sequence, Tensor self, bool out_int32, bool right, Tensor? sorter) -> Tensor");
  m.def("my_set__source_Tensor(Tensor self, Tensor source) -> ()");
  m.def("my_slice_Tensor(Tensor self, int dim, int? start, int? end, int step) -> Tensor");
  m.def("my_soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor");
  m.def("my_sort(Tensor self, int dim, bool descending) -> (Tensor, Tensor)");
  m.def("my_sort_stable(Tensor self, bool? stable, int dim, bool descending) -> (Tensor, Tensor)");
  m.def("my_squeeze_dim(Tensor self, int dim) -> Tensor");
  m.def("my_to_sparse(Tensor self, Layout? layout, int[]? blocksize, int? dense_dim) -> Tensor");
  m.def("my_topk(Tensor self, int k, int dim, bool largest, bool sorted) -> (Tensor, Tensor)");
  m.def("my_uniform(Tensor self, float from, float to, Generator? generator) -> Tensor");
  m.def("my_upsample_bicubic2d_backward(Tensor grad_output, int[] output_size, int[] input_size, bool align_corners, float? scales_h, float? scales_w) -> Tensor");
  m.def("my_upsample_linear1d_backward(Tensor grad_output, int[] output_size, int[] input_size, bool align_corners, float? scales) -> Tensor");
  m.def("my_upsample_trilinear3d_backward(Tensor grad_output, int[] output_size, int[] input_size, bool align_corners, float? scales_d, float? scales_h, float? scales_w) -> Tensor");
  m.def("my_view_dtype(Tensor self, int dtype) -> Tensor");
  m.def("my_view_as_complex(Tensor self) -> Tensor");
  m.def("my_view_as_real(Tensor self) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_211, CompositeExplicitAutograd, m) {
    m.impl("my__adaptive_avg_pool2d", TORCH_BOX(&my__adaptive_avg_pool2d));
    m.impl("my__adaptive_avg_pool2d_backward", TORCH_BOX(&my__adaptive_avg_pool2d_backward));
    m.impl("my__adaptive_avg_pool3d", TORCH_BOX(&my__adaptive_avg_pool3d));
    m.impl("my__adaptive_avg_pool3d_backward", TORCH_BOX(&my__adaptive_avg_pool3d_backward));
    m.impl("my__cdist_backward", TORCH_BOX(&my__cdist_backward));
    m.impl("my__cdist_forward", TORCH_BOX(&my__cdist_forward));
    m.impl("my__embedding_bag", TORCH_BOX(&my__embedding_bag));
    m.impl("my__embedding_bag_forward_only", TORCH_BOX(&my__embedding_bag_forward_only));
    m.impl("my__embedding_bag_per_sample_weights_backward", TORCH_BOX(&my__embedding_bag_per_sample_weights_backward));
    m.impl("my__fft_c2c", TORCH_BOX(&my__fft_c2c));
    m.impl("my__fft_r2c", TORCH_BOX(&my__fft_r2c));
    m.impl("my__fused_moving_avg_obs_fq_helper_functional", TORCH_BOX(&my__fused_moving_avg_obs_fq_helper_functional));
    m.impl("my__fused_rms_norm", TORCH_BOX(&my__fused_rms_norm));
    m.impl("my__pdist_forward", TORCH_BOX(&my__pdist_forward));
    m.impl("my__scaled_dot_product_fused_attention_overrideable", TORCH_BOX(&my__scaled_dot_product_fused_attention_overrideable));
    m.impl("my__scaled_dot_product_fused_attention_overrideable_backward", TORCH_BOX(&my__scaled_dot_product_fused_attention_overrideable_backward));
    m.impl("my__thnn_fused_lstm_cell", TORCH_BOX(&my__thnn_fused_lstm_cell));
    m.impl("my__trilinear", TORCH_BOX(&my__trilinear));
    m.impl("my_abs", TORCH_BOX(&my_abs));
    m.impl("my_adaptive_max_pool2d", TORCH_BOX(&my_adaptive_max_pool2d));
    m.impl("my_adaptive_max_pool2d_backward", TORCH_BOX(&my_adaptive_max_pool2d_backward));
    m.impl("my_adaptive_max_pool3d", TORCH_BOX(&my_adaptive_max_pool3d));
    m.impl("my_adaptive_max_pool3d_backward", TORCH_BOX(&my_adaptive_max_pool3d_backward));
    m.impl("my_add_Scalar", TORCH_BOX(&my_add_Scalar));
    m.impl("my_add_Tensor", TORCH_BOX(&my_add_Tensor));
    m.impl("my_addbmm", TORCH_BOX(&my_addbmm));
    m.impl("my_addmm_out", TORCH_BOX(&my_addmm_out));
    m.impl("my_addmv", TORCH_BOX(&my_addmv));
    m.impl("my_angle", TORCH_BOX(&my_angle));
    m.impl("my_avg_pool2d", TORCH_BOX(&my_avg_pool2d));
    m.impl("my_avg_pool2d_backward", TORCH_BOX(&my_avg_pool2d_backward));
    m.impl("my_avg_pool3d", TORCH_BOX(&my_avg_pool3d));
    m.impl("my_avg_pool3d_backward", TORCH_BOX(&my_avg_pool3d_backward));
    m.impl("my_baddbmm_out", TORCH_BOX(&my_baddbmm_out));
    m.impl("my_bernoulli__Tensor", TORCH_BOX(&my_bernoulli__Tensor));
    m.impl("my_bernoulli__float", TORCH_BOX(&my_bernoulli__float));
    m.impl("my_bmm_out", TORCH_BOX(&my_bmm_out));
    m.impl("my_bucketize_Tensor", TORCH_BOX(&my_bucketize_Tensor));
    m.impl("my_cat", TORCH_BOX(&my_cat));
    m.impl("my_cholesky_solve", TORCH_BOX(&my_cholesky_solve));
    m.impl("my_convolution", TORCH_BOX(&my_convolution));
    m.impl("my_convolution_backward", TORCH_BOX(&my_convolution_backward));
    m.impl("my_cummax", TORCH_BOX(&my_cummax));
    m.impl("my_cummin", TORCH_BOX(&my_cummin));
    m.impl("my_cumsum", TORCH_BOX(&my_cumsum));
    m.impl("my_exponential", TORCH_BOX(&my_exponential));
    m.impl("my_fill__Scalar", TORCH_BOX(&my_fill__Scalar));
    m.impl("my_grid_sampler_2d_backward", TORCH_BOX(&my_grid_sampler_2d_backward));
    m.impl("my_hann_window", TORCH_BOX(&my_hann_window));
    m.impl("my_histc", TORCH_BOX(&my_histc));
    m.impl("my_index_Tensor", TORCH_BOX(&my_index_Tensor));
    m.impl("my_index_put", TORCH_BOX(&my_index_put));
    m.impl("my_kthvalue", TORCH_BOX(&my_kthvalue));
    m.impl("my_logcumsumexp", TORCH_BOX(&my_logcumsumexp));
    m.impl("my_masked_scatter", TORCH_BOX(&my_masked_scatter));
    m.impl("my_masked_scatter_backward", TORCH_BOX(&my_masked_scatter_backward));
    m.impl("my_masked_select", TORCH_BOX(&my_masked_select));
    m.impl("my_max_pool2d_with_indices", TORCH_BOX(&my_max_pool2d_with_indices));
    m.impl("my_max_pool2d_with_indices_backward", TORCH_BOX(&my_max_pool2d_with_indices_backward));
    m.impl("my_max_pool3d_with_indices", TORCH_BOX(&my_max_pool3d_with_indices));
    m.impl("my_max_pool3d_with_indices_backward", TORCH_BOX(&my_max_pool3d_with_indices_backward));
    m.impl("my_max_unpool2d", TORCH_BOX(&my_max_unpool2d));
    m.impl("my_max_unpool3d", TORCH_BOX(&my_max_unpool3d));
    m.impl("my_median", TORCH_BOX(&my_median));
    m.impl("my_mm_out", TORCH_BOX(&my_mm_out));
    m.impl("my_mul_Scalar", TORCH_BOX(&my_mul_Scalar));
    m.impl("my_mul_Tensor", TORCH_BOX(&my_mul_Tensor));
    m.impl("my_nanmedian", TORCH_BOX(&my_nanmedian));
    m.impl("my_narrow", TORCH_BOX(&my_narrow));
    m.impl("my_native_dropout", TORCH_BOX(&my_native_dropout));
    m.impl("my_nonzero", TORCH_BOX(&my_nonzero));
    m.impl("my_normal_functional", TORCH_BOX(&my_normal_functional));
    m.impl("my_pad", TORCH_BOX(&my_pad));
    m.impl("my_permute", TORCH_BOX(&my_permute));
    m.impl("my_polar", TORCH_BOX(&my_polar));
    m.impl("my_pow_Scalar", TORCH_BOX(&my_pow_Scalar));
    m.impl("my_pow_Tensor_Scalar", TORCH_BOX(&my_pow_Tensor_Scalar));
    m.impl("my_pow_Tensor_Tensor", TORCH_BOX(&my_pow_Tensor_Tensor));
    m.impl("my_rand", TORCH_BOX(&my_rand));
    m.impl("my_rand_generator", TORCH_BOX(&my_rand_generator));
    m.impl("my_randint", TORCH_BOX(&my_randint));
    m.impl("my_randint_generator", TORCH_BOX(&my_randint_generator));
    m.impl("my_randint_low", TORCH_BOX(&my_randint_low));
    m.impl("my_randint_low_out", TORCH_BOX(&my_randint_low_out));
    m.impl("my_randn", TORCH_BOX(&my_randn));
    m.impl("my_randn_generator", TORCH_BOX(&my_randn_generator));
    m.impl("my_randperm", TORCH_BOX(&my_randperm));
    m.impl("my_replication_pad1d_backward", TORCH_BOX(&my_replication_pad1d_backward));
    m.impl("my_replication_pad2d_backward", TORCH_BOX(&my_replication_pad2d_backward));
    m.impl("my_reshape", TORCH_BOX(&my_reshape));
    m.impl("my_resize_", TORCH_BOX(&my_resize_));
    m.impl("my_resize_as_", TORCH_BOX(&my_resize_as_));
    m.impl("my_scatter_src_out", TORCH_BOX(&my_scatter_src_out));
    m.impl("my_scatter_value_out", TORCH_BOX(&my_scatter_value_out));
    m.impl("my_searchsorted_Scalar", TORCH_BOX(&my_searchsorted_Scalar));
    m.impl("my_searchsorted_Tensor", TORCH_BOX(&my_searchsorted_Tensor));
    m.impl("my_set__source_Tensor", TORCH_BOX(&my_set__source_Tensor));
    m.impl("my_slice_Tensor", TORCH_BOX(&my_slice_Tensor));
    m.impl("my_soft_margin_loss_backward", TORCH_BOX(&my_soft_margin_loss_backward));
    m.impl("my_sort", TORCH_BOX(&my_sort));
    m.impl("my_sort_stable", TORCH_BOX(&my_sort_stable));
    m.impl("my_squeeze_dim", TORCH_BOX(&my_squeeze_dim));
    m.impl("my_to_sparse", TORCH_BOX(&my_to_sparse));
    m.impl("my_topk", TORCH_BOX(&my_topk));
    m.impl("my_uniform", TORCH_BOX(&my_uniform));
    m.impl("my_upsample_bicubic2d_backward", TORCH_BOX(&my_upsample_bicubic2d_backward));
    m.impl("my_upsample_linear1d_backward", TORCH_BOX(&my_upsample_linear1d_backward));
    m.impl("my_upsample_trilinear3d_backward", TORCH_BOX(&my_upsample_trilinear3d_backward));
    m.impl("my_view_dtype", TORCH_BOX(&my_view_dtype));
    m.impl("my_view_as_complex", TORCH_BOX(&my_view_as_complex));
    m.impl("my_view_as_real", TORCH_BOX(&my_view_as_real));
}