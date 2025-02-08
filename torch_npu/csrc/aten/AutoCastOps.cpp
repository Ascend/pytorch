#include <ATen/autocast_mode.h>
#include <iostream>
#include <exception>

namespace {
using namespace at::autocast;

/*******************************
Banned functions
*******************************/

at::Tensor binary_cross_entropy_banned(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, int64_t)
{
    AT_ERROR("torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.\n"
             "Many models use a sigmoid layer right before the binary cross entropy layer.\n"
             "In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits\n"
             "or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are\n"
             "safe to autocast.");
}
TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
    // lower_precision_fp
    KERNEL_PRIVATEUSEONE(_convolution, deprecated, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(_convolution, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(conv1d, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(conv2d, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(conv3d, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(conv_tbc, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(conv_transpose1d, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(conv_transpose2d, input, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(conv_transpose3d, input, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(convolution, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(cudnn_convolution, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(cudnn_convolution_transpose, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(prelu, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(addmm, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(addmv, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(addr, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(matmul, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(einsum, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(mm, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(mv, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(linear, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(addbmm, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(baddbmm, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(bmm, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(chain_matmul, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(linalg_multi_dot, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(_thnn_fused_lstm_cell, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(_thnn_fused_gru_cell, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(lstm_cell, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(gru_cell, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(rnn_tanh_cell, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(rnn_relu_cell, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(_scaled_dot_product_flash_attention, lower_precision_fp)
    KERNEL_PRIVATEUSEONE(scaled_dot_product_attention, lower_precision_fp)

    // fp32
    KERNEL_PRIVATEUSEONE(acos, fp32)
    KERNEL_PRIVATEUSEONE(asin, fp32)
    KERNEL_PRIVATEUSEONE(cosh, fp32)
    KERNEL_PRIVATEUSEONE(erfinv, fp32)
    KERNEL_PRIVATEUSEONE(exp, fp32)
    KERNEL_PRIVATEUSEONE(expm1, fp32)
    KERNEL_PRIVATEUSEONE(log, fp32)
    KERNEL_PRIVATEUSEONE(log10, fp32)
    KERNEL_PRIVATEUSEONE(log2, fp32)
    KERNEL_PRIVATEUSEONE(log1p, fp32)
    KERNEL_PRIVATEUSEONE(reciprocal, fp32)
    KERNEL_PRIVATEUSEONE(rsqrt, fp32)
    KERNEL_PRIVATEUSEONE(sinh, fp32)
    KERNEL_PRIVATEUSEONE(tan, fp32)
    KERNEL_PRIVATEUSEONE(pow, Tensor_Scalar, fp32)
    KERNEL_PRIVATEUSEONE(pow, Tensor_Tensor, fp32)
    KERNEL_PRIVATEUSEONE(pow, Scalar, fp32)
    KERNEL_PRIVATEUSEONE(softplus, fp32)
    KERNEL_PRIVATEUSEONE(layer_norm, fp32)
    KERNEL_PRIVATEUSEONE(native_layer_norm, fp32)
    KERNEL_PRIVATEUSEONE(group_norm, fp32)
    KERNEL_PRIVATEUSEONE(frobenius_norm, dim, fp32)
    KERNEL_PRIVATEUSEONE(nuclear_norm, fp32)
    KERNEL_PRIVATEUSEONE(nuclear_norm, dim, fp32)
    KERNEL_PRIVATEUSEONE(cosine_similarity, fp32)
    KERNEL_PRIVATEUSEONE(poisson_nll_loss, fp32)
    KERNEL_PRIVATEUSEONE(cosine_embedding_loss, fp32)
    KERNEL_PRIVATEUSEONE(nll_loss, fp32)
    KERNEL_PRIVATEUSEONE(nll_loss2d, fp32)
    KERNEL_PRIVATEUSEONE(hinge_embedding_loss, fp32)
    KERNEL_PRIVATEUSEONE(kl_div, fp32)
    KERNEL_PRIVATEUSEONE(l1_loss, fp32)
    KERNEL_PRIVATEUSEONE(smooth_l1_loss, fp32)
    KERNEL_PRIVATEUSEONE(huber_loss, fp32)
    KERNEL_PRIVATEUSEONE(mse_loss, fp32)
    KERNEL_PRIVATEUSEONE(margin_ranking_loss, fp32)
    KERNEL_PRIVATEUSEONE(multilabel_margin_loss, fp32)
    KERNEL_PRIVATEUSEONE(soft_margin_loss, fp32)
    KERNEL_PRIVATEUSEONE(triplet_margin_loss, fp32)
    KERNEL_PRIVATEUSEONE(multi_margin_loss, fp32)
    KERNEL_PRIVATEUSEONE(binary_cross_entropy_with_logits, fp32)
    KERNEL_PRIVATEUSEONE(dist, fp32)
    KERNEL_PRIVATEUSEONE(pdist, fp32)
    KERNEL_PRIVATEUSEONE(cdist, fp32)
    KERNEL_PRIVATEUSEONE(renorm, fp32)
    KERNEL_PRIVATEUSEONE(logsumexp, fp32)
    // fp32_set_opt_dtype
    KERNEL_PRIVATEUSEONE(prod, fp32_set_opt_dtype)
    KERNEL_PRIVATEUSEONE(prod, dim_int, fp32_set_opt_dtype)
    KERNEL_PRIVATEUSEONE(prod, dim_Dimname, fp32_set_opt_dtype)
    KERNEL_PRIVATEUSEONE(softmax, int, fp32_set_opt_dtype)
    KERNEL_PRIVATEUSEONE(softmax, Dimname, fp32_set_opt_dtype)
    KERNEL_PRIVATEUSEONE(log_softmax, int, fp32_set_opt_dtype)
    KERNEL_PRIVATEUSEONE(log_softmax, Dimname, fp32_set_opt_dtype)
    KERNEL_PRIVATEUSEONE(cumprod, fp32_set_opt_dtype)
    KERNEL_PRIVATEUSEONE(cumprod, dimname, fp32_set_opt_dtype)
    KERNEL_PRIVATEUSEONE(cumsum, fp32_set_opt_dtype)
    KERNEL_PRIVATEUSEONE(cumsum, dimname, fp32_set_opt_dtype)
    KERNEL_PRIVATEUSEONE(linalg_vector_norm, fp32_set_opt_dtype)
    KERNEL_PRIVATEUSEONE(linalg_matrix_norm, fp32_set_opt_dtype)
    KERNEL_PRIVATEUSEONE(linalg_matrix_norm, str_ord, fp32_set_opt_dtype)
    // commenting these out because they accept an explicit (not-optional) dtype, and we shouldn't try to flip that even
    // when autocasting.
    KERNEL_PRIVATEUSEONE(sum, fp32_set_opt_dtype)
    KERNEL_PRIVATEUSEONE(sum, dim_IntList, fp32_set_opt_dtype)
    KERNEL_PRIVATEUSEONE(sum, dim_DimnameList, fp32_set_opt_dtype)
    // fp32_append_dtype
    // The fp32_append_dtype wrapper overrides implicit promotion behavior.
    // norm does not implicitly promote, but be aware when adding new ops to this policy.
    KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE(ADD_NS(norm),
        "norm.Scalar", at::Tensor (const at::Tensor &, const c10::Scalar&),
        at::Tensor (const at::Tensor &, const c10::optional<c10::Scalar>&, at::ScalarType),
        fp32_append_dtype)
    KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE(ADD_NS(norm), "norm.ScalarOpt_dim",
        at::Tensor (const at::Tensor &, const c10::optional<c10::Scalar>&, at::IntArrayRef, bool),
        at::Tensor (const at::Tensor &, const c10::optional<c10::Scalar>&, at::IntArrayRef, bool, at::ScalarType),
        fp32_append_dtype)
    KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE(ADD_NS(norm), "norm.names_ScalarOpt_dim",
    at::Tensor (const at::Tensor &, const c10::optional<c10::Scalar>&, at::DimnameList, bool),
        at::Tensor (const at::Tensor &, const c10::optional<c10::Scalar>&, at::DimnameList, bool, at::ScalarType),
        fp32_append_dtype)
    // promote
    KERNEL_PRIVATEUSEONE(addcdiv, promote)
    KERNEL_PRIVATEUSEONE(addcmul, promote)
    KERNEL_PRIVATEUSEONE(atan2, promote)
    KERNEL_PRIVATEUSEONE(bilinear, promote)
    KERNEL_PRIVATEUSEONE(cross, promote)
    KERNEL_PRIVATEUSEONE(dot, promote)
    KERNEL_PRIVATEUSEONE(grid_sampler, promote)
    KERNEL_PRIVATEUSEONE(index_put, promote)
    KERNEL_PRIVATEUSEONE(tensordot, promote)
    KERNEL_PRIVATEUSEONE(scatter_add, promote)

    m.impl(TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
           TORCH_FN((&binary_cross_entropy_banned)));
}

}
