#include <ATen/autocast_mode.h>
#include <iostream>
#include <exception>

namespace {
using namespace at::autocast;
using namespace at;

/*******************************
Banned functions
*******************************/

at::Tensor binary_cross_entropy_banned(
    const at::Tensor&,
    const at::Tensor&,
    const c10::optional<at::Tensor>&,
    int64_t) {
  AT_ERROR(
      "torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.\n"
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
#define _KERNEL_PRIVATEUSEONE_LOW_PRECISION_FP(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, lower_precision_fp)

  AT_FORALL_LOWER_PRECISION_FP(_KERNEL_PRIVATEUSEONE_LOW_PRECISION_FP)
  KERNEL_PRIVATEUSEONE(cudnn_convolution, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(cudnn_convolution_transpose, lower_precision_fp)
  // fp32
#define _KERNEL_PRIVATEUSEONE_FP32(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, fp32)

  AT_FORALL_FP32(_KERNEL_PRIVATEUSEONE_FP32)
  // fp32_set_opt_dtype
#define _KERNEL_PRIVATEUSEONE_FP32_SET_OPT_DTYPE(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, fp32_set_opt_dtype)

  AT_FORALL_FP32_SET_OPT_DTYPE(_KERNEL_PRIVATEUSEONE_FP32_SET_OPT_DTYPE)
  // commenting these out because they accept an explicit (not-optional) dtype,
  // and we shouldn't try to flip that even when autocasting.

  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.
  // norm does not implicitly promote, but be aware when adding new ops to this
  // policy.
  AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE)
  // promote
#define _KERNEL_PRIVATEUSEONE_PROMOTE(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, promote)

  AT_FORALL_PROMOTE(_KERNEL_PRIVATEUSEONE_PROMOTE)

  m.impl(TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"), TORCH_FN((&binary_cross_entropy_banned)));
}

} // namespace
