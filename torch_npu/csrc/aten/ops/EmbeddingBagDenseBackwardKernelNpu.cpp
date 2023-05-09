#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::_embedding_bag_dense_backward(
    const at::Tensor& grad,
    const at::Tensor& indices,
    const at::Tensor& offset2bag,
    const at::Tensor& bag_size,
    const at::Tensor& max_indices,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const c10::optional<at::Tensor>& per_sample_weights_opt,
    int64_t padding_idx) {
    const at::Tensor& per_sample_weights = c10::value_or_else(per_sample_weights_opt, [] {return at::Tensor();});

  at::Tensor grad_cpu = grad.to("cpu");
  at::Tensor indices_cpu = indices.to("cpu");
  at::Tensor offset2bag_cpu = offset2bag.to("cpu");
  at::Tensor bag_size_cpu = bag_size.to("cpu");
  at::Tensor max_indices_cpu = max_indices.to("cpu");
  at::Tensor per_sample_weights_opt_cpu = per_sample_weights;
  if (per_sample_weights_opt_cpu.defined()) {
    at::Tensor per_sample_weights_opt_cpu = per_sample_weights_opt_cpu.to("cpu");
  }

  at::Tensor result = at::_embedding_bag_dense_backward(
      grad_cpu, indices_cpu, offset2bag_cpu, bag_size_cpu,
      max_indices_cpu, num_weights, scale_grad_by_freq, mode, per_sample_weights_opt_cpu, padding_idx);

  result = result.to(indices.device());

  return result;
}
} // namespace native
} // namespace at_npu
