#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor NPUNativeFunctions::_embedding_bag_backward_symint(
    const at::Tensor& grad,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const at::Tensor& offset2bag,
    const at::Tensor& bag_size,
    const at::Tensor& maximum_indices,
    c10::SymInt num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    const c10::optional<at::Tensor>& per_sample_weights_opt,
    int64_t padding_idx) {
  const at::Tensor& per_sample_weights = c10::value_or_else(per_sample_weights_opt, [] {return at::Tensor();});

  at::Tensor grad_cpu = grad.to("cpu");
  at::Tensor indices_cpu = indices.to("cpu");
  at::Tensor offsets_cpu = offsets.to("cpu");
  at::Tensor offset2bag_cpu = offset2bag.to("cpu");
  at::Tensor bag_size_cpu = bag_size.to("cpu");
  at::Tensor maximum_indices_cpu = maximum_indices.to("cpu");
  at::Tensor per_sample_weights_cpu = per_sample_weights;
  if (per_sample_weights_cpu.defined()) {
    per_sample_weights_cpu = per_sample_weights_cpu.to("cpu");
  }

  at::Tensor result = at::_embedding_bag_backward_symint(
      grad_cpu, indices_cpu, offsets_cpu, offset2bag_cpu, bag_size_cpu,
      maximum_indices_cpu, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights_cpu);
  
  result = at::native::sparse_to_dense(result);
  result = result.to(indices.device());

  return result;
}
} // namespace native
} // namespace at_npu
