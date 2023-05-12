#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
c10::SmallVector<int64_t, SIZE> _embedding_bag_npu_output_size(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets) {
  c10::SmallVector<int64_t, SIZE> outputSize = {};
  if (indices.dim() == 1) {
    outputSize = {offsets.size(0), weight.size(1)};
  } else {
    outputSize = {indices.size(0), weight.size(1)};
  }
  return outputSize;
}

string get_mode_str(bool mode) {
  string modeStr = "mean";
  if (mode == 0) {
    modeStr = "sum";
  } else if (mode == 1) {
    modeStr = "mean";
  } else {
    modeStr = "max";
  }
  return modeStr;
}

tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::_embedding_bag(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const c10::optional<at::Tensor>& per_sample_weights_opt,
    bool include_last_offset,
    int64_t padding_idx) {
  const at::Tensor& per_sample_weights = c10::value_or_else(per_sample_weights_opt, [] {return at::Tensor();});

  at::Tensor weight_cpu = weight.to("cpu").requires_grad_();
  at::Tensor indices_cpu = indices.to("cpu");
  at::Tensor offsets_cpu = offsets.to("cpu");
  at::Tensor per_sample_weights_cpu = per_sample_weights;
  if (per_sample_weights_cpu.defined()) {
    per_sample_weights_cpu = per_sample_weights_cpu.to("cpu");
  }
  
  auto result = at::native::_embedding_bag_cpu(weight_cpu, indices_cpu, offsets_cpu, scale_grad_by_freq, mode, sparse, per_sample_weights_cpu, include_last_offset);
  
  at::Tensor output = std::get<0>(result).to(weight.device());
  at::Tensor offset2bag = std::get<1>(result).to(weight.device());
  at::Tensor bag_size = std::get<2>(result).to(weight.device());
  at::Tensor max_indices = std::get<3>(result).to(weight.device());
  
  return std::tie(output, offset2bag, bag_size, max_indices);
}
tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::_embedding_bag_forward_only(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const c10::optional<at::Tensor>& per_sample_weights_opt,
    bool include_last_offset,
    int64_t padding_idx) {
  return _embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights_opt, include_last_offset, padding_idx);
}
} // namespace native
} // namespace at_npu
