#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

namespace {

Tensor& embedding_dense_backward_nocheck(
    Tensor& result,
    const Tensor& grad_output,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  // indices must be int64 in pytorch, but npu can only support int32
  auto indices_int32 = indices.to(at::kInt);

  OpCommand cmd;
  cmd.Name("EmbeddingDenseGrad")
      .Input(grad_output)
      .Input(indices_int32)
      .Attr("num_weights", num_weights)
      .Attr("padding_idx", padding_idx)
      .Attr("scale_grad_by_freq", scale_grad_by_freq)
      .Output(result)
      .Run();
  return result;
}
} // namespace

Tensor embedding_dense_backward_npu(
    const Tensor& grad_weight,
    const Tensor& indices, 
    int64_t num_weights, 
    int64_t padding_idx, 
    bool scale_grad_by_freq) {        
    // calculate the output size
    auto outputSize = {num_weights, grad_weight.size(-1)};

    // construct the output tensor of the NPU
    Tensor result = OpPreparation::ApplyTensor(grad_weight, outputSize);

    // calculate the output resugt of the NPU
    embedding_dense_backward_nocheck(
        result, grad_weight, indices, num_weights, padding_idx, scale_grad_by_freq);

    return result;
}

} // namespace native
} // namespace at