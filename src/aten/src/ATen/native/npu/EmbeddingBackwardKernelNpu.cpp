#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor embedding_backward_npu(
    const Tensor& grad, 
    const Tensor& indices, 
    int64_t num_weights, 
    int64_t padding_idx, 
    bool scale_grad_by_freq, 
    bool sparse) {
    TORCH_CHECK(sparse == false, "NPU error, not yet support sparse tensor, when sparse == True");

    // run dense tensor backward
    return at::embedding_dense_backward(
        grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}

} // namespace native
} // namespace at