#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::embedding_backward_symint(
    const at::Tensor& grad, 
    const at::Tensor& indices, 
    c10::SymInt num_weights,
    c10::SymInt padding_idx,
    bool scale_grad_by_freq, 
    bool sparse) {
    TORCH_CHECK(sparse == false, "NPU error, not yet support sparse tensor, when sparse == True");

    // run dense tensor backward
    return at::embedding_dense_backward(
        grad,
        indices,
        num_weights.guard_int(__FILE__, __LINE__),
        padding_idx.guard_int(__FILE__, __LINE__),
        scale_grad_by_freq);
}

} // namespace native
} // namespace at