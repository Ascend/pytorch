#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::gelu(const at::Tensor& self, c10::string_view approximate) {
    at::Tensor result = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
    OpCommand cmd;
    cmd.Name("Gelu")
        .Input(self)
        .Output(result)
        .Run();
    
    return result;
}
} // namespace native
} // namespace at_npu 
