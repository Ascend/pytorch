#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::flip(const at::Tensor& self, at::IntArrayRef dims){
    at::Tensor result = OpPreparation::ApplyTensor(self);
    at::SmallVector<int64_t,N> dimVec = array_to_small_vector(dims);
    OpCommand cmd;
    cmd.Name("ReverseV2") 
      .Input(self) 
      .Input(dimVec, at::kLong) 
      .Output(result) 
      .Run();
    return result;
}
} // namespace native
} // namespace at_npu