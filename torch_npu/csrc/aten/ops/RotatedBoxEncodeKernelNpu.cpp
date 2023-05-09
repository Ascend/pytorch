#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::npu_rotated_box_encode(
    const at::Tensor& self, 
    const at::Tensor& gtBox, 
    const at::Tensor& weight){
  at::Tensor result = OpPreparation::ApplyTensor(self);
  at::Tensor weightContiguous = weight.to(at::Device(at::kCPU), at::kFloat);
  at::ArrayRef<float> weightList(weightContiguous.data_ptr<float>(), weightContiguous.numel());
  
  OpCommand cmd;
  cmd.Name("RotatedBoxEncode")
      .Input(self)
      .Input(gtBox)
      .Output(result)
      .Attr("weight", weightList)    
      .Run();
  return result;      
}    
} // namespace native
} // namespace at_npu