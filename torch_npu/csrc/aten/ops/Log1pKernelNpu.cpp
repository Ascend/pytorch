#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu { 
namespace native {

at::Tensor& log1p_out_npu_nocheck(const at::Tensor& self, at::Tensor& result){   
  OpCommand cmd;
  cmd.Name("Log1p")
      .Input(self)
      .Output(result)
      .Run();
  return result; 
}

at::Tensor& NPUNativeFunctions::log1p_out(const at::Tensor& self, at::Tensor& result){ 
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self](at::Tensor& result){log1p_out_npu_nocheck(self, result);})
   .Call(result);
}

at::Tensor NPUNativeFunctions::log1p(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  log1p_out_npu_nocheck(self, result);
  return result;
}

at::Tensor& NPUNativeFunctions::log1p_(at::Tensor& self) {
  log1p_out(self, self);
  return self;
}
}}
