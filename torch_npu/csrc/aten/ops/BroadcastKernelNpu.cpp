#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {
    at::Tensor &NPUNativeFunctions::npu_broadcast_out(
        const at::Tensor &self,
        at::IntArrayRef size,
        at::Tensor &result)
    {
      // executing the NPU operator
      OpCommand cmd;
      cmd.Name("BroadcastTo")
          .Input(self)
          .Input(size)
          .Output(result)
          .Run();
      return result;
    }

    at::Tensor NPUNativeFunctions::npu_broadcast(const at::Tensor &self, at::IntArrayRef size)
    {
      at::Tensor input = self;
      if (self.dtype() == at::kBool)
      {
        input = NPUNativeFunctions::npu_dtype_cast(input, at::kInt);
      }

      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          size,
          input.options(),
          CalcuOpUtil::GetTensorNpuFormat(self));

      NPUNativeFunctions::npu_broadcast_out(input, size, result);

      if (self.dtype() == at::kBool)
      {
        result = NPUNativeFunctions::npu_dtype_cast(result, at::kBool);
      }

      return result;
    }

  } // namespace native
} // namespace at_npu