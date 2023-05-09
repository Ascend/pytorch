#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"

namespace at_npu
{
  namespace native
  {
    at::Tensor threshold_backward_out_npu(
        at::Tensor &result,
        const at::Tensor &grad_output,
        const at::Tensor &self,
        at::Scalar threshold)
    {
      OpCommand cmd;

      // The performance of the ReluGrad operator is better than that of ThresholdGradV2D.
      // However, ReluGrad does not support the scenario where threshold is not 0.
      if (CalcuOpUtil::GetScalarFloatValue(threshold) != 0)
      {
        cmd.Name("ThresholdGradV2D")
            .Input(grad_output)
            .Input(self)
            .Output(result)
            .Attr("threshold", threshold)
            .Run();
      }
      else
      {
        cmd.Name("ReluGrad")
            .Input(grad_output)
            .Input(self)
            .Output(result)
            .Run();
      }

      return result;
    }

    at::Tensor NPUNativeFunctions::threshold_backward(
        const at::Tensor &grad_output,
        const at::Tensor &self,
        const at::Scalar &threshold)
    {
      // calculate the output size
      auto outputSize = input_same_output_size(self);

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize, self.options(), CalcuOpUtil::GetTensorNpuFormat(self));

      // use 5HD in Relu
      if ((torch_npu::NPUBridge::GetNpuStorageImpl(grad_output)->npu_desc_.npu_format_ ==
           ACL_FORMAT_NCHW) &&
          (torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.npu_format_ ==
           ACL_FORMAT_NC1HWC0))
      {
        at::Tensor grad_output_5HD =
            NPUNativeFunctions::npu_format_cast(grad_output, ACL_FORMAT_NC1HWC0);
        threshold_backward_out_npu(result, grad_output_5HD, self, threshold);
        return result;
      }
      else
      {
        threshold_backward_out_npu(result, grad_output, self, threshold);
        return result;
      }
    }

  } // namespace native
} // namespace at_npu