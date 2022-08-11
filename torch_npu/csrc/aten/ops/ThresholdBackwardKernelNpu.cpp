// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/XLANativeFunctions.h"
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
      if (CalcuOpUtil::get_scalar_float_value(threshold) != 0)
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

    at::Tensor XLANativeFunctions::threshold_backward(
        const at::Tensor &grad_output,
        const at::Tensor &self,
        const at::Scalar &threshold)
    {
      // calculate the output size
      auto outputSize = input_same_output_size(self);

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

      // use 5HD in Relu
      if ((torch_npu::NPUBridge::GetNpuStorageImpl(grad_output)->npu_desc_.npu_format_ ==
           ACL_FORMAT_NCHW) &&
          (torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.npu_format_ ==
           ACL_FORMAT_NC1HWC0))
      {
        at::Tensor grad_output_5HD =
            XLANativeFunctions::npu_format_cast(grad_output, ACL_FORMAT_NC1HWC0);
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