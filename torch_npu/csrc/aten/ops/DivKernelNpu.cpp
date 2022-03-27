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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    at::Tensor &div_scalar_out_npu(const at::Tensor &self, const at::Scalar other, at::Tensor &result)
    {
      auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
      OpCommand cmd;
      cmd.Name("Div")
          .Expect(unified_result)
          .Input(self)
          .Input(other, self.scalar_type())
          .Output(result)
          .Run();

      return result;
    }

    at::Tensor &div_out_npu_nocheck(const at::Tensor &self, const at::Tensor &other, at::Tensor &result)
    {

      // executing the NPU operator
      if (other.dim() == 0)
      {
        div_scalar_out_npu(self, other.item(), result);
      }
      else
      {
        auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
        OpCommand cmd;
        cmd.Name("Div")
            .Expect(unified_result)
            .Input(self)
            .Input(other)
            .Output(result)
            .Run();
      }

      return result;
    }

    at::Tensor &NPUNativeFunctions::div_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &result)
    {  
      at::Tensor selfTemp = self;
      at::Tensor otherTemp = other;
      if (self.scalar_type() == at::ScalarType::Int) {
        selfTemp = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
      }

      if (other.scalar_type() == at::ScalarType::Int||other.scalar_type() == at::ScalarType::Bool) {
        otherTemp = NPUNativeFunctions::npu_dtype_cast(other, at::ScalarType::Float);
      }
      // calculate the output size
      at::Tensor outputTensor = CalcuOpUtil::is_scalar_wrapped_to_tensor(selfTemp) ? otherTemp : selfTemp;
      auto outputSize = broadcast_ops_npu_output_size(selfTemp, otherTemp);
      OpPreparation::CheckOut(
          {selfTemp},
          result,
          CalcuOpUtil::get_tensor_npu_format(outputTensor),
          selfTemp.scalar_type(),
          outputSize);
      div_out_npu_nocheck(selfTemp, otherTemp, result);

      return result;
    }

    at::Tensor NPUNativeFunctions::div(const at::Tensor &self, const at::Tensor &other)
    {
      at::Tensor selfTemp = self;
      at::Tensor otherTemp = other;
      if (self.scalar_type() == at::ScalarType::Int) {
        selfTemp = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
      }

      if (other.scalar_type() == at::ScalarType::Int||other.scalar_type() == at::ScalarType::Bool) {
        otherTemp = NPUNativeFunctions::npu_dtype_cast(other, at::ScalarType::Float);
      }
      // calculate the output size
      bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(selfTemp);
      at::Tensor outputTensor = isSelfWrapped ? otherTemp : selfTemp;

      auto outputSize = broadcast_ops_npu_output_size(selfTemp, otherTemp);

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize,
          outputTensor.options(),
          CalcuOpUtil::get_tensor_npu_format(outputTensor));

      // calculate the output result of the NPU
      div_out_npu_nocheck(selfTemp, otherTemp, result);

      return result;
    }

    at::Tensor NPUNativeFunctions::div(const at::Tensor &self, at::Scalar other)
    {
      // calculate the output size
      auto outputSize = input_same_output_size(self);

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize,
          self.options(),
          CalcuOpUtil::get_tensor_npu_format(self));

      // calculate the output result of the NPU
      div_scalar_out_npu(self, other, result);

      return result;
    }

    at::Tensor &NPUNativeFunctions::div_(at::Tensor &self, const at::Tensor &other)
    {
      c10::SmallVector<at::Tensor, N> inputs = {self, other};
      c10::SmallVector<at::Tensor, N> outputs = {self};
      CalcuOpUtil::check_memory_over_laps(inputs, outputs);

      if (!NpuUtils::check_match(&self))
      {
        at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
        at::Tensor result = div_out_npu_nocheck(contiguousSelf, other, contiguousSelf);
        NpuUtils::format_fresh_view(self, result);
      }
      else
      {
        div_out_npu_nocheck(self, other, self);
      }

      return self;
    }

    at::Tensor &NPUNativeFunctions::div_(at::Tensor &self, at::Scalar other)
    {
      if (!NpuUtils::check_match(&self))
      {
        at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);

        div_scalar_out_npu(contiguousSelf, other, contiguousSelf);

        NpuUtils::format_fresh_view(self, contiguousSelf);
      }
      else
      {
        div_scalar_out_npu(self, other, self);
      }
      return self;
    }

  } // namespace native
} // namespace at_npu