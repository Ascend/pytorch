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

    at::Tensor &sub_scalar_out_npu(
        at::Tensor &result,
        const at::Tensor &self,
        at::Scalar other,
        at::Scalar alpha)
    {
      // other*alpha
      float otherValue = CalcuOpUtil::get_scalar_float_value(other);
      float alphaValue = CalcuOpUtil::get_scalar_float_value(alpha);
      at::Scalar scalarValue(otherValue * alphaValue);

      OpCommand cmd;
      cmd.Name("Sub")
          .Input(self)
          .Input(scalarValue, self.scalar_type())
          .Output(result)
          .Run();

      return result;
    }

    at::Tensor &sub_out_npu_nocheck(
        at::Tensor &result,
        const at::Tensor &self,
        const at::Tensor &other,
        at::Scalar alpha)
    {
      auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
      if (other.dim() == 0)
      {
        sub_scalar_out_npu(result, self, other.item(), alpha);
      }
      else
      {
        at::Tensor otherMulResult = other;
        if (!CalcuOpUtil::is_scalar_one(alpha))
        {
          otherMulResult = at::mul(other, alpha);
        }

        OpCommand cmd;
        cmd.Name("Sub")
            .Expect(unified_result)
            .Input(self)
            .Input(otherMulResult)
            .Output(result)
            .Run();
      }

      return result;
    }

    at::Tensor &NPUNativeFunctions::sub_out(
        const at::Tensor &self,
        const at::Tensor &other,
        const at::Scalar &alpha,
        at::Tensor &result)
    {
      at::Tensor outputTensor = CalcuOpUtil::is_scalar_wrapped_to_tensor(self) ? other : self;
      auto outputSize = broadcast_ops_npu_output_size(self, other);
      OpPreparation::CheckOut(
          {self},
          result,
          CalcuOpUtil::get_tensor_npu_format(outputTensor),
          self.scalar_type(),
          outputSize);
      sub_out_npu_nocheck(result, self, other, alpha);

      return result;
    }

    at::Tensor NPUNativeFunctions::sub(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha)
    {
      bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);
      at::Tensor outputTensor = isSelfWrapped ? other : self;

      // calculate the output size
      auto outputSize = broadcast_ops_npu_output_size(self, other);

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize,
          outputTensor.options(),
          CalcuOpUtil::get_tensor_npu_format(outputTensor));

      // calculate the output result of the NPU
      sub_out_npu_nocheck(result, self, other, alpha);

      return result;
    }

    at::Tensor NPUNativeFunctions::sub(const at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha)
    {
      // calculate the output size
      auto outputSize = input_same_output_size(self);

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

      // calculate the output result of the NPU
      sub_scalar_out_npu(result, self, other, alpha);

      return result;
    }

    at::Tensor &NPUNativeFunctions::sub_(at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha)
    {
      c10::SmallVector<at::Tensor, N> inputs = {self, other};
      c10::SmallVector<at::Tensor, N> outputs = {self};
      CalcuOpUtil::check_memory_over_laps(inputs, outputs);

      if (!NpuUtils::check_match(&self))
      {
        at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
        at::Tensor result = sub_out_npu_nocheck(contiguousSelf, contiguousSelf, other, alpha);
        NpuUtils::format_fresh_view(self, result);
      }
      else
      {
        sub_out_npu_nocheck(self, self, other, alpha);
      }

      return self;
    }

    at::Tensor &NPUNativeFunctions::sub_(at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha)
    {
      if (!NpuUtils::check_match(&self))
      {
        at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
        at::Tensor result = sub_scalar_out_npu(contiguousSelf, contiguousSelf, other, alpha);
        NpuUtils::format_fresh_view(self, result);
      }
      else
      {
        sub_scalar_out_npu(self, self, other, alpha);
      }

      return self;
    }

  } // namespace native
} // namespace at_npu
