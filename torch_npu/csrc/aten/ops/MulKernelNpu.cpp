// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#include "torch_npu/csrc/register/OptionsManager.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    at::Tensor mul_dest_output(const at::Tensor &self, const at::Tensor &other)
    {
      bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);
      return isSelfWrapped ? other : self;
    }

    at::Tensor &muls_out_npu(at::Tensor &result, const at::Tensor &self, const at::Scalar other)
    {
      auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
      OpCommand cmd;
      if (torch_npu::option::OptionsManager::CheckDynamicOptimizer("MUL"))
      {
        cmd.Name("Mul")
            .Expect(unified_result)
            .Input(self)
            .Input(other, self.scalar_type())
            .Output(result)
            .Run();
      }
      else
      {
        cmd.Name("Muls")
            .Expect(unified_result)
            .Input(self)
            .Output(result)
            .Attr("value", other)
            .Run();
      }

      return result;
    }

    at::Tensor &mul_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &other)
    {
      if (other.dim() == 0 && !other.is_npu())
      {
        muls_out_npu(result, self, other.item());
      }
      else if (self.dim() == 0 && !self.is_npu())
      {
        muls_out_npu(result, other, self.item());
      }
      else
      {
        auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
        OpCommand cmd;
        cmd.Name("Mul")
            .Expect(unified_result)
            .Input(self)
            .Input(other)
            .Output(result)
            .Run();
      }

      return result;
    }

    at::Tensor &NPUNativeFunctions::mul_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &result)
    {
      // calculate the output size
      at::Tensor outputTensor = mul_dest_output(self, other);
      auto outputSize = broadcast_ops_npu_output_size(self, other);
      OpPreparation::CheckOut(
          {self},
          result,
          CalcuOpUtil::get_tensor_npu_format(outputTensor),
          self.scalar_type(),
          outputSize);
      mul_out_npu_nocheck(result, self, other);

      return result;
    }

    at::Tensor NPUNativeFunctions::mul(const at::Tensor &self, const at::Tensor &other)
    {
      at::Tensor selfCast = self;
      at::Tensor otherCast = other;
      if (self.dtype() == c10::ScalarType::Bool && other.dtype() == c10::ScalarType::Bool)
      {
        selfCast = self.to(c10::ScalarType::Float);
        otherCast = other.to(c10::ScalarType::Float);
      }

      // calculate the output size
      at::Tensor outputTensor = mul_dest_output(selfCast, otherCast);
      auto outputSize = broadcast_ops_npu_output_size(selfCast, otherCast);

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize,
          outputTensor.options(),
          CalcuOpUtil::get_tensor_npu_format(outputTensor));

      // calculate the output result of the NPU
      mul_out_npu_nocheck(result, selfCast, otherCast);

      if (self.dtype() == c10::ScalarType::Bool && other.dtype() == c10::ScalarType::Bool)
      {
        result = result.to(c10::ScalarType::Bool);
      }

      return result;
    }

    at::Tensor NPUNativeFunctions::mul(const at::Tensor &self, at::Scalar other)
    {
      // calculate the output size
      auto outputSize = input_same_output_size(self);

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

      // calculate the output result of the NPU
      muls_out_npu(result, self, other);

      return result;
    }

    at::Tensor &NPUNativeFunctions::mul_(at::Tensor &self, const at::Tensor &other)
    {
      TORCH_CHECK(self.is_npu(), "Input1 must be NPU-Tensor");

      c10::SmallVector<at::Tensor, N> inputs = {self, other};
      c10::SmallVector<at::Tensor, N> outputs = {self};
      CalcuOpUtil::check_memory_over_laps(inputs, outputs);

      if (!NpuUtils::check_match(&self))
      {
        at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
        at::Tensor result = mul_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
        NpuUtils::format_fresh_view(self, result);
      }
      else
      {
        mul_out_npu_nocheck(self, self, other);
      }

      return self;
    }

    at::Tensor &NPUNativeFunctions::mul_(at::Tensor &self, at::Scalar other)
    {
      if (!NpuUtils::check_match(&self))
      {
        at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
        at::Tensor result = muls_out_npu(contiguousSelf, contiguousSelf, other);
        NpuUtils::format_fresh_view(self, result);
      }
      else
      {
        muls_out_npu(self, self, other);
      }

      return self;
    }

  } // namespace native
} // namespace at_npu