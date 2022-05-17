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
      cmd.Name("RealDiv")
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
      if (other.dim() == 0 && !at_npu::key::isDeviceTensor(other))
      {
        div_scalar_out_npu(self, other.item(), result);
      }
      else
      {
        auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
        OpCommand cmd;
        cmd.Name("RealDiv")
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
      // calculate the output size
      at::Tensor outputTensor = CalcuOpUtil::is_scalar_wrapped_to_tensor(self) ? other : self;
      auto outputSize = broadcast_ops_npu_output_size(self, other);
      OpPreparation::CheckOut(
          {self},
          result,
          CalcuOpUtil::get_tensor_npu_format(outputTensor),
          self.scalar_type(),
          outputSize);
      div_out_npu_nocheck(self, other, result);

      return result;
    }

    at::Tensor NPUNativeFunctions::div(const at::Tensor &self, const at::Tensor &other)
    {
      // calculate the output size
      bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);
      at::Tensor outputTensor = isSelfWrapped ? other : self;

      auto outputSize = broadcast_ops_npu_output_size(self, other);

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize,
          outputTensor.options(),
          CalcuOpUtil::get_tensor_npu_format(outputTensor));

      // calculate the output result of the NPU
      div_out_npu_nocheck(self, other, result);

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

    at::Tensor NPUNativeFunctions::div(
        const at::Tensor& self, 
        const at::Tensor& other,
        std::string rounding_mode) {
      if (rounding_mode == "floor") {
        return NPUNativeFunctions::floor_divide(self, other);
      }
      at::Tensor true_div_res = NPUNativeFunctions::div(self, other);
      if (rounding_mode == "true") {
        return true_div_res;
      } else if (rounding_mode == "trunc") {
        return NPUNativeFunctions::trunc(true_div_res);
      }

      TORCH_CHECK(false,
          "div expected rounding_mode to be one of 'true', 'trunc', or 'floor' "
          "but found '", rounding_mode, "'");
    }

    at::Tensor& NPUNativeFunctions::div_(
        at::Tensor& self, 
        const at::Tensor& other,
        std::string rounding_mode) {
      if (rounding_mode == "floor") {
        return NPUNativeFunctions::floor_divide_(self, other);
      }
      NPUNativeFunctions::div_(self, other);
      if (rounding_mode == "true") {
        return self;
      } else if (rounding_mode == "trunc") {
        return NPUNativeFunctions::trunc_(self);
      }

      TORCH_CHECK(false,
          "div expected rounding_mode to be one of 'true', 'trunc', or 'floor' "
          "but found '", rounding_mode, "'");
    }

    at::Tensor& NPUNativeFunctions::div_out(
        const at::Tensor& self, 
        const at::Tensor& other,
        std::string rounding_mode,
        at::Tensor& result) {
      TORCH_CHECK((rounding_mode == "true" || rounding_mode == "trunc" || rounding_mode == "floor"),
          "div expected rounding_mode to be one of 'true', 'trunc', or 'floor' "
          "but found '", rounding_mode, "'");

      if (rounding_mode == "floor") {
        NPUNativeFunctions::floor_divide_out(self, other, result);
        return result;
      }
      NPUNativeFunctions::div_out(self, other, result);
      if (rounding_mode == "trunc") {
        NPUNativeFunctions::trunc_(result);
      }

      return result;
    }

    at::Tensor NPUNativeFunctions::div(
        const at::Tensor& self, 
        at::Scalar other,
        std::string rounding_mode) {
      TORCH_CHECK((rounding_mode == "true" || rounding_mode == "trunc" || rounding_mode == "floor"),
          "div expected rounding_mode to be one of 'true', 'trunc', or 'floor' "
          "but found '", rounding_mode, "'");

      if (rounding_mode == "floor") {
        return NPUNativeFunctions::floor_divide(self, other);
      }
      at::Tensor true_div_res = NPUNativeFunctions::div(self, other);
      if (rounding_mode == "true") {
        return true_div_res;
      } else if (rounding_mode == "trunc") {
        return NPUNativeFunctions::trunc(true_div_res);
      }
    }

    at::Tensor& NPUNativeFunctions::div_(
        at::Tensor& self, 
        at::Scalar other,
        std::string rounding_mode) {
      if (rounding_mode == "floor") {
        return NPUNativeFunctions::floor_divide_(self, other);
      }
      NPUNativeFunctions::div_(self, other);
      if (rounding_mode == "true") {
        return self;
      } else if (rounding_mode == "trunc") {
        return NPUNativeFunctions::trunc_(self);
      }

      TORCH_CHECK(false,
          "div expected rounding_mode to be one of 'true', 'trunc', or 'floor' "
          "but found '", rounding_mode, "'");
    }

  } // namespace native
} // namespace at_npu