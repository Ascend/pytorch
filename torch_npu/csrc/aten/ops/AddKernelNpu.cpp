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

#include <ATen/Tensor.h>
#include <c10/util/SmallVector.h>

#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"

namespace at_npu
{
  namespace native
  {

    inline void alpha_check_npu(const at::ScalarType dtype, at::Scalar alpha)
    {
      TORCH_CHECK(
          !alpha.isBoolean() || dtype == at::ScalarType::Bool,
          "Boolean alpha only supported for Boolean results.");
      TORCH_CHECK(
          isFloatingType(dtype) || alpha.isIntegral(true),
          "For integral input tensors, argument alpha must not be a floating point number.");
    }

    at::Tensor add_dest_output(const at::Tensor &self, const at::Tensor &other)
    {
      bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);
      return isSelfWrapped ? other : self;
    }

    at::Tensor &adds_out_npu_nocheck(
        at::Tensor &result,
        const at::Tensor &self,
        const at::Scalar other,
        const at::Scalar alpha)
    {
      // constructs the input and output NPUTensorDesc
      alpha_check_npu(self.scalar_type(), alpha);
      float otherValue = CalcuOpUtil::get_scalar_float_value(other);
      float alphaValue = CalcuOpUtil::get_scalar_float_value(alpha);
      float value = otherValue * alphaValue;
      OpCommand cmd;
      std::string real_type = "";
      if (self.scalar_type() == c10::ScalarType::Bool)
      {
        auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
        if (unified_result.common_type == c10::ScalarType::Bool)
        {
          unified_result.common_type = c10::ScalarType::Byte;
          unified_result.result_type_defined = true;
          real_type = "uint8";
        }
        cmd.Expect(unified_result);
      }
      cmd.Name("Add")
          .Input(self)
          .Input(value, self.scalar_type())
          .Output(result, real_type)
          .Run();

      return result;
    }

    at::Tensor &add_out_npu_nocheck(
        at::Tensor &result,
        const at::Tensor &self,
        const at::Tensor &other,
        at::Scalar alpha)
    {
      auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
      if (other.dim() == 0 && !other.is_npu())
      {
        adds_out_npu_nocheck(result, self, other.item(), alpha);
      }
      else if (self.dim() == 0 && !self.is_npu())
      {
        adds_out_npu_nocheck(result, other, self.item(), alpha);
      }
      else
      {
        alpha_check_npu(self.scalar_type(), alpha);
        OpCommand cmd;
        cmd.Expect(unified_result);
        // executing the NPU operator
        if (CalcuOpUtil::is_scalar_one(alpha))
        {
          if (self.scalar_type() == at::kLong)
          {
            TORCH_WARN_ONCE("The oprator of add is executed, Currently High Accuracy but Low Performance OP with 64-bit has been used,"
                            "Please Do Some Cast at Python Functions with 32-bit for Better Performance!");
          }

          std::string real_type = "";
          if (self.scalar_type() == c10::ScalarType::Bool && other.scalar_type() == c10::ScalarType::Bool)
          {
            unified_result.common_type = c10::ScalarType::Byte;
            unified_result.result_type_defined = true;
            cmd.Expect(unified_result);
            real_type = "uint8";
          }
          cmd.Name("Add")
              .Input(self)
              .Input(other)
              .Output(result, real_type)
              .Run();
        }
        else
        {
          if (torch_npu::option::OptionsManager::CheckDynamicOptimizer("ADD"))
          {
            cmd.Name("AxpyV2")
                .Input(self)
                .Input(other)
                .Input(alpha, self.scalar_type())
                .Output(result)
                .Run();
          }
          else
          {
            cmd.Name("Axpy")
                .Input(self)
                .Input(other)
                .Attr("alpha", alpha)
                .Output(result)
                .Run();
          }
        }
      }

      return result;
    }

    bool check_size(const at::Tensor &self, const at::Tensor &other)
    {
      if (self.dim() != other.dim())
      {
        return false;
      }
      for (size_t i = 0; i < self.dim(); i++)
      {
        if (self.size(i) != other.size(i))
        {
          return false;
        }
      }
      return true;
    }

    at::Tensor stride_add_tensor_get(const at::Tensor &src)
    {
      if (src.is_contiguous())
      {
        return src;
      }
      else
      {
        auto src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
        at::Tensor src_new = OpPreparation::ApplyTensorWithFormat(
            src_desc.base_sizes_, src.options(), ACL_FORMAT_NC1HWC0);
        src_new.set_(
            src.storage(),
            src_new.storage_offset(),
            src_new.sizes(),
            src_new.strides());
        return src_new;
      }
    }

    at::Tensor NPUNativeFunctions::add(const at::Tensor &self, const at::Tensor &other, at::Scalar alpha)
    {
      alpha_check_npu(self.scalar_type(), alpha);
      if ((!(self.is_contiguous() && other.is_contiguous())) &&
          (NpuUtils::check_5d_5d_match(self) ||
           NpuUtils::check_5d_5d_match(other)) &&
          check_size(self, other))
      {
        int64_t c0_len = 16;
        at::Tensor self_use = stride_add_tensor_get(self);
        at::Scalar self_c1_offset(
            self.storage_offset() / (self.size(2) * self.size(3) * c0_len));
        at::Tensor other_use = stride_add_tensor_get(other);
        at::Scalar other_c1_offset(
            other.storage_offset() / (other.size(2) * other.size(3) * c0_len));
        at::Scalar stride_len(self.size(1) / c0_len);
        at::Tensor result = NPUNativeFunctions::npu_stride_add(
            self_use, other_use, self_c1_offset, other_c1_offset, stride_len);
        return result;
      }
      // calculate the output size
      at::Tensor outputTensor = add_dest_output(self, other);
      auto outputSize = broadcast_ops_npu_output_size(self, other);

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize,
          outputTensor.options(),
          CalcuOpUtil::get_tensor_npu_format(outputTensor));

      // calculate the output result of the NPU
      add_out_npu_nocheck(result, self, other, alpha);

      return result;
    }

    at::Tensor NPUNativeFunctions::add(const at::Tensor &self, at::Scalar other, at::Scalar alpha)
    {
      alpha_check_npu(self.scalar_type(), alpha);
      // calculate the output size
      auto outputSize = input_same_output_size(self);
      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

      // calculate the output result of the NPU
      adds_out_npu_nocheck(result, self, other, alpha);

      return result;
    }

    at::Tensor &NPUNativeFunctions::add_(at::Tensor &self, const at::Tensor &other, at::Scalar alpha)
    {
      c10::SmallVector<at::Tensor, N> inputs = {self, other};
      c10::SmallVector<at::Tensor, N> outputs = {self};
      CalcuOpUtil::check_memory_over_laps(inputs, outputs);

      if (!NpuUtils::check_match(&self))
      {
        at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
        at::Tensor result = add_out_npu_nocheck(contiguousSelf, contiguousSelf, other, alpha);
        NpuUtils::format_fresh_view(self, result);
      }
      else
      {
        add_out_npu_nocheck(self, self, other, alpha);
      }

      return self;
    }

    at::Tensor &NPUNativeFunctions::add_(at::Tensor &self, at::Scalar other, at::Scalar alpha)
    {
      if (!NpuUtils::check_match(&self))
      {
        at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
        at::Tensor result = adds_out_npu_nocheck(contiguousSelf, contiguousSelf, other, alpha);
        NpuUtils::format_fresh_view(self, result);
      }
      else
      {
        adds_out_npu_nocheck(self, self, other, alpha);
      }

      return self;
    }

    at::Tensor &NPUNativeFunctions::add_out(
        const at::Tensor &self,
        const at::Tensor &other,
        at::Scalar alpha,
        at::Tensor &result)
    {
      bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);

      at::Tensor outputTensor;
      if (not isSelfWrapped)
      {
        outputTensor = self;
      }
      else
      {
        outputTensor = other;
      }
      auto outputSize = broadcast_ops_npu_output_size(self, other);
      OpPreparation::CheckOut(
          {self},
          result,
          CalcuOpUtil::get_tensor_npu_format(result),
          outputTensor.scalar_type(),
          outputSize);

      OpPipeWithDefinedOut pipe;
      return pipe.CheckMemory({self, other}, {result})
          .Func([&self, &other, &alpha](at::Tensor &result)
                { add_out_npu_nocheck(result, self, other, alpha); })
          .Call(result);
    }

  }
}