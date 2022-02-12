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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/OpTemplate.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    at::Tensor &NPUNativeFunctions::normal_out(
        const at::Tensor &mean,
        double std,
        c10::optional<at::Generator> generator,
        at::Tensor &result)
    {
      TORCH_CHECK(std > 0.0, "normal_ expects std > 0.0, but found std=", std);

      at::Tensor resultCopy = result;
      at::Tensor dtypeCastOfMean = mean;
      if (dtypeCastOfMean.scalar_type() == at::ScalarType::Half)
      {
        dtypeCastOfMean = dtypeCastOfMean.to(at::ScalarType::Float);
        resultCopy = resultCopy.to(at::ScalarType::Float);
      }

      OpCommand cmd;
      cmd.Name("Normal")
          .Input(dtypeCastOfMean)
          .Input(std, at::ScalarType::Float)
          .Output(resultCopy)
          .Run();

      result.copy_(resultCopy);

      return result;
    }

    at::Tensor &NPUNativeFunctions::normal_out(
        double mean,
        const at::Tensor &std,
        c10::optional<at::Generator> generator,
        at::Tensor &result)
    {
      at::Tensor resultCopy = result;
      at::Tensor dtypeCastOfStd = std;
      if (dtypeCastOfStd.scalar_type() == at::ScalarType::Half)
      {
        dtypeCastOfStd = dtypeCastOfStd.to(at::ScalarType::Float);
        resultCopy = resultCopy.to(at::ScalarType::Float);
      }

      OpCommand cmd;
      cmd.Name("Normal")
          .Input(mean, at::ScalarType::Float)
          .Input(dtypeCastOfStd)
          .Output(resultCopy)
          .Run();

      result.copy_(resultCopy);

      return result;
    }

    at::Tensor &NPUNativeFunctions::normal_out(
        const at::Tensor &mean,
        const at::Tensor &std,
        c10::optional<at::Generator> generator,
        at::Tensor &result)
    {
      at::Tensor resultCopy = result;
      at::Tensor dtypeCastOfMean = mean;
      at::Tensor dtypeCastOfStd = std;
      if (dtypeCastOfMean.scalar_type() == at::ScalarType::Half)
      {
        dtypeCastOfMean = dtypeCastOfMean.to(at::ScalarType::Float);
        resultCopy = resultCopy.to(at::ScalarType::Float);
      }
      if (dtypeCastOfStd.scalar_type() == at::ScalarType::Half)
      {
        dtypeCastOfStd = dtypeCastOfStd.to(at::ScalarType::Float);
      }

      OpCommand cmd;
      cmd.Name("Normal")
          .Input(dtypeCastOfMean)
          .Input(dtypeCastOfStd)
          .Output(resultCopy)
          .Run();

      result.copy_(resultCopy);

      return result;
    }

    at::Tensor &NPUNativeFunctions::normal_out(
        double mean,
        double std,
        at::IntArrayRef size,
        c10::optional<at::Generator> generator,
        at::Tensor &result)
    {
      TORCH_CHECK(std > 0.0, "normal_ expects std > 0.0, but found std=", std);

      // the op of PTNormalFloatFloat only support format of ND
      at::Tensor formatCastOfResult = result.npu_format_cast(ACL_FORMAT_ND);
      if (formatCastOfResult.scalar_type() == at::ScalarType::Half)
      {
        formatCastOfResult = formatCastOfResult.to(at::ScalarType::Float);
      }

      at::Tensor meanTensor = OpPreparation::ApplyTensor(size, result.options(), result);
      meanTensor.fill_(mean);
      OpCommand cmd;
      cmd.Name("Normal")
          .Input(meanTensor)
          .Input(std, at::ScalarType::Float)
          .Output(formatCastOfResult)
          .Run();

      result.copy_(formatCastOfResult);

      return result;
    }

    at::Tensor NPUNativeFunctions::normal(
        const at::Tensor &mean,
        double std,
        c10::optional<at::Generator> generator)
    {
      at::Tensor result = OpPreparation::ApplyTensor(mean);
      NPUNativeFunctions::normal_out(mean, std, generator, result);

      return result;
    }

    at::Tensor NPUNativeFunctions::normal(
        double mean,
        const at::Tensor &std,
        c10::optional<at::Generator> generator)
    {
      at::Tensor result = OpPreparation::ApplyTensor(std);
      NPUNativeFunctions::normal_out(mean, std, generator, result);

      return result;
    }

    at::Tensor NPUNativeFunctions::normal(
        const at::Tensor &mean,
        const at::Tensor &std,
        c10::optional<at::Generator> generator)
    {
      at::Tensor result = OpPreparation::ApplyTensor(mean);
      NPUNativeFunctions::normal_out(mean, std, generator, result);

      return result;
    }

    at::Tensor NPUNativeFunctions::normal(
        double mean,
        double std,
        at::IntArrayRef size,
        c10::optional<at::Generator> generator,
        c10::optional<at::ScalarType> dtype_opt,
        c10::optional<c10::Layout> layout_opt,
        c10::optional<c10::Device> device_opt,
        c10::optional<bool> pin_memory_opt)
    {
      // construct the output tensor of the NPU
      at::Tensor result = NPUNativeFunctions::empty_with_format(
          size, dtype_opt, layout_opt, device_opt, pin_memory_opt, ACL_FORMAT_ND);

      // calculate the output result of the NPU
      NPUNativeFunctions::normal_out(mean, std, size, generator, result);

      return result;
    }

    at::Tensor &NPUNativeFunctions::normal_(
        at::Tensor &self,
        double mean,
        double std,
        c10::optional<at::Generator> generator)
    {
      OpPreparation::CheckMemory({self}, {self});
      if (!NpuUtils::check_match(&self))
      {
        at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
        at::Tensor result = NPUNativeFunctions::normal_out(mean, std, contiguousSelf.sizes(), generator, contiguousSelf);
        NpuUtils::format_fresh_view(self, result);
      }
      else
      {
        NPUNativeFunctions::normal_out(mean, std, self.sizes(), generator, self);
      }

      return self;
    }

  } // namespace native
} // namespace at_npu