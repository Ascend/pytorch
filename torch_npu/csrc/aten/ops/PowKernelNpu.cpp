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
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    // pow.Tensor_Tensor_out
    at::Tensor &pow_tensor_tensor_out_npu_nocheck(const at::Tensor &self, const at::Tensor &exp, at::Tensor &result)
    {
      OpCommand cmd;
      cmd.Name("Pow")
          .Input(self)
          .Input(exp)
          .Output(result)
          .Run();

      return result;
    }

    // pow.Tensor_Scalar_out
    at::Tensor &pow_tensor_scalar_out_npu_nocheck(const at::Tensor &self, at::Scalar exp, at::Tensor &result)
    {
      OpCommand cmd;
      cmd.Name("Pow")
          .Input(self)
          .Input(exp, self.scalar_type())
          .Output(result)
          .Run();

      return result;
    }

    // pow.Scalar_out
    at::Tensor &pow_scalar_out_npu_nocheck(at::Scalar self, const at::Tensor &exp, at::Tensor &result)
    {
      OpCommand cmd;
      cmd.Name("Pow")
          .Input(self, exp.scalar_type())
          .Input(exp)
          .Output(result)
          .Run();

      return result;
    }

    // pow.Tensor_Tensor_out
    at::Tensor &NPUNativeFunctions::pow_out(const at::Tensor &self, const at::Tensor &exp, at::Tensor &result)
    {
      auto outputSize = broadcast_ops_npu_output_size(self, exp);
      OpPreparation::CheckOut(
          {self, exp},
          result,
          self,
          outputSize);

      OpPipeWithDefinedOut pipe;
      return pipe.CheckMemory({self, exp}, {result})
          .Func([&self, &exp](at::Tensor &result)
                { pow_tensor_tensor_out_npu_nocheck(self, exp, result); })
          .Call(result);
    }

    // pow.Tensor_Scalar_out
    at::Tensor &NPUNativeFunctions::pow_out(const at::Tensor &self, const at::Scalar& exp, at::Tensor &result)
    {
      OpPreparation::CheckOut(
          {self},
          result,
          self);

      OpPipeWithDefinedOut pipe;
      return pipe.CheckMemory({self}, {result})
          .Func([&self, &exp](at::Tensor &result)
                { pow_tensor_scalar_out_npu_nocheck(self, exp, result); })
          .Call(result);
    }

    // pow.Scalar_out
    at::Tensor &NPUNativeFunctions::pow_out(const at::Scalar& self, const at::Tensor &exp, at::Tensor &result)
    {
      OpPreparation::CheckOut(
          {exp},
          result,
          exp);

      OpPipeWithDefinedOut pipe;
      return pipe.CheckMemory({exp}, {result})
          .Func([&self, &exp](at::Tensor &result)
                { pow_scalar_out_npu_nocheck(self, exp, result); })
          .Call(result);
    }

    at::Tensor NPUNativeFunctions::pow(const at::Tensor &self, const at::Tensor &exp)
    {
      // calculate the output size
      auto outputSize = broadcast_ops_npu_output_size(self, exp);
      at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
      pow_tensor_tensor_out_npu_nocheck(self, exp, result);
      return result;
    }

    at::Tensor NPUNativeFunctions::pow(const at::Tensor &self, const at::Scalar& exp)
    {
      at::Tensor result = OpPreparation::ApplyTensor(self);
      pow_tensor_scalar_out_npu_nocheck(self, exp, result);
      return result;
    }

    at::Tensor NPUNativeFunctions::pow(const at::Scalar& self, const at::Tensor &exp)
    {
      // calculate the output size
      auto outputSize = input_same_output_size(exp);

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithSizes(outputSize, exp.options());

      // calculate the output result of the NPU
      pow_scalar_out_npu_nocheck(self, exp, result);
      return result;
    }

    at::Tensor &NPUNativeFunctions::pow_(at::Tensor &self, const at::Tensor &exp)
    {
      NPUNativeFunctions::pow_out(self, exp, self);
      return self;
    }

    at::Tensor &NPUNativeFunctions::pow_(at::Tensor &self, const at::Scalar& exp)
    {
      NPUNativeFunctions::pow_out(self, exp, self);
      return self;
    }

  } // namespace native
} // namespace at_npu
