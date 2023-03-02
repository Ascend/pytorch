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
      if (exp.toFloat() == 2.0) {
        NPUNativeFunctions::mul_out(self, self, result);
        return result;
      }
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
    at::Tensor &NPUNativeFunctions::pow_out(const at::Tensor &self, at::Scalar exp, at::Tensor &result)
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
    at::Tensor &NPUNativeFunctions::pow_out(at::Scalar self, const at::Tensor &exp, at::Tensor &result)
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

    at::Tensor NPUNativeFunctions::pow(const at::Tensor &self, at::Scalar exp)
    {
      auto result_type = at::result_type(self, exp);
      at::Tensor result = OpPreparation::ApplyTensor(self, self.options().dtype(result_type));
      at::Tensor self_copy = (self.scalar_type() != result_type) ? NPUNativeFunctions::npu_dtype_cast(self, result_type) : self;
      pow_tensor_scalar_out_npu_nocheck(self_copy, exp, result);
      return result;
    }

    at::Tensor NPUNativeFunctions::pow(at::Scalar self, const at::Tensor &exp)
    {
      auto result_type = at::result_type(exp, self);
      at::Tensor result = OpPreparation::ApplyTensor(exp, exp.options().dtype(result_type));
      at::Tensor exp_copy = (exp.scalar_type() != result_type) ? NPUNativeFunctions::npu_dtype_cast(exp, result_type) : exp;
      pow_scalar_out_npu_nocheck(self, exp_copy, result);
      return result;
    }

    at::Tensor &NPUNativeFunctions::pow_(at::Tensor &self, const at::Tensor &exp)
    {
      NPUNativeFunctions::pow_out(self, exp, self);
      return self;
    }

    at::Tensor &NPUNativeFunctions::pow_(at::Tensor &self, at::Scalar exp)
    {
      NPUNativeFunctions::pow_out(self, exp, self);
      return self;
    }

  } // namespace native
} // namespace at_npu
