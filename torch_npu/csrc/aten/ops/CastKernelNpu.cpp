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
#include <torch/csrc/autograd/custom_function.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {
    using torch::autograd::Function;
    using torch::autograd::AutogradContext;
    using tensor_list = std::vector<at::Tensor>;
    namespace
    {
      at::Tensor &cast_nocheck(at::Tensor &result, const at::Tensor &self)
      {
        int64_t dstDataType =
            CalcuOpUtil::ConvertToAclDataType(result.scalar_type());
        OpCommand cmd;
        cmd.Name("Cast")
            .Input(self)
            .Output(result)
            .Attr("dst_type", dstDataType)
            .Run();
        return result;
      }
    } // namespace

    at::Tensor npu_dtype_cast_impl(const at::Tensor &self, at::ScalarType dtype)
    {
      if (self.dtype() == dtype)
      {
        return self.clone();
      }
      // calculate the output size
      auto outputSize = input_same_output_size(self);

      // construct the output tensor of the NPU
      at::Tensor result =
          OpPreparation::ApplyTensor(outputSize, self.options().dtype(dtype), self);

      // calculate the output result of the NPU
      cast_nocheck(result, self);

      return result;
    }

    at::Tensor &NPUNativeFunctions::npu_dtype_cast_(at::Tensor &self, const at::Tensor &src)
    {
      if (self.dtype() == src.dtype())
      {
        return self;
      }

      if (!NpuUtils::check_match(&self))
      {
        at::Tensor contiguousSelf = OpPreparation::ApplyTensor(self.sizes(), self.options().dtype(self.dtype()), self);
        at::Tensor result = cast_nocheck(contiguousSelf, src);
        NpuUtils::format_fresh_view(self, result);
      }
      else
      {
        cast_nocheck(self, src);
      }

      return self;
    }

    class NPUDtypeCastFunction : public torch::autograd::Function<NPUDtypeCastFunction> {
    public:
      static at::Tensor forward(AutogradContext *ctx,
          at::Tensor self, 
          at::ScalarType dtype) {
        at::AutoNonVariableTypeMode g;
        ctx->saved_data["dtype"] = self.scalar_type();
        return npu_dtype_cast_impl(self, dtype);
      }

      static tensor_list backward(AutogradContext *ctx,
          tensor_list grad_outputs) {
        auto dtype = ctx->saved_data["dtype"].toScalarType();
        grad_outputs[0].requires_grad_();
        return {NPUDtypeCastFunction::apply(grad_outputs[0], dtype), at::Tensor()};
      }
    };

    at::Tensor NPUNativeFunctions::npu_dtype_cast(const at::Tensor &self, 
        at::ScalarType dtype) {
      return NPUDtypeCastFunction::apply(self, dtype);
    }

  } // namespace native
} // namespace at_npu
