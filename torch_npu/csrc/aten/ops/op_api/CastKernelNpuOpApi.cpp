// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;

at::Tensor npu_dtype_cast_impl_op_api(const at::Tensor& self, at::ScalarType dtype) {
  if (self.dtype() == dtype) {
    return self.clone();
  }
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self.sizes(), self.options().dtype(dtype));

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnCast, self, dtype, result);

  return result;
}

class NPUDtypeCastOpApiFunction : public torch::autograd::Function<NPUDtypeCastOpApiFunction> {
public:
  static at::Tensor forward(AutogradContext* ctx, at::Tensor self, at::ScalarType dtype) {
    at::AutoNonVariableTypeMode g;
    ctx->saved_data["dtype"] = self.scalar_type();
    return npu_dtype_cast_impl_op_api(self, dtype);
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto dtype = ctx->saved_data["dtype"].toScalarType();
    grad_outputs[0].requires_grad_();
    return {NPUDtypeCastOpApiFunction::apply(grad_outputs[0], dtype), at::Tensor()};
  }
};

at::Tensor NPUNativeOpApiFunctions::npu_dtype_cast(const at::Tensor& self, at::ScalarType dtype) {
  DO_COMPATIBILITY(aclnnCast, NPUNativeFunctions::npu_dtype_cast(self, dtype));
  return NPUDtypeCastOpApiFunction::apply(self, dtype);
}

}  // namespace native
}  // namespace at_npu
