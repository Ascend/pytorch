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

namespace at_npu {
namespace native {
at::Tensor& bincount_npu_nocheck(
    const at::Tensor& self, 
    const at::Tensor& weights, 
    int64_t sizes, 
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Bincount")
      .Input(self)
      .Input(at::Scalar(sizes), at::ScalarType::Int)
      .Input(weights)
      .Output(result)
      .Run();
  return result;
}

at::Tensor NPUNativeFunctions::bincount(
    const at::Tensor& self, 
    const c10::optional<at::Tensor>& weight_opt, 
    int64_t minlength) {  
  const at::Tensor& weights = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  if (self.sizes()[0] == 0) {
      auto result = OpPreparation::ApplyTensorWithSizes(
          {0}, 
          self.options().dtype(at::ScalarType::Long));
      return result;
  }  

  // calculate output size
  auto sizes = static_cast<int64_t>(
      CalcuOpUtil::GetScalarFloatValue(NPUNativeFunctions::max(self).item()));
  sizes = (sizes < minlength) ? minlength : (sizes + 1);

  // input convert to int32
  if (self.dtype() == at::ScalarType::Long) {
      TORCH_WARN_ONCE("CANN: Bincount cann't support dtype int64.");
  }
  auto input = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Int);

  // weight convert dtype as same as output defined by torch
  auto weight = weights;
  if (!weights.defined()) {
      at::TensorOptions options = input.options();
      weight = NPUNativeFunctions::ones(input.sizes(), at::ScalarType::Long, options.layout(), options.device(), options.pinned_memory());
  } else if (!(weights.dtype() == at::ScalarType::Float)) {
      weight = NPUNativeFunctions::npu_dtype_cast(weights, at::ScalarType::Double);
  }
  
  auto result = OpPreparation::ApplyTensor(weight, {sizes});
  bincount_npu_nocheck(input, weight, sizes, result);

  return result;
}
} // namespace native
} // namespace at
