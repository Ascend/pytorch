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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& bincount_npu_nocheck(
    const Tensor& self, 
    const Tensor& weights, 
    int64_t sizes, 
    Tensor& result) {
  OpCommand cmd;
  cmd.Name("Bincount")
      .Input(self)
      .Input(Scalar(sizes), ScalarType::Int)
      .Input(weights)
      .Output(result)
      .Run();
  return result;
}

Tensor bincount_npu(
    const Tensor& self, 
    const Tensor& weights, 
    int64_t minlength) {  
  if (self.sizes()[0] == 0) {
      auto result = OpPreparation::ApplyTensorWithSizes(
          {0}, 
          self.options().dtype(at::ScalarType::Long));
      return result;
  }  

  // calculate output size
  auto sizes = static_cast<int64_t>(
      CalcuOpUtil::get_scalar_float_value(max_npu(self).item()));
  sizes = (sizes < minlength) ? minlength : (sizes + 1);
  
  // input convert to int32
  if (self.dtype() == at::ScalarType::Long) {
      TORCH_WARN_ONCE("CANN: Bincount cann't support dtype int64.");
  }
  auto input = self.npu_dtype_cast(at::ScalarType::Int);

  // weight convert dtype as same as output defined by torch
  auto weight = weights;
  if (!weights.defined()) {
      TensorOptions options = input.options();
      weight = ones_npu(input.sizes(), options.dtype(at::ScalarType::Long));
  } else if (!(weights.dtype() == at::ScalarType::Float)) {
      weight = weights.npu_dtype_cast(at::ScalarType::Double);
  }
  
  auto result = OpPreparation::ApplyTensor(weight, {sizes});
  bincount_npu_nocheck(input, weight, sizes, result);

  return result;
}
} // namespace native
} // namespace at