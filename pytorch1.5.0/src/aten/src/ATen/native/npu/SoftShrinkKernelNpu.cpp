// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& softshrink_out_npu_nocheck(   
    Tensor& result, 
    const Tensor& self,
    Scalar lambd) {
  TORCH_CHECK(lambd.toFloat() > 0, "lambd should be greater than 0");
  float lambd_value = CalcuOpUtil::get_scalar_float_value(lambd);
  OpCommand cmd;
  cmd.Name("SoftShrink")
      .Input(self)
      .Output(result)
      .Attr("lambd", lambd_value)
      .Run();
  return result;
}

Tensor& softshrink_out_npu(   
    Tensor& result, 
    const Tensor& self,
    Scalar lambd) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  if (!NpuUtils::check_match(&result)) {
    Tensor contiguousResult = NpuUtils::format_contiguous(result);
    softshrink_out_npu_nocheck(contiguousResult, self, lambd);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
     softshrink_out_npu_nocheck(result, self, lambd);
  }
    return result;
}

Tensor softshrink_npu(const Tensor& self, Scalar lambd) {
  Tensor result = OpPreparation::ApplyTensor(self);
  softshrink_out_npu_nocheck(result, self, lambd);
  return result;
}

} // namespace native
} // namespace at