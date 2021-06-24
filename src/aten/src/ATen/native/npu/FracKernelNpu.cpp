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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& frac_out_npu(Tensor& out, const Tensor& self) {
  Tensor cast_return_Tensor = at::npu_dtype_cast(self, ScalarType::Int);
  at::native::sub_out_npu(out,self, cast_return_Tensor);
  return out;
}

Tensor frac_npu(const Tensor& self) {
  Tensor result = at::empty_with_format(
      self.sizes(), self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  Tensor cast_return_Tensor = at::npu_dtype_cast(self, ScalarType::Int);
  frac_out_npu(result, self);

  return result;
}

Tensor& frac_npu_(Tensor& self) {
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = frac_out_npu(contiguousSelf, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    frac_out_npu(self, self);
  }
  return self;
}
}
}
