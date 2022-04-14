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

#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& take_out_nocheck(const Tensor& self, const Tensor& index, Tensor& result) {
  Tensor input_tensor = self.reshape(-1);
  Tensor contiguousSelf = NpuUtils::format_contiguous(input_tensor);
  Tensor contiguousIndex = NpuUtils::format_contiguous(index);
  OpCommand cmd;
  cmd.Name("Gather")
      .Input(contiguousSelf)
      .Input(contiguousIndex)
      .Output(result)
      .Attr("validate_indices", false)
      .Run();
  return result;
}

Tensor& take_out_npu(Tensor& result, const Tensor& self, const Tensor& index) {
  OpPreparation::CheckOut(
      {self, index},
      result,
      self,
      index.sizes());

  if (!NpuUtils::check_match(&result)) {
    Tensor contiguousResult = NpuUtils::format_contiguous(result);
    take_out_nocheck(self, index, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    take_out_nocheck(self, index, result);
  }
  return result;
}

Tensor take_npu(const Tensor& self, const Tensor& index) {
  Tensor result = OpPreparation::ApplyTensor(self, index.sizes());
  take_out_nocheck(self, index, result);
  return result;
}
} // namespace native
} // namespace at
