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

Tensor& erfinv_out_npu_nocheck(const Tensor& self, Tensor& result) {
  OpCommand cmd;
  cmd.Name("Erfinv")
      .Input(self)
      .Output(result)
      .Run();
  return result;
}

Tensor& erfinv_out_npu(Tensor& result, const Tensor& self) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  if (!NpuUtils::check_match(&result)) {
    Tensor contiguousResult = NpuUtils::format_contiguous(result);
    erfinv_out_npu_nocheck(self, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    erfinv_out_npu_nocheck(self, result);
  }
  return result;
}

Tensor erfinv_npu(const Tensor &self) {
  auto result = OpPreparation::ApplyTensor(self);
  erfinv_out_npu_nocheck(self, result);
  return result;
}

Tensor& erfinv_npu_(Tensor& self) {
  erfinv_out_npu(self, self);
  return self;
}

}  // namespace native
}  // namespace at
