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
#include "c10/npu/OptionsManager.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& fill_out_npu(Tensor& result, Tensor& self, const Tensor& other) {
  if (!c10::npu::OptionsManager::CheckDynamicEnable()) {
    SmallVector<int64_t, N> dims;
    if (self.dim() != 0){
      dims = array_to_small_vector(self.sizes());
    } else {
      dims = {1};
    }
    OpCommand cmd;
    cmd.Name("Fill")
        .Input(dims, at::kLong)
        .Input(other)
        .Output(result)
        .Run();
  } else {
    OpCommand cmd;
    cmd.Name("FillD")
      .Input(other)
      .Output(result)
      .Attr("dims", self.sizes())
      .Run();
  }
  return result;
}

Tensor& fills_out_npu(Tensor& result, Tensor& self, Scalar value) {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, self.scalar_type(), "fills_out_npu", [&]() {
    auto value_converted = value.to<scalar_t>();});
  OpCommand cmd;
  cmd.Name("Fills")
      .Input(self)
      .Output(result)
      .Attr("value", value)
      .Run();

  return result;
}

Tensor& fill_npu_(Tensor& self, const Tensor& other) {
  if (other.dim() == 0 && !other.is_npu()) {
    fills_out_npu(self, self, other.item());
  } else {
    fill_out_npu(self, self, other);
  }

  return self;
}

Tensor& fill_npu_(Tensor& self, Scalar value) {
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = fills_out_npu(contiguousSelf, contiguousSelf, value);
    NpuUtils::format_fresh_view(self, result);
  } else {
    fills_out_npu(self, self, value);
  }

  return self;
}

} // namespace native
} // namespace at
