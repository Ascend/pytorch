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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& one_hot_out_npu(
    Tensor& result,
    const Tensor& self,
    int64_t axis,
    int64_t depth,
    Scalar on_value,
    Scalar off_value) {
  Tensor on_tmp = OpPreparation::ApplyTensor(
                                 {1},
                                 self.options().dtype(ScalarType::Float),
                                 self)
                                 .fill_(on_value);
  Tensor off_tmp = OpPreparation::ApplyTensor(
                                  {1},
                                  self.options().dtype(ScalarType::Float),
                                  self)
                                  .fill_(off_value);
  OpCommand cmd;
  cmd.Name("OneHotD")
      .Input(self)
      .Input(on_tmp)
      .Input(off_tmp)
      .Output(result)
      .Attr("axis", axis)
      .Attr("depth", depth)
      .Run();
  return result;
}

Tensor one_hot_npu(
    const Tensor& self,
    int64_t axis,
    int64_t depth,
    Scalar on_value,
    Scalar off_value) {
  auto outputSize = array_to_small_vector(self.sizes());
  outputSize.emplace_back(depth);

  Tensor result = OpPreparation::ApplyTensor(
      outputSize,
      self.options().dtype(ScalarType::Float),
      self);
  one_hot_out_npu(result, self, axis, depth, on_value, off_value);

  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m){
  m.impl("npu_one_hot", TORCH_FN(one_hot_npu));
}

} // namespace native
} // namespace at
