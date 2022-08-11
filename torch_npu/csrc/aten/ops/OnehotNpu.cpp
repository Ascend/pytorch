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
#include "torch_npu/csrc/aten/XLANativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& one_hot_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t axis,
    int64_t depth,
    const at::Scalar& on_value,
    const at::Scalar& off_value) {
  at::Tensor selfCp = XLANativeFunctions::npu_dtype_cast(self, at::kInt);
  at::Tensor on_tmp = OpPreparation::ApplyTensor(
      {1},
      selfCp.options().dtype(at::ScalarType::Float),
      selfCp)
      .fill_(on_value);
  at::Tensor off_tmp = OpPreparation::ApplyTensor(
      {1},
      selfCp.options().dtype(at::ScalarType::Float),
      selfCp)
      .fill_(off_value);
  OpCommand cmd;
  cmd.Name("OneHotD")
      .Input(selfCp)
      .Input(on_tmp)
      .Input(off_tmp)
      .Output(result)
      .Attr("axis", axis)
      .Attr("depth", depth)
      .Run();
  return result;
}

at::Tensor XLANativeFunctions::npu_one_hot(
    const at::Tensor& self,
    int64_t axis,
    int64_t depth,
    const at::Scalar& on_value,
    const at::Scalar& off_value) {
  auto outputSize = array_to_small_vector(self.sizes());
  outputSize.emplace_back(depth);

  at::Tensor result = OpPreparation::ApplyTensor(
      outputSize,
      self.options().dtype(at::ScalarType::Float),
      self);
  one_hot_out_npu(result, self, axis, depth, on_value, off_value);

  return result;
}
} // namespace native
} // namespace at_npu