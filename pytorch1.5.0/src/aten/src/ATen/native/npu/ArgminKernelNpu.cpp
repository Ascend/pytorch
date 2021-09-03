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

Tensor argmin_npu(const Tensor& self, optional<int64_t> dim, bool keepdim) {
  TORCH_CHECK(
      self.numel() > 0,
      "cannot perform reduction function argmin on a "
      "tensor with no elements because the operation does not have an identity");

  Tensor input = dim.has_value() ? self : self.reshape({-1});
  int64_t realDim = dim.has_value() ? dim.value() : 0;
  bool realKeepDim = dim.has_value() ? keepdim : false;

  // calculate the output size  
  auto outputSize = reduce_ops_npu_output_size(input, realDim, realKeepDim);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      self.options().dtype(at::kInt),
      ACL_FORMAT_ND);
  SmallVector<int64_t, N> DimVec = {realDim};
  // calculate the output result of the NPU
  OpCommand cmd;
  cmd.Name("ArgMin")
      .Input(input)
      .Input(DimVec, at::kInt)
      .Output(result)
      .Attr("keep_dims", realKeepDim)
      .Run();

  result = result.to(ScalarType::Long);
  return result;
}
} // namespace native
} // namespace at