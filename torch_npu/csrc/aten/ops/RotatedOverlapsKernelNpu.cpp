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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& rotated_overlaps_npu_nocheck(
    at::Tensor& overlaps,
    const at::Tensor& self,
    const at::Tensor& query_boxes,
    bool trans) {
  OpCommand cmd;
  cmd.Name("RotatedOverlaps")
      .Input(self)
      .Input(query_boxes)
      .Output(overlaps)
      .Attr("trans", trans)
      .Run();
  return overlaps;
}

at::Tensor NPUNativeFunctions::npu_rotated_overlaps(
    const at::Tensor& self,
    const at::Tensor& query_boxes,
    bool trans) {
  TORCH_CHECK(self.ndimension() == 3 && query_boxes.ndimension() == 3,
              "boxes' dim should be equal to query_boxes' ndimension() ",
              "and equal to 3!");
  auto origin_dtype = self.scalar_type();
  // the Op only support fp32 currently!
  at::Tensor selfCp = NPUNativeFunctions::npu_dtype_cast(self, at::kFloat).permute({0, 2, 1});
  at::Tensor queryBoxesCp = NPUNativeFunctions::npu_dtype_cast(query_boxes, at::kFloat).permute({0, 2, 1});

  int64_t B = selfCp.size(0);
  int64_t N = selfCp.size(-1);
  int64_t K = queryBoxesCp.size(-1);

  c10::SmallVector<int64_t, SIZE> output_size({B, N, K});
  at::Tensor overlaps = OpPreparation::ApplyTensor(selfCp, output_size);

  rotated_overlaps_npu_nocheck(overlaps, selfCp, queryBoxesCp, trans);
  overlaps = NPUNativeFunctions::npu_dtype_cast(overlaps, origin_dtype);
  return overlaps;
}
} // namespace native
} // namespace at_npu