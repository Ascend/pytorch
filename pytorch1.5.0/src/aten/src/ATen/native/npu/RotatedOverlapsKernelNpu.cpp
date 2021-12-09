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

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& rotated_overlaps_npu_nocheck(
    Tensor overlaps,
    const Tensor& self,
    const Tensor& query_boxes,
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

Tensor rotated_overlaps_npu(
    const Tensor& self,
    const Tensor& query_boxes,
    bool trans) {
  TORCH_CHECK(self.ndimension() == 3 && query_boxes.ndimension() == 3,
              "boxes' dim should be equal to query_boxes' ndimension() ",
              "and equal to 3!");
  auto origin_dtype = self.scalar_type();
  // the Op only support fp32 currently!
  Tensor selfCp = self.npu_dtype_cast(at::kFloat).permute({0, 2, 1});
  Tensor queryBoxesCp = query_boxes.npu_dtype_cast(at::kFloat).permute({0, 2, 1});

  int64_t B = selfCp.size(0);
  int64_t N = selfCp.size(-1);
  int64_t K = queryBoxesCp.size(-1);

  SmallVector<int64_t, SIZE> output_size({B, N, K});
  Tensor overlaps = OpPreparation::ApplyTensor(selfCp, output_size);

  rotated_overlaps_npu_nocheck(overlaps, selfCp, queryBoxesCp, trans);
  overlaps = overlaps.npu_dtype_cast(origin_dtype);
  return overlaps;
}

} // namespace native
} // namespace at
