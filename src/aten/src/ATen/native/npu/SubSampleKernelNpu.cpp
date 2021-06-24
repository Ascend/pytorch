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

Tensor sub_sample_npu(const Tensor &self, int64_t per_images,
                      double positive_fraction) {
  Tensor result = at::empty_with_format(
      self.sizes(), self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  OpCommand cmd;
  cmd.Name("SubSample")
      .Input(self)
      .Output(result)
      .Attr("batch_size_per_images", per_images)
      .Attr("positive_fraction", (float)positive_fraction)
      .Run();

  return result;
}

} // namespace native
} // namespace at
