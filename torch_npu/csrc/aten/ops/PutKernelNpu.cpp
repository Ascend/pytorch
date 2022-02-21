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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::put_(
    at::Tensor& self,
    const at::Tensor& index,
    const at::Tensor& source,
    bool accumulate) {
  TORCH_CHECK(index.numel() == source.numel(), "source should have the same number of elements as index");
  if (source.numel() == 0) {
    return self;
  }
  c10::SmallVector<at::Tensor, N> inputs = {self};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  at::Tensor selfFlatten = NpuUtils::format_contiguous(self.reshape(-1));
  at::Tensor indexFlatten = index.reshape({-1, 1});
  at::Tensor sourceFlatten = source.reshape(-1);

  OpCommand cmd;
  accumulate ? cmd.Name("ScatterNdAdd") : cmd.Name("ScatterNdUpdate");
  cmd.Input(selfFlatten)
     .Input(indexFlatten)
     .Input(sourceFlatten)
     .Output(selfFlatten)
     .Attr("use_locking", false)
     .Run();

  self.copy_(selfFlatten);
  return self;
}
}  // namespace native
}  // namespace at_npu