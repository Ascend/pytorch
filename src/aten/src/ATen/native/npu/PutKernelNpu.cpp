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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& put_npu_(Tensor& self, const Tensor& index, const Tensor& source, bool accumulate) {
  TORCH_CHECK(index.numel() == source.numel(), "source should have the same number of elements as index");
  if (source.numel() == 0) {
    return self;
  }
  SmallVector<Tensor, N> inputs = {self};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);
  
  // ScatterNdAdd与ScatterNdUpdate是内存复用算子，所以得保证调用cmd时的input与进入算子的是同一个tensor
  Tensor selfFlatten = NpuUtils::format_contiguous(self.reshape(-1));
  Tensor indexFlatten = index.reshape({-1, 1});
  Tensor sourceFlatten = source.reshape(-1);
  
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
}  // namespace at
