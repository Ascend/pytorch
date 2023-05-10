// Copyright (c) 2022, Huawei Technologies.All rights reserved.
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

at::Tensor& channel_shuffle_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, int64_t groups) {
  OpCommand cmd;
  cmd.Name("ShuffleChannel")
     .Input(self)
     .Output(result)
     .Attr("group", groups)
     .Run();
  return result;
}

at::Tensor NPUNativeFunctions::channel_shuffle(const at::Tensor& self, int64_t groups) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  channel_shuffle_out_npu_nocheck(result, self, groups);
  return result;
}

} // namespace native
} // namespace at_npu
