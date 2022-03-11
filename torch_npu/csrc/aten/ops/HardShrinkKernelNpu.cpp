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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& hardshrink_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar lambd) {
  OpCommand cmd;
  cmd.Name("HardShrink")
    .Input(self)
    .Attr("lambd", lambd)
    .Output(result).Run();
    
    return result;
}

at::Tensor NPUNativeFunctions::hardshrink(const at::Tensor& self, at::Scalar lambd) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  hardshrink_nocheck(result, self, lambd);

  return result;
}

} // namespace native
} // namespace at_npu