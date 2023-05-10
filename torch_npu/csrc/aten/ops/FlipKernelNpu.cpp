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

at::Tensor NPUNativeFunctions::flip(const at::Tensor& self, at::IntArrayRef dims){
    at::Tensor result = OpPreparation::ApplyTensor(self);
    at::SmallVector<int64_t,N> dimVec = array_to_small_vector(dims);
    OpCommand cmd;
    cmd.Name("ReverseV2") 
      .Input(self) 
      .Input(dimVec, at::kLong) 
      .Output(result) 
      .Run();
    return result;
}
} // namespace native
} // namespace at_npu