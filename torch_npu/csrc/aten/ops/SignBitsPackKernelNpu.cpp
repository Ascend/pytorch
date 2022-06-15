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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& sign_bits_pack_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t size) {
  OpCommand cmd;
  cmd.Name("SignBitsPack")
    .Input(self)
    .Output(result) 
    .Attr("size", size)
    .Run();
  return result;
}

at::Tensor NPUNativeFunctions::npu_sign_bits_pack(const at::Tensor& self, int64_t size) {
  TORCH_CHECK(self.dim() == 1, "input must be one-dimensional");
  TORCH_CHECK(self.scalar_type() == at::ScalarType::Half || self.scalar_type() == at::ScalarType::Float,
      "all only supports torch.float16 and torch.float32 dtypes");
  auto ysize = (self.numel() + 7) / 8;
  TORCH_CHECK(size != 0 && ysize % size == 0, "all must be divisible by size");
  at::Tensor result = OpPreparation::ApplyTensor({size, ysize / size}, self.options().dtype(at::kByte), self);
  
  // calculate the output result of the NPU
  sign_bits_pack_npu_nocheck(result, self, size);

  return result;
}

} // namespace native
} // namespace at_npu