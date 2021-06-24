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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor isfinite_npu(const Tensor& self_ex) {
  Tensor self = self_ex;
  if (self.storage().unsafeGetStorageImpl()->npu_desc_.npu_format_ !=
      ACL_FORMAT_ND) {
    self = self_ex.npu_format_cast(ACL_FORMAT_ND);
  }
  if (self.scalar_type() == ScalarType::Half) {
    self = self.npu_dtype_cast(ScalarType::Float);
  }
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options().dtype(kBool), ACL_FORMAT_ND);

  // calculate the output result of the NPU
  OpCommand cmd;
  cmd.Name("PTIsfinite")
      .Input(self)
      .Output(result)
      .Attr("kernel_name", "PTIsfinite")
      .Run();
  return result;
}

} // namespace native
} // namespace at