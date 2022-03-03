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

at::Tensor NPUNativeFunctions::isfinite(const at::Tensor& self_ex) {
  at::Tensor self = self_ex;
  if (self.storage().unsafeGetStorageImpl()->npu_desc_.npu_format_ !=
      ACL_FORMAT_ND) {
    self = NPUNativeFunctions::npu_format_cast(self_ex, ACL_FORMAT_ND);
  }
  if (self.scalar_type() == at::ScalarType::Half) {
    self = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }
  auto outputSize = self.sizes();
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, self.options().dtype(at::kBool), ACL_FORMAT_ND);
  OpCommand cmd;
  cmd.Name("IsFinite")
      .Input(self)
      .Output(result)
      .Run();
  return result;
}

} // namespace native
} // namespace at_npu