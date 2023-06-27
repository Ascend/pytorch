// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::hardtanh(const at::Tensor& self, const at::Scalar& min, const at::Scalar& max) {
  DO_COMPATIBILITY(aclnnHardtanh, NPUNativeFunctions::hardtanh(self, min, max));
  at::Tensor out = OpPreparation::ApplyTensor(self);
  EXEC_NPU_CMD(aclnnHardtanh, self, min, max, out);
  return out;
}

at::Tensor& NPUNativeOpApiFunctions::hardtanh_(at::Tensor& self, const at::Scalar& min, const at::Scalar& max) {
  DO_COMPATIBILITY(aclnnInplaceHardtanh, NPUNativeFunctions::hardtanh_(self, min, max));
  EXEC_NPU_CMD(aclnnInplaceHardtanh, self, min, max);
  return self;
}

} // namespace native
} // namespace at_npu
