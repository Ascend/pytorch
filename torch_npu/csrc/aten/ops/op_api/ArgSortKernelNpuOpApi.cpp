// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include<ATen/NamedTensorUtils.h>
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
  at::Tensor NPUNativeOpApiFunctions::argsort(const at::Tensor& self, int64_t dim, bool descending) {
    DO_COMPATIBILITY(aclnnArgsort, NPUNativeFunctions::argsort(self, dim, descending));

    // construct the output tensor of the NPU
    at::Tensor indices = OpPreparation::ApplyTensorWithoutFormat(self.sizes(), self.options().dtype(at::kLong));
    EXEC_NPU_CMD(aclnnArgsort, self, dim, descending, indices);
    return indices;
  }

  at::Tensor NPUNativeOpApiFunctions::argsort(const at::Tensor &self, at::Dimname dim, bool descending) {
    DO_COMPATIBILITY(aclnnArgsort, NPUNativeFunctions::argsort(self, dim, descending));

    return NPUNativeOpApiFunctions::argsort(self, dimname_to_position(self, dim), descending);
  }
} // namespace native
} // namespace at_npu
