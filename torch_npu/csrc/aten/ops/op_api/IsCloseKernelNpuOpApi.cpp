// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::isclose(
    const at::Tensor& self,
    const at::Tensor& other,
    double rtol,
    double atol,
    bool equal_nan) {
  DO_COMPATIBILITY(aclnnIsClose, NPUNativeFunctions::isclose(self, other, rtol, atol, equal_nan));
  at::Tensor out;
  // calculate the output size
  if (equal_nan == true) {
    auto output_size = input_same_output_size(self);
    out = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options().dtype(at::kBool));
  } else {
    auto output_size = broadcast_ops_npu_output_size(self, other);
    out = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options().dtype(at::kBool));
  }
  // construct the output tensor of the NPU
  EXEC_NPU_CMD(aclnnIsClose, self, other, rtol, atol, equal_nan, out);
  return out;
}

} // namespace native
} // namespace at_npu
