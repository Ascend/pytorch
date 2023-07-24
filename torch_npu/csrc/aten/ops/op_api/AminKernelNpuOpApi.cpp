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
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::amin_out(const at::Tensor& self, at::IntArrayRef dim, bool keepdim, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnAmin, NPUNativeFunctions::amin_out(self, dim, keepdim, result));

  auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
  // check result for return
  OpPreparation::CheckOut({self}, result, result.scalar_type(), outputSize);
  EXEC_NPU_CMD(aclnnAmin, self, dim, keepdim, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::amin(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
  DO_COMPATIBILITY(aclnnAmin, NPUNativeFunctions::amin(self, dim, keepdim));

  // calculate the output size
  auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, outputSize);
  EXEC_NPU_CMD(aclnnAmin, self, dim, keepdim, result);
  return result;
}

} // namespace native
} // namespace at_npu
