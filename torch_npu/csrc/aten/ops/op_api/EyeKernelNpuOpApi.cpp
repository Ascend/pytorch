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

at::Tensor& NPUNativeOpApiFunctions::eye_out(int64_t n, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnEye, NPUNativeFunctions::eye_out(n, result));
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);
  result.resize_({n, n});
  EXEC_NPU_CMD(aclnnEye, n, n, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::eye_out(int64_t n, int64_t m, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnEye, NPUNativeFunctions::eye_out(n, m, result));
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);
  TORCH_CHECK(m >= 0, "m must be greater or equal to 0, got ", m);
  result.resize_({n, m});
  EXEC_NPU_CMD(aclnnEye, n, m, result);
  return result;
}

} // namespace native
} // namespace at_npu
