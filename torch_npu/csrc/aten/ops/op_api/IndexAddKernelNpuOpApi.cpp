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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::index_add_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnIndexAdd, NPUNativeFunctions::index_add_out(self, dim, index, source, alpha, result));
  OpPreparation::CheckOut(
      {self, index, source},
      result,
      result.scalar_type(),
      self.sizes());
  EXEC_NPU_CMD(aclnnIndexAdd, self, dim, index, source, alpha, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::index_add(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha) {
  DO_COMPATIBILITY(aclnnIndexAdd, NPUNativeFunctions::index_add(self, dim, index, source, alpha));
  at::Tensor result = OpPreparation::ApplyTensor(self, self.options());
  EXEC_NPU_CMD(aclnnIndexAdd, self, dim, index, source, alpha, result);
  return result;
}

} // namespace native
} // namespace at_npu