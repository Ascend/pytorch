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
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::gather_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnGather, NPUNativeFunctions::gather_out(self, dim, index, sparse_grad, result));
  auto output_size = index.sizes();
  OpPreparation::CheckOut(
      {self},
      result,
      self.scalar_type(),
      output_size);
  EXEC_NPU_CMD(aclnnGather, self, dim, index, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::gather_out(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnGather, NPUNativeFunctions::gather_out(self, dim, index, sparse_grad, result));
  auto output_size = index.sizes();
  OpPreparation::CheckOut(
      {self},
      result,
      self.scalar_type(),
      output_size);
  const int64_t real_dim = dimname_to_position(self, dim);
  EXEC_NPU_CMD(aclnnGather, self, real_dim, index, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::gather(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad) {
  DO_COMPATIBILITY(aclnnGather, NPUNativeFunctions::gather(self, dim, index, sparse_grad));
  auto outputSize = index.sizes();
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, outputSize);
  EXEC_NPU_CMD(aclnnGather, self, dim, index, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::gather(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    bool sparse_grad) {
  DO_COMPATIBILITY(aclnnGather, NPUNativeFunctions::gather(self, dim, index, sparse_grad));
  auto outputSize = index.sizes();
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, outputSize);
  const int64_t real_dim = dimname_to_position(self, dim);
  EXEC_NPU_CMD(aclnnGather, self, real_dim, index, result);
  return result;
}
} // namespace native
} // namespace at_npu
