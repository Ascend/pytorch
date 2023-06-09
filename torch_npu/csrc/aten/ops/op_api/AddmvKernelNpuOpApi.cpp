// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

static at::Tensor& AddmvOutOpApi(const at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& result) {
  int8_t cube_math_type = 1;
  EXEC_NPU_CMD(aclnnAddmv, self, mat, vec, alpha, beta, result, cube_math_type);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::addmv_out(
    const at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnAddmv, NPUNativeFunctions::addmv_out(self, mat, vec, beta, alpha, result));
  AddmvOutOpApi(self, mat, vec, beta, alpha, result);
  return result;
}


at::Tensor NPUNativeOpApiFunctions::addmv(
    const at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  DO_COMPATIBILITY(aclnnAddmv, NPUNativeFunctions::addmv(self, mat, vec, beta, alpha));
  auto output_size = addmv_npu_output_size(self, mat, vec, beta, alpha);
  at::Tensor result = OpPreparation::ApplyTensor(self, output_size);
  AddmvOutOpApi(self, mat, vec, beta, alpha, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::addmv_(
    at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  DO_COMPATIBILITY(aclnnAddmv, NPUNativeFunctions::addmv_(self, mat, vec, beta, alpha));
  OpPreparation::CheckMemory({self, mat, vec}, {self});
  AddmvOutOpApi(self, mat, vec, beta, alpha, self);
  return self;
}
} // namespace native
} // namespace at_npu
