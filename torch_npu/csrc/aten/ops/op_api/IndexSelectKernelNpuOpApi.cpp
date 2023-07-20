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
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::index_select_out(const at::Tensor& self,
                                                      int64_t dim,
                                                      const at::Tensor& index,
                                                      at::Tensor& result) {
  DO_COMPATIBILITY(aclnnIndexSelect, NPUNativeFunctions::index_select_out(self, dim, index, result));
  auto outputSize = index_select_npu_output_size(self, dim, index);
  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);
  EXEC_NPU_CMD(aclnnIndexSelect, self, dim, index, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::index_select(const at::Tensor& self,
                                                 int64_t dim,
                                                 const at::Tensor& index) {
  DO_COMPATIBILITY(aclnnIndexSelect, NPUNativeFunctions::index_select(self, dim, index));
  auto outputSize = index_select_npu_output_size(self, dim, index);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, outputSize);
  EXEC_NPU_CMD(aclnnIndexSelect, self, dim, index, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::index_select_out(const at::Tensor& self,
                                                      at::Dimname dim,
                                                      const at::Tensor& index,
                                                      at::Tensor& result) {
  DO_COMPATIBILITY(aclnnIndexSelect, NPUNativeFunctions::index_select_out(self, dim, index, result));
  return NPUNativeOpApiFunctions::index_select_out(self, dimname_to_position(self, dim), index, result);
}

at::Tensor NPUNativeOpApiFunctions::index_select(const at::Tensor& self,
                                                 at::Dimname dim,
                                                 const at::Tensor& index) {
  DO_COMPATIBILITY(aclnnIndexSelect, NPUNativeFunctions::index_select(self, dim, index));
  return NPUNativeOpApiFunctions::index_select(self, dimname_to_position(self, dim), index);
}

} // namespace native
} // namespace at_npu
