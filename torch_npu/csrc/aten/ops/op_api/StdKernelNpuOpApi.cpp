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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::std_out(
    const at::Tensor& self, 
    at::IntArrayRef dim, 
    bool unbiased, 
    bool keepdim,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnStd, NPUNativeFunctions::std_out(self, dim, unbiased, keepdim, result));
  return NPUNativeOpApiFunctions::std_out(self, c10::optional<at::IntArrayRef>(dim), int64_t{unbiased ? 1 : 0}, keepdim, result);
}

at::Tensor& NPUNativeOpApiFunctions::std_out(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnStd, NPUNativeFunctions::std_out(self, dim, unbiased, keepdim, result));
  return NPUNativeOpApiFunctions::std_out(self, dimnames_to_positions(self, dim), unbiased, keepdim, result);
}

at::Tensor& NPUNativeOpApiFunctions::std_out(
    const at::Tensor& self,
    c10::optional<at::IntArrayRef> dim,
    c10::optional<int64_t> correction,
    bool keepdim,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnStd, NPUNativeFunctions::std_out(self, dim, correction, keepdim, result));
  c10::SmallVector<int64_t, SIZE> real_dim = {};
  if (dim.has_value()) {
    real_dim = array_to_small_vector(dim.value());
  }
  auto output_size = reduce_ops_npu_output_size(self, real_dim, keepdim);
  auto real_correction = correction.has_value() ? correction.value() : 1;

  OpPreparation::CheckOut(
      {self}, 
      result, 
      self,
      output_size);

  EXEC_NPU_CMD(aclnnStd, self, dim, real_correction, keepdim, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::std_out(
    const at::Tensor& self,
    at::DimnameList dim,
    c10::optional<int64_t> correction,
    bool keepdim,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnStd, NPUNativeFunctions::std_out(self, dim, correction, keepdim, result));
  return NPUNativeOpApiFunctions::std_out(self, dimnames_to_positions(self, dim), correction, keepdim, result);
}

at::Tensor NPUNativeOpApiFunctions::std(
    const at::Tensor & self,
    c10::optional<at::IntArrayRef> dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  DO_COMPATIBILITY(aclnnStd, NPUNativeFunctions::std(self, dim, correction, keepdim));
  c10::SmallVector<int64_t, SIZE> real_dim = {};
  if (dim.has_value()) {
    real_dim = array_to_small_vector(dim.value());
  }
  auto output_size = reduce_ops_npu_output_size(self, real_dim, keepdim);
  auto result = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options());
  return NPUNativeOpApiFunctions::std_out(self, dim, correction, keepdim, result);
}

at::Tensor NPUNativeOpApiFunctions::std(
    const at::Tensor & self,
    at::DimnameList dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  DO_COMPATIBILITY(aclnnStd, NPUNativeFunctions::std(self, dim, correction, keepdim));
  return NPUNativeOpApiFunctions::std(self, dimnames_to_positions(self, dim), correction, keepdim);
}

} // namespace native
} // namespace at_npu
