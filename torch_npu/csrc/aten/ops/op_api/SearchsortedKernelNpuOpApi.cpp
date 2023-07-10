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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::searchsorted_out(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<at::Tensor>& sorter_opt,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnSearchSorted,
                   NPUNativeFunctions::searchsorted_out(sorted_sequence, self, out_int32,
                                                        right, side_opt, sorter_opt, result));
  at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
  OpPreparation::CheckOut(
      {sorted_sequence, self},
      result,
      scalar_type,
      self.sizes());
  EXEC_NPU_CMD(aclnnSearchSorted, sorted_sequence, self, out_int32, right, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::searchsorted(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<at::Tensor>& sorter_opt) {
  DO_COMPATIBILITY(aclnnSearchSorted,
                   NPUNativeFunctions::searchsorted(sorted_sequence, self, out_int32, right, side_opt, sorter_opt));
  at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self.sizes(), self.options().dtype(scalar_type));
  EXEC_NPU_CMD(aclnnSearchSorted, sorted_sequence, self, out_int32, right, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::searchsorted(
    const at::Tensor& sorted_sequence,
    const at::Scalar& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<at::Tensor>& sorter_opt) {
  DO_COMPATIBILITY(aclnnSearchSorteds,
                   NPUNativeFunctions::searchsorted(sorted_sequence, self, out_int32, right, side_opt, sorter_opt));
  at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat({}, sorted_sequence.options().dtype(scalar_type));
  EXEC_NPU_CMD(aclnnSearchSorteds, sorted_sequence, self, out_int32, right, result);
  return result;
}
} // namespace native
} // namespace at_npu

