// Copyright (c) 2021 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/aten/XLANativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& reshape_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef shape,
    bool can_refresh) {
  if (can_refresh) {
    StorageDescHelper::SetDesc(
        result,
        array_to_small_vector(result.sizes()),
        array_to_small_vector(result.strides()));
  } else {
    copy_d2d_by_memcpy(
        result,
        self,
        c10::multiply_integers(torch_npu::NPUBridge::GetNpuStorageImpl(result)->get_npu_desc().storage_sizes_));
  }
  return result;
}

at::Tensor& XLANativeFunctions::npu_reshape_out(
    const at::Tensor& self,
    at::IntArrayRef shape,
    bool can_refresh,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self,
      shape);
  return reshape_out_nocheck(result, self, shape, can_refresh);
}

at::Tensor XLANativeFunctions::npu_reshape(const at::Tensor& self, at::IntArrayRef shape, bool can_refresh) {
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      shape, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  reshape_out_nocheck(result, self, shape, can_refresh);

  return result;
}

} // namespace native
} // namespace at_npu