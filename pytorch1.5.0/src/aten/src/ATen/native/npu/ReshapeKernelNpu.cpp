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

#include "ATen/native/npu/common/InnerNpuNativeFunction.h"
#include "ATen/native/npu/frame/StorageDescHelper.h"
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;
Tensor& reshape_out_npu(
    Tensor& result,
    const Tensor& self,
    IntArrayRef shape,
    bool can_refresh) {
  if (c10::npu::NpuRunMode::IsGraphMode()) {
    std::vector<int64_t> out_strides = at::detail::defaultStrides(shape);
    if (result.sizes() != shape || result.strides() != out_strides) {
      auto allow_flag =
          result.unsafeGetTensorImpl()->allow_tensor_metadata_change();
      result.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(true);
      StorageDescHelper::SetDesc(result, shape, out_strides);
      result.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(
          allow_flag);
    }

    OpCommand cmd;
    cmd.Name("Reshape")
        .InputWithoutContiguous(self)
        .Input(shape, at::kLong)
        .Output(result)
        .Run();
  } else if (can_refresh) {
    StorageDescHelper::SetDesc(
        result,
        array_to_small_vector(result.sizes()),
        array_to_small_vector(result.strides()));
  } else {
    copy_d2d_by_memcpy(
        result,
        self,
        prod_intlist(result.storage().get_npu_desc().storage_sizes_));
  }
  return result;
}

Tensor reshape_npu(const Tensor& self, IntArrayRef shape, bool can_refresh) {
  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      shape, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  
  // calculate the output result of the NPU
  reshape_out_npu(result, self, shape, can_refresh);

  return result;
}
} // namespace native
} // namespace at