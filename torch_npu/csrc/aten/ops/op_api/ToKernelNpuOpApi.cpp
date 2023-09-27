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

#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/NativeFunctions.h>
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::to(const at::Tensor& self, at::ScalarType dtype, bool non_blocking, bool copy,
                                       c10::optional<c10::MemoryFormat> optional_memory_format) {
  DO_COMPATIBILITY(aclnnCast, NPUNativeFunctions::to(self, dtype, non_blocking, copy, optional_memory_format));
  if (self.dtype() == dtype && !copy) {
    return self;
  }

  if (dtype == at::ScalarType::Double) {
    TORCH_NPU_WARN_ONCE("Device do not support double dtype now, "
                        "dtype cast repalce with float.");
    dtype = at::ScalarType::Float;
  }
  return custom_ops::npu_dtype_cast(self, dtype);
}

}  // namespace native
}  // namespace at_npu
