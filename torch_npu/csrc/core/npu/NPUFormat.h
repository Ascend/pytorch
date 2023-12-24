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
#pragma once

#include <ATen/ATen.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace at_npu {
namespace native {

TORCH_NPU_API int64_t get_npu_format(const at::Tensor &tensor);

TORCH_NPU_API std::vector<int64_t> get_npu_storage_sizes(const at::Tensor &tensor);

TORCH_NPU_API at::Tensor npu_format_cast(const at::Tensor &tensor, int64_t acl_format);

TORCH_NPU_API at::Tensor empty_with_format(c10::IntArrayRef sizes, const c10::TensorOptions &options,
                                           int64_t format, bool keep_format = false);

} // namespace native
} // namespace at_npu
