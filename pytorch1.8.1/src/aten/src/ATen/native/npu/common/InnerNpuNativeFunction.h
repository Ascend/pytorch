// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#ifndef __NATIVE_NPU_COMMON_INNER_NATIVE_FUNCTION__
#define __NATIVE_NPU_COMMON_INNER_NATIVE_FUNCTION__

#include <ATen/ATen.h>

namespace at {
namespace native {
bool can_use_memcpy(Tensor& dst, const Tensor& src);
void copy_kernel_npu(Tensor& self, const Tensor& src, bool non_blocking);
void copy_d2d_by_memcpy(Tensor& dst, const Tensor& src, int64_t exceptSize=0);
void copy_d2d_dtype(Tensor& self, const Tensor& src, bool non_blocking);
bool try_to_optimize_copy_with_any_format(Tensor& self, const Tensor& src);
Tensor matmul_by_bmmV2(const Tensor& tensor1, const Tensor& tensor2);
} // namespace native
} // namespace at

#endif