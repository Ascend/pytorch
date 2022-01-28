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

#ifndef __NATIVE_NPU_CONTIGUOUS_RESHAPE__
#define __NATIVE_NPU_CONTIGUOUS_RESHAPE__

#include <ATen/native/npu/contiguous/ContiguousOpt.h>
#include <ATen/native/npu/utils/KernelNpuOutputSize.h>
#include <THNPU/THNPUCachingHostAllocator.h>
#include "ATen/native/npu/common/InnerNpuNativeFunction.h"

namespace at {
namespace native {
namespace npu {
bool can_use_memecpy_for_NZ_format(const ContiguousTensorDesc&);
bool can_use_memcpy_for_other_format(const ContiguousTensorDesc&);
bool check_reshape_match_flex(const ContiguousTensorDesc&, const ContiguousTensorDesc&);
bool check_reshape_match(const ContiguousTensorDesc&, const ContiguousTensorDesc&);
bool check_reshape_match_flex(const ContiguousTensorDesc&);
bool check_reshape_match(const ContiguousTensorDesc&);
bool CanUseMemcpyForOtherFormat(const Tensor&);
} // namespace npu
} // namespace native
} // namespace at

#endif