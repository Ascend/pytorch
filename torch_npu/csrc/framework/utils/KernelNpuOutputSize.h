// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#ifndef __PLUGIN_NATIVE_NPU_UTILS_KERNEL_NPU_OUTPUT_SIZE__
#define __PLUGIN_NATIVE_NPU_UTILS_KERNEL_NPU_OUTPUT_SIZE__

#include <ATen/ATen.h>


namespace at_npu {
namespace native {
// npu tensor max size
const int SIZE = 8;
c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape);
}  // namespace native
}  // namespace at_npu

#endif
