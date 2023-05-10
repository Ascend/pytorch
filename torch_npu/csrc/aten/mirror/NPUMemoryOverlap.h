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

#ifndef __PLUGIN_NATIVE_UTILS_NPU_MEMORY_OVERLAP__
#define __PLUGIN_NATIVE_UTILS_NPU_MEMORY_OVERLAP__

#include <ATen/ATen.h>

namespace at_npu { namespace native {

// MemOverlap: Whether or not there is memory overlap
//
// NO: Absolutely no memory overlap
// YES: Absolutely yes memory overlap
// TOO_HARD: There might be memory overlap, but it was too expensive to compute
// IS_NULL: In npu graph mode, some tensors have no device ptr.
//
// NB: Please update the python test for these if you renumber them.
enum class MemOverlap { NO, YES, TOO_HARD, IS_NULL };
enum class MemOverlapStatus { FULL, PARTIAL, NO, TOO_HARD, IS_NULL };

MemOverlap has_internal_overlap(const at::Tensor& t);
MemOverlap has_internal_overlap(at::TensorImpl* t);

void assert_no_internal_overlap(const at::Tensor& t);
void assert_no_internal_overlap(at::TensorImpl* t);

MemOverlapStatus get_overlap_status(const at::Tensor& a, const at::Tensor& b);
MemOverlapStatus get_overlap_status(const at::TensorImpl* a, const at::TensorImpl* b);

void assert_no_partial_overlap(const at::Tensor& a, const at::Tensor& b);
void assert_no_partial_overlap(at::TensorImpl* a, at::TensorImpl* b);

}}

#endif