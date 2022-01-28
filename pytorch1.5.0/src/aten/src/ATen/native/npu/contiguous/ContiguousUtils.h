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

#ifndef __NATIVE_NPU_CONTIGUOUS_CONTIGUOUS_UTILS__
#define __NATIVE_NPU_CONTIGUOUS_CONTIGUOUS_UTILS__

#include "c10/util/SmallVector.h"
#include <third_party/acl/inc/acl/acl_base.h>
#include <ATen/native/npu/utils/NpuUtils.h>
namespace at {
namespace native {
namespace npu {

// Max size of discontiguous cases vector
constexpr int MAX_CASES = 8;
// Max size of shape size
constexpr int MAX_DIM = 5;

// Define the discontiguous cases vector to be optimized
using OptimizationCases = SmallVector<string, MAX_CASES>;

struct ContiguousTensorDesc {
  bool is_contiguous_;
  SmallVector<int64_t, MAX_DIM> sizes_;
  SmallVector<int64_t, MAX_DIM> strides_;
  int64_t offset_;
  SmallVector<int64_t, MAX_DIM> base_sizes_;
  SmallVector<int64_t, MAX_DIM> base_strides_;
  SmallVector<int64_t, MAX_DIM> storage_sizes_;
  int64_t base_offset_;
  aclFormat npu_format_;
  OptimizationCases opt_cases_;
  void refresh_contiguous_using_size_and_stride();
  void reset_optimization_cases(const OptimizationCases& opt_cases);
  void add_optimization_case(const string& opt_case);
  void find_match_optimization_cases();
};

} // namespace npu
} // namespace native
} // namespace at

#endif