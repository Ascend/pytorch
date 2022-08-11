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

#include "torch_npu/csrc/framework/contiguous/ContiguousUtils.h"

namespace at_npu {
namespace native {

void ContiguousTensorDesc::refresh_contiguous_using_size_and_stride() {
  if (c10::multiply_integers(sizes_) == 0) {
    is_contiguous_ = true;
  }
  int64_t infer_axis_size = 1;
  for (int64_t dim = sizes_.size() - 1; dim >= 0; dim--) {
    if (sizes_[dim] != 1) {
      if (strides_[dim] == infer_axis_size) {
        infer_axis_size *= sizes_[dim];
      } else {
        is_contiguous_ = false;
        return;
      }
    }
  }
  is_contiguous_ = true;
}

void ContiguousTensorDesc::reset_optimization_cases(
    const OptimizationCases &opt_cases) {
  opt_cases_ = opt_cases;
}

void ContiguousTensorDesc::add_optimization_case(const std::string &opt_case) {
  opt_cases_.emplace_back(opt_case);
}

void ContiguousTensorDesc::find_match_optimization_cases() {
  for (auto i = 0; i < sizes_.size(); i++) {
    if (strides_[i] == 0) {
      opt_cases_.emplace_back("broadcast");
      return;
    }
  }

  for (auto i = 0; i < strides_.size() - 1; i++) {
    if (strides_[i] < strides_[i + 1]) {
      opt_cases_.emplace_back("permute");
      return;
    }
  }

  // Considering combined-cases, we cannot split slice cases any further.
  if (c10::multiply_integers(sizes_) < c10::multiply_integers(base_sizes_)) {
    opt_cases_.emplace_back("slice");
    opt_cases_.emplace_back("select");
    opt_cases_.emplace_back("indexing");
    return;
  }
}

} // namespace native
} // namespace at_npu