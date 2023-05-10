// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#pragma once

// NB: Must be at the top of file to avoid including the deprecated "math.h".
// https://stackoverflow.com/questions/6563810/m-pi-works-with-math-h-but-not-with-cmath-in-visual-studio
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#endif

#include <ATen/ATen.h>

#include "torch_npu/csrc/aten/Functions.h"

namespace at_npu {
namespace autograd {
namespace generated {
namespace details {

// A simple way to imperatively compute index ranges for slots
// that have been flattened
struct IndexRangeGenerator {
  IndexRange range(size_t range_size) {
    i += range_size;
    return {i - range_size, i};
  }
  size_t size() { return i; }
  private:
    size_t i = 0;
};

Tensor toNonOptFwGrad(const c10::optional<Tensor>& t);
Tensor toNonOptPrimal(const c10::optional<Tensor>& t);
Tensor toNonOptTensor(const c10::optional<Tensor>& t);

Tensor apply_loss_reduction(const Tensor& unreduced, int64_t reduction);
bool any_variable_defined(const variable_list& variables);
void copy_range(variable_list& out, IndexRange range, const at::Tensor& t);
void copy_range(variable_list& out, IndexRange range, at::ArrayRef<at::Tensor> t);
at::Tensor not_implemented(const char* name, const char* reason="");
std::vector<Tensor> not_implemented_list(const char* name, const char* reason="");


} // namespace details
} // namespace generated
} // namespace autograd
} // namespace at_npu
