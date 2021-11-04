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

#pragma once

#include <c10/core/TensorOptions.h>

// npu_lazy_init() is always compiled, even for CPU-only builds.
// Thus, it does not live in the npu/ folder.

namespace torch {
namespace utils {

// The INVARIANT is that this function MUST be called before you attempt
// to get a NPU Type object from ATen, in any way.  Here are some common
// ways that a Type object may be retrieved:
//
//    - You call getNonVariableType or getNonVariableTypeOpt
//    - You call toBackend() on a Type
//
// It's important to do this correctly, because if you forget to add it
// you'll get an oblique error message about "Cannot initialize NPU without
// ATen_cuda library" if you try to use NPU functionality from a CPU-only
// build, which is not good UX.

void npu_lazy_init();
void npu_set_run_yet_variable_to_false();

static void maybe_initialize_npu(const at::TensorOptions& options) {
  if (options.device().is_npu()) {
    torch::utils::npu_lazy_init();
  }
}

}
}
