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
#include <ATen/ATen.h>

#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::contiguous(const at::Tensor& self, c10::MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }

  TORCH_CHECK(
      memory_format != c10::MemoryFormat::Preserve,
      "preserve memory format is unsupported by the contiguous operator");
  return self.clone();
}

at::Tensor NPUNativeFunctions::format_contiguous(const at::Tensor &self) {
  return NpuUtils::format_contiguous(self);
}

}
}