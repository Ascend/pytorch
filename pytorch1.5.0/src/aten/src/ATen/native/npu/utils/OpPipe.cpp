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

#include "ATen/native/npu/utils/OpPipe.h"
#include "ATen/native/npu/utils/OpPreparation.h"

namespace at {
namespace native {
namespace npu {

OpPipeWithDefinedOut& OpPipeWithDefinedOut::CheckMemory(const std::initializer_list<Tensor>& inputs, const std::initializer_list<Tensor>& outputs) {
  OpPreparation::CheckMemory(inputs, outputs);
  return *this;
}

Tensor& OpPipeWithDefinedOut::Call(Tensor &dst) {
  if (!NpuUtils::check_match(&dst)) {
    Tensor contigTensor = NpuUtils::format_contiguous(dst);
    this->func(contigTensor);
    NpuUtils::format_fresh_view(dst, contigTensor);
  } else {
    this->func(dst);
  }
  return dst;
}

OpPipeWithApplyOut& OpPipeWithApplyOut::ApplyOutputSameAs(const Tensor& src) {
  this->dst = OpPreparation::ApplyTensor(src);
  return *this;
}

Tensor& OpPipeWithApplyOut::Call() {
  this->func(this->dst);
  return this->dst;
}


} // namespace npu
} // namespace native
} // namespace at