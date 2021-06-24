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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& logical_not_out_npu(Tensor& result, const Tensor& self) {
  ScalarType src_type = self.scalar_type();
  if (ScalarType::Bool == src_type) {
    OpCommand cmd;
    cmd.Name("LogicalNot")
        .Input(self)
        .Output(result)
        .Run();
  } else {
    at::eq_out(result, self, 0);
  }

  return result;
}

Tensor logical_not_npu(const Tensor& self) {
  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      self.sizes(),
      self.options().dtype(kBool),
      ACL_FORMAT_NCHW);

  // calculate the output result of the NPU
  logical_not_out_npu(result, self);
  return result;
}

Tensor& logical_not_npu_(Tensor& self) {
  SmallVector<Tensor, N> inputs = {self};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  Tensor result = at::empty_with_format(
      self.sizes(),
      self.options().dtype(ScalarType::Byte),
      CalcuOpUtil::get_tensor_npu_format(self));

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    logical_not_out_npu(result, contiguousSelf);
  } else {
    logical_not_out_npu(result, self);
  }
  // uint8 to self dtype
  self.npu_dtype_cast_(result);

  return self;
}

} // namespace native
} // namespace at