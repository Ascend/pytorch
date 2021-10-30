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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& masked_fill_out_npu(Tensor& result, const Tensor& self, const Tensor& mask, const Tensor& value) {
  Tensor maskBool = mask;
  int64_t dimOfSelf = self.dim();

  /* Avoid the problem that the TBE operator does not support 0-dimensional tensor input */
  if (dimOfSelf == 0) {
    self.unsqueeze_(0);
  }

  if (!(mask.dtype() == at::kBool)) {
    maskBool = mask.to(at::kBool);
  }
  Tensor valueTensor = value;
  if (value.dtype() != self.dtype()) {
    valueTensor = valueTensor.to(self.dtype());
  }

  OpCommand cmd;
  cmd.Name("MaskedFill")
      .Input(self)
      .Input(maskBool)
      .Input(valueTensor)      
      .Output(result)
      .Run();
  
  if (dimOfSelf == 0) {
    result.squeeze_(0);
  }
  
  return result;
}

Tensor& masked_fill_out_npu(Tensor& result, const Tensor& self, const Tensor& mask, Scalar value) {
  Tensor maskBool = mask;
  int64_t dimOfSelf = self.dim();

  /* Avoid the problem that the TBE operator does not support 0-dimensional tensor input */
  if (dimOfSelf == 0) {
    self.unsqueeze_(0);
  }

  if (!(mask.dtype() == at::kBool)) {
    maskBool = mask.to(at::kBool);
  }

  OpCommand cmd;
  if (c10::npu::OptionsManager::CheckDynamicOptimizer("MASKFill")) {
    cmd.Name("MaskedFill")
      .Input(self)
      .Input(maskBool)
      .Input(value, self.scalar_type(), MemoryType::MEMORY_HOST)
      .Output(result)
      .Run();
  } else {
    cmd.Name("MaskedFill")
      .Input(self)
      .Input(maskBool)
      .Input(value, self.scalar_type())      
      .Output(result)
      .Run();
  }
  
  if (dimOfSelf == 0) {
    result.squeeze_(0);
  }
  return result;
}

Tensor& masked_fill_npu_(Tensor& self, const Tensor& mask, const Tensor& value) {
  OpPreparation::CheckMemory({self, mask, value}, {self});
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = masked_fill_out_npu(contiguousSelf, contiguousSelf, mask, value);
    self.copy_(result);
  } else {
    masked_fill_out_npu(self, self, mask, value);
  }
  return self;
}

Tensor& masked_fill_npu_(Tensor& self, const Tensor& mask, Scalar value) {
  OpPreparation::CheckMemory({self, mask}, {self});

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = masked_fill_out_npu(contiguousSelf, contiguousSelf, mask, value);
    self.copy_(result);
  } else {
    masked_fill_out_npu(self, self, mask, value);
  }

  return self;
}
} // namespace native
} // namespace at