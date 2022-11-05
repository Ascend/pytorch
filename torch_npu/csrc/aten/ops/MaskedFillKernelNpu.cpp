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
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& masked_fill_out_npu_nocheck(const at::Tensor& self, const at::Tensor& mask, const at::Tensor& value, at::Tensor& result) {
  at::Tensor maskBool = mask;
  int64_t dimOfSelf = self.dim();

  /* Avoid the problem that the TBE operator does not support 0-dimensional tensor input */
  if (dimOfSelf == 0) {
    self.unsqueeze_(0);
  }

  if ((mask.dtype() != at::kBool)) {
    maskBool = NPUNativeFunctions::npu_dtype_cast(mask, at::kBool);
  }
  at::Tensor valueTensor = value;
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

at::Tensor& masked_fill_out_npu_nocheck(const at::Tensor& self, const at::Tensor& mask, at::Scalar value, at::Tensor& result) {
  at::Tensor maskBool = mask;
  int64_t dimOfSelf = self.dim();

  /* Avoid the problem that the TBE operator does not support 0-dimensional tensor input */
  if (dimOfSelf == 0) {
    self.unsqueeze_(0);
  }

  if (!(mask.dtype() == at::kBool)) {
    maskBool = NPUNativeFunctions::npu_dtype_cast(mask, at::kBool);
  }

  OpCommand cmd;
  cmd.Name("MaskedFill")
    .Input(self)
    .Input(maskBool)
    .Input(value, self.scalar_type())
    .Output(result)
    .Run();
  
  if (dimOfSelf == 0) {
    result.squeeze_(0);
  }
  return result;
}

at::Tensor& NPUNativeFunctions::masked_fill_(at::Tensor& self, const at::Tensor& mask, const at::Tensor& value) {
  // OpPreparation::CheckMemory({self, mask, value}, {self});
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = masked_fill_out_npu_nocheck(contiguousSelf, mask, value, contiguousSelf);
    self.copy_(result);
  } else {
    masked_fill_out_npu_nocheck(self, mask, value, self);
  }
  return self;
}

at::Tensor& NPUNativeFunctions::masked_fill_(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = masked_fill_out_npu_nocheck(contiguousSelf, mask, value, contiguousSelf);
    self.copy_(result);
  } else {
    masked_fill_out_npu_nocheck(self, mask, value, self);
  }

  return self;
}

} // namespace native
} // namespace at_npu
