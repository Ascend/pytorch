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

SmallVector<int64_t, SIZE> masked_select_npu_output_size(
    const Tensor& self,
    const Tensor& mask) {
  int64_t shape;
  shape = mask.sum().item().toInt();
  return {shape};
}

Tensor& masked_select_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& mask) {
  Tensor maskBool = mask;
  if (!(mask.dtype() == at::kBool)) {
    maskBool = mask.to(at::kBool);
  }

  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("MaskedSelect")
      .Input(self)
      .Input(maskBool)
      .Output(result)
      .Run();

  return result;
}

Tensor& masked_select_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& mask) {
  Tensor dtypeCastOfSelf = self;
  Tensor maskCast = mask;
  if (maskCast.sizes() != dtypeCastOfSelf.sizes()) {
    maskCast = broadcast_npu(mask, dtypeCastOfSelf.sizes());
  }
  if (dtypeCastOfSelf.scalar_type() == ScalarType::Half) {
    dtypeCastOfSelf = dtypeCastOfSelf.npu_dtype_cast(ScalarType::Float);
    result = result.to(ScalarType::Float);
  }
  auto outputSize = masked_select_npu_output_size(dtypeCastOfSelf, maskCast);

  OpPreparation::CheckOut(
      {dtypeCastOfSelf},
      result,
      dtypeCastOfSelf,
      outputSize);

  OpPipeWithDefinedOut pipe;
  result = pipe.CheckMemory({dtypeCastOfSelf, maskCast}, {result})
      .Func([&dtypeCastOfSelf, &maskCast](Tensor& result)
      {masked_select_out_npu_nocheck(result, dtypeCastOfSelf, maskCast);})
      .Call(result);

  if (result.scalar_type() != self.scalar_type()) {
    result = result.npu_dtype_cast(ScalarType::Half);
  }
  return result;
}

Tensor masked_select_npu(
    const Tensor& self,
    const Tensor& mask) {
  Tensor dtypeCastOfSelf = self;
  Tensor maskCast = mask;
  if (maskCast.sizes() != dtypeCastOfSelf.sizes()) {
    maskCast = broadcast_npu(mask, dtypeCastOfSelf.sizes());
  }
  if (dtypeCastOfSelf.scalar_type() == ScalarType::Half) {
    dtypeCastOfSelf = dtypeCastOfSelf.npu_dtype_cast(ScalarType::Float);
  }
  auto outputSize = masked_select_npu_output_size(dtypeCastOfSelf, maskCast);

  Tensor result = OpPreparation::ApplyTensor(dtypeCastOfSelf, outputSize);

  masked_select_out_npu_nocheck(result, dtypeCastOfSelf, maskCast);

  if (result.scalar_type() != self.scalar_type()) {
    result = result.npu_dtype_cast(ScalarType::Half);
  }
  return result;
}

} // namespace native
} // namespace at
