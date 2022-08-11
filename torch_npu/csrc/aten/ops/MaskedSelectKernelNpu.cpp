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
#include "torch_npu/csrc/aten/XLANativeFunctions.h"

namespace at_npu {
namespace native {

at::SmallVector<int64_t, SIZE> masked_select_npu_output_size(
    const at::Tensor& self,
    const at::Tensor& mask) {
  int64_t shape;
  shape = mask.sum().item().toInt();
  return {shape};
}

at::Tensor& masked_select_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& mask) {
  at::Tensor maskBool = mask;
  if (!(mask.dtype() == at::kBool)) {
    maskBool = XLANativeFunctions::npu_dtype_cast(mask, at::kBool);
  }

  OpCommand cmd;
  cmd.Name("MaskedSelect")
      .Input(self)
      .Input(maskBool)
      .Output(result)
      .Run();

  return result;
}

at::Tensor& XLANativeFunctions::masked_select_out(
    const at::Tensor& self,
    const at::Tensor& mask,
    at::Tensor& result) {
  at::Tensor dtypeCastOfSelf = self;
  at::Tensor maskCast = mask;
  if (maskCast.sizes() != dtypeCastOfSelf.sizes()) {
    maskCast = XLANativeFunctions::npu_broadcast(mask, dtypeCastOfSelf.sizes());
  }
  if (dtypeCastOfSelf.scalar_type() == at::ScalarType::Half) {
    dtypeCastOfSelf = XLANativeFunctions::npu_dtype_cast(dtypeCastOfSelf, at::ScalarType::Float);
    result = XLANativeFunctions::npu_dtype_cast(result, at::ScalarType::Float);
  }
  auto outputSize = masked_select_npu_output_size(dtypeCastOfSelf, maskCast);

  OpPreparation::CheckOut(
      {dtypeCastOfSelf},
      result,
      dtypeCastOfSelf,
      outputSize);

  OpPipeWithDefinedOut pipe;
  result = pipe.CheckMemory({dtypeCastOfSelf, maskCast}, {result})
      .Func([&dtypeCastOfSelf, &maskCast](at::Tensor& result)
      {masked_select_out_npu_nocheck(result, dtypeCastOfSelf, maskCast);})
      .Call(result);

  if (result.scalar_type() != self.scalar_type()) {
    result = XLANativeFunctions::npu_dtype_cast(result, at::ScalarType::Half);
  }
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    masked_select_out_npu_nocheck(contiguousResult, self, mask);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    masked_select_out_npu_nocheck(result, self, mask);
  }
  return result;
}

at::Tensor XLANativeFunctions::masked_select(
    const at::Tensor& self,
    const at::Tensor& mask) {
  at::Tensor dtypeCastOfSelf = self;
  at::Tensor maskCast = mask;
  if (maskCast.sizes() != dtypeCastOfSelf.sizes()) {
    maskCast = XLANativeFunctions::npu_broadcast(mask, dtypeCastOfSelf.sizes());
  }
  if (dtypeCastOfSelf.scalar_type() == at::ScalarType::Half) {
    dtypeCastOfSelf = XLANativeFunctions::npu_dtype_cast(dtypeCastOfSelf, at::ScalarType::Float);
  }
  auto outputSize = masked_select_npu_output_size(dtypeCastOfSelf, maskCast);

  at::Tensor result = OpPreparation::ApplyTensor(dtypeCastOfSelf, outputSize);

  masked_select_out_npu_nocheck(result, dtypeCastOfSelf, maskCast);

  if (result.scalar_type() != self.scalar_type()) {
    result = XLANativeFunctions::npu_dtype_cast(result, at::ScalarType::Half);
  }
  return result;
}

} // namespace native
} // namespace at_npu
