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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUTensorDesc, N> masked_select_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> masked_select_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

Tensor& masked_select_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& mask) {
  Tensor maskBool = mask;
  if (!(mask.dtype() == at::kBool)) {
    maskBool = mask.to(at::kBool);
  }

  // constructs the input and output NPUTensorDesc
  auto inputs = masked_select_npu_input({self, maskBool});
  auto outputs = masked_select_npu_output({result});

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("MaskedSelect", inputs, outputs, {});

  return result;
}

Tensor masked_select_npu(const Tensor& self, const Tensor& mask) {
  Tensor dtypeCastOfSelf = self;
  Tensor maskCast = mask;

  if (maskCast.sizes() != dtypeCastOfSelf.sizes()) {
    maskCast = broadcast_npu(mask, dtypeCastOfSelf.sizes());
  }

  if (dtypeCastOfSelf.scalar_type() == ScalarType::Half) {
    dtypeCastOfSelf = dtypeCastOfSelf.npu_dtype_cast(ScalarType::Float);
  }
  auto outputSize = masked_select_npu_output_size(dtypeCastOfSelf, maskCast);

  Tensor result = at::empty_with_format(
      outputSize, dtypeCastOfSelf.options(), CalcuOpUtil::get_tensor_npu_format(self));

  masked_select_out_npu(result, dtypeCastOfSelf, maskCast);
  if (result.scalar_type() != self.scalar_type()) {
    result = result.npu_dtype_cast(ScalarType::Half);
  }

  return result;
}


} // namespace native
} // namespace at
