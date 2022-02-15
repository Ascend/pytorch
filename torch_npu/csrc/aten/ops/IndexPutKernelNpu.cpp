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

#include <ATen/native/IndexingUtils.h>

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& index_put_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::TensorList& indices,
    const at::Tensor& value,
    bool accumulate) {
  if (value.numel() == 0) {
    return result;
  }

  at::SmallVector<int64_t, N> masks;
  std::vector<at::Tensor> allDefinedIndices;
  for (int64_t i = 0; i < indices.size(); i++) {
    if (indices[i].defined()) {
      masks.emplace_back(1);
      allDefinedIndices.emplace_back(indices[i]);
    } else {
      masks.emplace_back(0);
    }
  }

  auto masksTensor = CalcuOpUtil::copy_tensor_host_to_device(
      at::from_blob(masks.data(), {masks.size()}, dtype(at::ScalarType::Long)));

  at::Tensor tempSelf = self;
  at::Tensor tempValue = value;
  if (self.scalar_type() == at::ScalarType::Half) {
    tempSelf = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
    tempValue = NPUNativeFunctions::npu_dtype_cast(value, at::ScalarType::Float);
    result = NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Float);
  }

  OpCommand cmd;
  cmd.Name("IndexPut")
      .Input(tempSelf)
      .Input(tempValue)
      .Input(masksTensor)
      .Inputs(allDefinedIndices)
      .Output(result)
      .Attr("accumulate", accumulate)
      .Run();

  if (self.scalar_type() == at::ScalarType::Half) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Half);
  }
  return result;
}

at::Tensor NPUNativeFunctions::index_put(
    const at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>> & indices,
    const at::Tensor& value,
    bool accumulate) {
  return self.clone(at::MemoryFormat::Contiguous)
      .index_put_(indices, value, accumulate);
}

at::Tensor& NPUNativeFunctions::index_put_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>> & indices,
    const at::Tensor& value,
    const bool accumulate) {
  return at::_index_put_impl_(
      self, indices, value, accumulate, false);
}

at::Tensor& NPUNativeFunctions::_index_put_impl_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>> & indices,
    const at::Tensor& value,
    const bool accumulate,
    const bool unsafe) {
  at::native::checkIndexTensorTypes(indices);
  auto indices_cast = at::native::expandTensors(self, indices);
  
  OpPreparation::CastBackToOriFormat(self);
  at::Tensor valueCopy = value;
  at::Tensor selfCopy = self;
  OpPreparation::CastBackToOriFormat(valueCopy);

  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(selfCopy);
    at::Tensor result = index_put_nocheck(
        contiguousSelf, contiguousSelf, indices_cast, valueCopy, accumulate);
    self.copy_(result);
  } else {
    index_put_nocheck(selfCopy, selfCopy, indices_cast, valueCopy, accumulate);
    self.copy_(selfCopy);
  }
  return self;
}
} // namespace native
} // namespace at_npu