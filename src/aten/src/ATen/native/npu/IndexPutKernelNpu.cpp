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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& index_put_nocheck(
    Tensor& result,
    const Tensor& self,
    const TensorList& indices,
    const Tensor& value,
    bool accumulate) {
  if (value.numel() == 0) {
    return result;
  }
  // masks corresponds to indices. 0 indicates undefined tensor.
  SmallVector<int64_t, N> masks;
  std::vector<Tensor> allDefinedIndices;
  for (int64_t i = 0; i < indices.size(); i++) {
    if (indices[i].defined()) {
      masks.emplace_back(1);
      allDefinedIndices.emplace_back(indices[i]);
    } else {
      masks.emplace_back(0);
    }
  }

  auto masksTensor = CalcuOpUtil::copy_tensor_host_to_device(
      from_blob(masks.data(), {masks.size()}, dtype(ScalarType::Long)));

  OpCommand cmd;
  cmd.Name("IndexPut")
      .Input(self)
      .Input(value)
      .Input(masksTensor)
      .Inputs(allDefinedIndices)
      .Output(result)
      .Attr("accumulate", accumulate)
      .Run();

  return result;
}

Tensor index_put_npu(
    const Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate) {
  return self.clone(at::MemoryFormat::Contiguous)
      .index_put_(indices, value, accumulate);
}

Tensor& index_put_npu_(
    Tensor& self,
    TensorList indices,
    const Tensor& value,
    const bool accumulate) {
  return at::_index_put_impl_(
      self, indices, value, accumulate, /*unsafe=*/false);
}

Tensor& _index_put_impl_npu_(
    Tensor& self,
    TensorList indices,
    const Tensor& value,
    const bool accumulate,
    const bool unsafe) {
  OpPreparation::CheckMemory({self}, {self});
  OpPreparation::CastBackToOriFormat(self);

  Tensor valueCopy = value;
  Tensor selfCopy = self;
  OpPreparation::CastBackToOriFormat(valueCopy);

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(selfCopy);
    Tensor result = index_put_nocheck(
        contiguousSelf, contiguousSelf, indices, valueCopy, accumulate);
    self.copy_(result);
  } else {
    index_put_nocheck(selfCopy, selfCopy, indices, valueCopy, accumulate);
    self.copy_(selfCopy);
  }
  return self;
}

} // namespace native
} // namespace at