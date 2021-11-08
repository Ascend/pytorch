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
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include <torch/csrc/autograd/record_function.h>

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& transpose_out_npu(
    Tensor& result,
    const Tensor& self,
    IntArrayRef perm) {
  if (!c10::npu::OptionsManager::CheckDynamicEnable()){
    OpCommand cmd;
    cmd.Name("Transpose")
        .Input(self)
        .Input(perm)
        .Output(result)
        .Run();
  } else {
    SmallVector<int64_t, N> permVec = array_to_small_vector(perm);
    OpDynamicCommand cmd;
    cmd.Name("TransposeD")
       .Input(self)
       .Output(result)
       .Attr("perm", perm);
    cmd.DynamicName("Transpose")
        .DynamicInput(self)
        .DynamicInput(permVec, at::kLong, at::kInt, "perm")
        .DynamicOutput(result)
        .DynamicOpRun();
  }
  return result;
}

Tensor transpose_npu(const Tensor& self, IntArrayRef perm) {
  auto outputSize = transpose_npu_output_size(self, perm);
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  transpose_out_npu(result, self, perm);

  return result;
}

SmallVector<NPUTensorDesc, N> transpose_to_contiguous_npu_input(
    const SmallVector<Tensor, N>& src) {

  SmallVector<NPUTensorDesc, N> inputs;
  for (int i = 0; i < src.size(); i++) {
    inputs.emplace_back(
        NPUTensorDesc(src[i]));

    if (src[i].dim() == 0) {
      inputs[i].tensorDescType = NPUTensorDesc::TensorDescType::TENSOR_SCALAR;
    }
  }
  return inputs;
}

SmallVector<NPUTensorDesc, N> transpose_to_contiguous_npu_output(
    const SmallVector<Tensor, N>& result) {
  return CalcuOpUtil::create_npu_output_tensor_desc(result);
}

Tensor transpose_to_contiguous_npu(const Tensor& self) {
  RECORD_HOST_FUNCTION("transpose_to_contiguous", vector<c10::IValue>({self}));
  int64_t self_format = CalcuOpUtil::get_tensor_npu_format(self);
  Tensor result = at::empty_with_format(self.sizes(), self.options(), self_format);

  // obtain the transpose axises
  IntArrayRef dim;
  if ((self.dim() == 2) && (self.stride(self.dim()-2) == 1)) {
    dim = IntArrayRef({1, 0});
  } else if ((self.dim() == 3) && (self.stride(self.dim()-2) == 1)) {
    dim = IntArrayRef({0, 2, 1});
  } else if ((self.dim() == 3) && (self.stride(0) <= self.stride(1))) {
    dim = IntArrayRef({1, 0, 2});
  }
  // constructs the input and output NPUTensorDesc
  auto inputs = transpose_to_contiguous_npu_input({self});
  auto outputs = transpose_to_contiguous_npu_output({result});

  // constructs the attr of the NPUAttrDesc
  NPUAttrDesc npuAttrTranspose = NPUAttrDesc("perm", dim);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrTranspose};

  CalcuOpUtil::execute_npu_operate("TransposeD", inputs, outputs, attrs);
  return result;
}

} // namespace native
} // namespace at