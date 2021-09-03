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
#include <torch/script.h>

namespace at {
namespace native {
using namespace at::native::npu;

// 将要废弃，当前仅针对strid_add特殊算子使用该逻辑
SmallVector<int64_t, SIZE> deprecated_broadcast_ops_npu_output_size(
    IntArrayRef shape1_,
    IntArrayRef shape2_) {
  auto shape1 = array_to_small_vector(shape1_);
  auto shape2 = array_to_small_vector(shape2_);

  SmallVector<int64_t, SIZE> output_shape;

  if (shape1.size() < shape2.size()) {
    SmallVector<int64_t, SIZE> shapeTemp = shape1;
    shape1 = shape2;
    shape2 = shapeTemp;
  }

  int64_t shape1_size = shape1.size();
  int64_t shape2_size = shape2.size();
  for (int i = 0; i < shape1_size - shape2_size; i++) {
    shape2.insert(shape2.begin(), 1);
  }

  for (int i = 0; i < shape1_size; i++) {
    if(shape1[i] == 0 || shape2[i] == 0) {
      output_shape.emplace_back((int64_t)0);
    } else {
      output_shape.emplace_back((shape1[i] > shape2[i]) ? shape1[i] : shape2[i]);
    }
  }

  return output_shape;
}

SmallVector<NPUTensorDesc, N> stride_add_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> stride_add_npu_output(const Tensor& result) {
  return CalcuOpUtil::create_npu_output_tensor_desc({result});
}

SmallVector<NPUAttrDesc, N> stride_add_npu_attr(
    Scalar offset1,
    Scalar offset2,
    Scalar c1_len) {
  NPUAttrDesc npuAttrX1 = NPUAttrDesc("x1_c1_offset", (int64_t)offset1.toInt());
  NPUAttrDesc npuAttrX2 = NPUAttrDesc("x2_c1_offset", (int64_t)offset2.toInt());
  NPUAttrDesc npuAttrC1 = NPUAttrDesc("c1_len", (int64_t)c1_len.toInt());
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrX1, npuAttrX2, npuAttrC1};
  return attrs;
}

Tensor& stride_add_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    Scalar offset1,
    Scalar offset2,
    Scalar c1_len) {
  // constructs the input and output NPUTensorDesc
  auto inputs = stride_add_npu_input({self, other});
  auto outputs = stride_add_npu_output({result});

  // constructs the attr of the NPUAttrDesc
  auto attrs = stride_add_npu_attr(offset1, offset2, c1_len);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("StrideAdd", inputs, outputs, attrs);

  return result;
}

Tensor stride_add_npu(
    const Tensor& self,
    const Tensor& other,
    Scalar offset1,
    Scalar offset2,
    Scalar c1_len) {
  // calculate the output size
  auto outputSize = deprecated_broadcast_ops_npu_output_size(self.sizes(), other.sizes());
  outputSize[1] = c1_len.toInt() * 16;

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  stride_add_out_npu(result, self, other, offset1, offset2, c1_len);

  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("npu_stride_add", TORCH_FN(stride_add_npu));
}

Tensor npu_stride_add(  const Tensor& self,
    const Tensor& other,
    Scalar offset1,
    Scalar offset2,
    Scalar c1_len) {
  return at::native::stride_add_npu(self, other, offset1, offset2, c1_len);
}

} // namespace native
} // namespace at
