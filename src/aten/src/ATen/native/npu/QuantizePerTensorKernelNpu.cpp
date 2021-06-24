// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUTensorDesc, N> quantize_per_tensor_npu_input(
    const SmallVector<Tensor, N>& self) {
  return CalcuOpUtil::create_npu_input_tensor_desc(self);
}

SmallVector<NPUTensorDesc, N> quantize_per_tensor_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> quantize_per_tensor_npu_attr(ScalarType dtype) {
  NPUAttrDesc npuAttrAxis = NPUAttrDesc("axis", (int64_t)1);
  string dtypeStr = "torch.qint8";
  if (dtype == ScalarType::QInt8) {
    dtypeStr = "torch.qint8";
  } else if (dtype == ScalarType::QUInt8) {
    dtypeStr = "torch.quint8";
  } else if (dtype == ScalarType::QInt32) {
    dtypeStr = "torch.qint32";
  }
  NPUAttrDesc npuAttrDtype = NPUAttrDesc("dtype", dtypeStr);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrAxis, npuAttrDtype};

  return attrs;
}

Tensor& quantize_per_tensor_out_npu(
    Tensor& result, 
    const Tensor& self, 
    const Tensor& scales, 
    const Tensor& zero_points, 
    ScalarType dtype) {
  // constructs the input and output NPUTensorDesc
  auto inputs = quantize_per_tensor_npu_input({self, scales, zero_points});
  auto outputs = quantize_per_tensor_npu_output({result});

  // constructs the attr of the NPUAttrDesc
  auto attrs = quantize_per_tensor_npu_attr(dtype);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("Quantize", inputs, outputs, attrs);

  return result;
}

Tensor quantize_per_tensor_npu(
    const Tensor& self, 
    double scale, 
    int64_t zero_point, 
    ScalarType dtype) {
  // constructs the input and output NPUTensorDesc
  float scaleFloat = (float)scale;
  auto outputSize = input_same_output_size(self);
  auto outputDtype = kInt;
  if (dtype == ScalarType::QInt8) {
    outputDtype = kChar;
  } else if (dtype == ScalarType::QUInt8) {
    outputDtype = kByte;
  } else if (dtype == ScalarType::QInt32) {
    outputDtype = kInt;
  }
  Tensor scaleTensor = at::empty_with_format(
      {1},
      self.options().dtype(kFloat),
      CalcuOpUtil::get_tensor_npu_format(self));
  scaleTensor[0] = scaleFloat;
  Tensor zpTensor = at::empty_with_format(
      {1},
      self.options().dtype(kInt),
      CalcuOpUtil::get_tensor_npu_format(self));
  zpTensor[0] = zero_point;
  Tensor result = at::empty_with_format(
      outputSize,
      self.options().dtype(outputDtype),
      CalcuOpUtil::get_tensor_npu_format(self));
  quantize_per_tensor_out_npu(result, self, scaleTensor, zpTensor, dtype);
  return result;
}

} // namespace native
} // namespace at