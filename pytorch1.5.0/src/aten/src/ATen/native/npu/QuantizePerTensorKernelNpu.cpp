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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& quantize_per_tensor_out_npu(
    Tensor& result, 
    const Tensor& self, 
    const Tensor& scales, 
    const Tensor& zero_points, 
    ScalarType dtype) {
  string dtypeStr = "torch.qint8";
  if (dtype == ScalarType::QInt8) {
    dtypeStr = "torch.qint8";
  } else if (dtype == ScalarType::QUInt8) {
    dtypeStr = "torch.quint8";
  } else if (dtype == ScalarType::QInt32) {
    dtypeStr = "torch.qint32";
  }

  OpCommand cmd;
  cmd.Name("Quantize")
      .Input(self)
      .Input(scales)
      .Input(zero_points)
      .Output(result)
      .Attr("axis", (int64_t)1)
      .Attr("dtype", dtypeStr)
      .Run();
  return result;
}

Tensor quantize_per_tensor_npu(
    const Tensor& self, 
    double scale, 
    int64_t zero_point, 
    ScalarType dtype) {
  // constructs the input and output NPUTensorDesc
  float scaleFloat = static_cast<float>(scale);
  auto outputDtype = kInt;
  if (dtype == ScalarType::QInt8) {
    outputDtype = kChar;
  } else if (dtype == ScalarType::QUInt8) {
    outputDtype = kByte;
  } else if (dtype == ScalarType::QInt32) {
    outputDtype = kInt;
  }
  Tensor scaleTensor = OpPreparation::ApplyTensor(
      {1},
      self.options().dtype(kFloat),
      self);
  scaleTensor[0] = scaleFloat;
  Tensor zpTensor = OpPreparation::ApplyTensor(
      {1},
      self.options().dtype(kInt),
      self);
  zpTensor[0] = zero_point;
  Tensor result = OpPreparation::ApplyTensor(
      self,
      self.options().dtype(outputDtype));
  quantize_per_tensor_out_npu(result, self, scaleTensor, zpTensor, dtype);
  return result;
}

} // namespace native
} // namespace at