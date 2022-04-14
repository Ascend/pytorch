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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& quantize_per_tensor_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    at::ScalarType dtype) {
  string dtypeStr = "torch.qint8";
  if (dtype == at::ScalarType::QUInt8) {
    dtypeStr = "torch.quint8";
  } else if (dtype == at::ScalarType::QInt32) {
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

at::Tensor NPUNativeFunctions::quantize_per_tensor(
    const at::Tensor& self,
    double scale,
    int64_t zero_point,
    at::ScalarType dtype) {
  float scaleFloat = static_cast<float>(scale);
  auto outputSize = input_same_output_size(self);
  auto outputDtype = at::kInt;
  if (dtype == at::ScalarType::QInt8) {
    outputDtype = at::kChar;
  } else if (dtype == at::ScalarType::QUInt8) {
    outputDtype = at::kByte;
  } else if (dtype == at::ScalarType::QInt32) {
    outputDtype = at::kInt;
  }
  at::Tensor scaleTensor = OpPreparation::ApplyTensorWithFormat(
      {1},
      self.options().dtype(at::kFloat),
      CalcuOpUtil::get_tensor_npu_format(self));
  scaleTensor[0] = scaleFloat;
  at::Tensor zpTensor = OpPreparation::ApplyTensorWithFormat(
      {1},
      self.options().dtype(at::kInt),
      CalcuOpUtil::get_tensor_npu_format(self));
  zpTensor[0] = zero_point;
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      self.options().dtype(outputDtype),
      CalcuOpUtil::get_tensor_npu_format(self));
  quantize_per_tensor_out_nocheck(result, self, scaleTensor, zpTensor, dtype);
  return result;
}

} // namespace native
} // namespace at_npu