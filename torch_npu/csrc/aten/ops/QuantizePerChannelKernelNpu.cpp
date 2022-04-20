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

c10::SmallVector<int64_t, SIZE> quantize_reshape_size(
    const at::Tensor& self,
    int64_t axis) {
  c10::SmallVector<int64_t, SIZE> outSize;
  for(int64_t i=0; i < self.dim(); i++) {
    if(i != axis) {
      outSize.emplace_back(1);
    } else {
      outSize.emplace_back(self.sizes()[i]);
    }
  }
  return outSize;
}

at::Tensor& quantize_per_channel_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    int64_t axis,
    at::ScalarType dtype) {
  auto reshapeSize = quantize_reshape_size(self, axis);
  at::Tensor scales_reshape = scales.reshape(reshapeSize);
  at::Tensor zp_reshape = zero_points.reshape(reshapeSize);
  at::Tensor scales_broadcast = NPUNativeFunctions::npu_broadcast(scales_reshape, self.sizes());
  at::Tensor zp_broadcast = NPUNativeFunctions::npu_broadcast(zp_reshape, self.sizes());
  string dtypeStr = "torch.qint8";
  if (dtype == at::ScalarType::QUInt8) {
    dtypeStr = "torch.quint8";
  } else if (dtype == at::ScalarType::QInt32) {
    dtypeStr = "torch.qint32";
  }
  OpCommand cmd;
  cmd.Name("Quantize")
     .Input(self)
     .Input(scales_broadcast)
     .Input(zp_broadcast)
     .Output(result)
     .Attr("axis", axis)
     .Attr("dtype", dtypeStr)
     .Run();
  return result;
}

at::Tensor NPUNativeFunctions::quantize_per_channel(
    const at::Tensor& self,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    int64_t axis,
    at::ScalarType dtype) {
  axis = CalcuOpUtil::make_wrap_dim(axis, self.dim());
  TORCH_CHECK(scales.dim() == 1, "Scales' dim should be equal to 1.");
  TORCH_CHECK(zero_points.dim() == 1, "Zero points' dim should be equal to 1.");
  TORCH_CHECK(scales.sizes()[0] == zero_points.sizes()[0], "Scales' size should be equal to zero points' size.");
  TORCH_CHECK(scales.sizes()[0] == self.sizes()[axis], "length of scales must equal to the specified dimension.");
  auto outputSize = input_same_output_size(self);
  auto outputDtype = at::kInt;
  if (dtype == at::ScalarType::QInt8) {
    outputDtype = at::kChar;
  } else if (dtype == at::ScalarType::QUInt8) {
    outputDtype = at::kByte;
  } else if (dtype == at::ScalarType::QInt32) {
    outputDtype = at::kInt;
  }
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      self.options().dtype(outputDtype),
      CalcuOpUtil::get_tensor_npu_format(self));
  quantize_per_channel_out_nocheck(result, self, scales, zero_points, axis, dtype);
  return result;
}

} // namespace native
} // namespace at_npu