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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

inline at::Tensor all_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    c10::SmallVector<int64_t, N> dimList,
    bool keepdim) {
  OpCommand cmd;
  cmd.Name("ReduceAll")
    .Input(self)
    .Input(dimList, at::kLong)
    .Output(result) 
    .Attr("keep_dims", keepdim)
    .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::all_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& result) {
  c10::SmallVector<int64_t, N> dimList = {dim};
  
  // check result for return
  auto outputSize = reduce_ops_npu_output_size(self, dimList, keepdim);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      self.scalar_type(),
      outputSize);

  // calculate the output result of the NPU    
  all_out_npu_nocheck(
      result, self, dimList, keepdim);

  return result;
}

at::Tensor NPUNativeFunctions::all(const at::Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.scalar_type() == at::ScalarType::Bool || self.scalar_type() == at::ScalarType::Byte,
      "all only supports torch.uint8 and torch.bool dtypes");
  if (self.numel() == 0) {
    at::Tensor res = OpPreparation::ApplyTensor({}, self.options().dtype(at::kInt), self).fill_(1).to(at::ScalarType::Bool); 
    return res;
  }

  // calculate the output size
  at::IntArrayRef dims(dim);
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  // calculate the output result of the NPU
  all_out_npu_nocheck(result, self, {dim}, keepdim);

  return result;
}

at::Tensor NPUNativeFunctions::all(const at::Tensor& self) {
  TORCH_CHECK(self.scalar_type() == at::ScalarType::Bool || self.scalar_type() == at::ScalarType::Byte,
      "all only supports torch.uint8 and torch.bool dtypes");
  if (self.numel() == 0) {
    at::Tensor res = OpPreparation::ApplyTensor({}, self.options().dtype(at::kInt), self).fill_(1).to(at::ScalarType::Bool);
    return res;
  }

  // calculate the output size
  at::IntArrayRef dims;
  auto outputSize = reduce_ops_npu_output_size(self, dims, false);
  
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  // calculate the output result of the NPU
  all_out_npu_nocheck(
      result,
      self,
      CalcuOpUtil::get_dimlist_for_tensor(self),
      false);

  return result;
}

} // namespace native
} // namespace at_npu