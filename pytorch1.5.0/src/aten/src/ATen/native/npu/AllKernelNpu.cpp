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
#include "c10/npu/OptionsManager.h"

namespace at {
namespace native {
using namespace at::native::npu;

inline Tensor all_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    SmallVector<int64_t, N> dimList,
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

Tensor& all_out_npu(
    Tensor& result,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  SmallVector<int64_t, N> dimList = {dim};
  
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

Tensor all_npu(const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.scalar_type() == ScalarType::Bool || self.scalar_type() == ScalarType::Byte,
      "all only supports torch.uint8 and torch.bool dtypes");
   TORCH_CHECK(dim >= -(self.dim()) && dim < self.dim(),
       "The value of dim must be greater than or equal to -self.dim() and less than self.dim()");
  Tensor selfCopy = self;
  if(selfCopy.scalar_type() == ScalarType::Byte){
    selfCopy = selfCopy.npu_dtype_cast(ScalarType::Bool);
  }
  if (self.numel() == 0) {
    SmallVector<int64_t, N> outputSize;
    for(int64_t i = 0; i < self.dim(); i++){
        if(dim != i){
            outputSize.emplace_back(self.size(i));
        }
    }
    Tensor res = OpPreparation::ApplyTensorWithFormat(
        outputSize,
        self.options().dtype(kInt), 
        CalcuOpUtil::get_tensor_npu_format(self)).fill_(1).npu_dtype_cast(self.scalar_type());
    return res;
  }

  // calculate the output size
  IntArrayRef dims(dim);
  auto outputSize = reduce_ops_npu_output_size(selfCopy, dims, keepdim);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, selfCopy.options(), CalcuOpUtil::get_tensor_npu_format(selfCopy));

  // calculate the output result of the NPU
  all_out_npu_nocheck(result, selfCopy, {dim}, keepdim);
  if(self.scalar_type() == ScalarType::Byte){
    result = result.npu_dtype_cast(ScalarType::Byte);
  }
  return result;
}

Tensor all_npu(const Tensor& self) {
  TORCH_CHECK(self.scalar_type() == ScalarType::Bool || self.scalar_type() == ScalarType::Byte,
      "all only supports torch.uint8 and torch.bool dtypes");
  Tensor selfCopy = self;
  if(selfCopy.scalar_type() == ScalarType::Byte){
    selfCopy = selfCopy.npu_dtype_cast(ScalarType::Bool);
  }

  if (self.numel() == 0) {
    Tensor res = OpPreparation::ApplyTensorWithFormat(
        {}, 
        self.options().dtype(kInt), 
        CalcuOpUtil::get_tensor_npu_format(self)).fill_(1).npu_dtype_cast(self.scalar_type());
    return res;
  }

  // calculate the output size
  IntArrayRef dims;
  auto outputSize = reduce_ops_npu_output_size(selfCopy, dims, false);
  
  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, selfCopy.options(), CalcuOpUtil::get_tensor_npu_format(selfCopy));

  // calculate the output result of the NPU
  all_out_npu_nocheck(
      result,
      selfCopy,
      CalcuOpUtil::get_dimlist_for_tensor(selfCopy),
      false);

  if(self.scalar_type() == ScalarType::Byte){
    result = result.npu_dtype_cast(ScalarType::Byte);
  }

  return result;
}

} // namespace native
} // namespace at