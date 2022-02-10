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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& index_select_out_npu_nocheck(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    at::Tensor& result) {
  if (self.scalar_type() == at::kLong) {
    TORCH_WARN_ONCE("The oprator of index_select is executed, Currently High Accuracy but Low Performance OP with 64-bit has been used,"
      "Please Do Some Cast at Python Functions with 32-bit for Better Performance!");
  }
  c10::SmallVector<int64_t, N> dimVec = {dim};
  OpCommand cmd;
  cmd.Name("GatherV2")
      .Input(self)
      .Input(index)
      .Input(dimVec, at::kInt)
      .Output(result)    
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::index_select_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    at::Tensor& result) {
  at::Tensor indexTmp(index);
  if (indexTmp.ndimension() == 0) {
    indexTmp = index.unsqueeze(0);
  }
  auto outputSize = index_select_npu_output_size(self, dim, indexTmp);
  int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(self);
  if (outputSize.empty()) {
    npu_format = ACL_FORMAT_ND;
  }
  at::Tensor input = self;
  if (self.dtype() == at::kBool) {
    input = NPUNativeFunctions::npu_dtype_cast(input, at::kInt);
  }
  OpPreparation::CheckOut(
      {input},
      result,
      npu_format,
      input.scalar_type(),
      outputSize);
  OpPipeWithDefinedOut pipe;
  result = pipe.CheckMemory({input, indexTmp}, {result})
      .Func([&input, &dim, &indexTmp](at::Tensor& result)
      {index_select_out_npu_nocheck(input, dim, indexTmp, result);})
      .Call(result);
  if (self.dtype() == at::kBool) {
    result = result.to(at::kBool);
  }
  return result;
}

at::Tensor NPUNativeFunctions::index_select(
    const at::Tensor& self, 
    int64_t dim, 
    const at::Tensor& index) {
  at::Tensor indexTmp(index);
  if (indexTmp.ndimension() == 0) {
    indexTmp = index.unsqueeze(0);
  }
  auto outputSize = index_select_npu_output_size(self, dim, indexTmp);
  int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(self);
  if (outputSize.empty()) {
    npu_format = ACL_FORMAT_ND;
  }
  at::Tensor input = self;
  if (self.dtype() == at::kBool) {
    input = NPUNativeFunctions::npu_dtype_cast(input, at::kInt);
  }
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(input, outputSize, npu_format);
  index_select_out_npu_nocheck(input, dim, indexTmp, result);
  if (self.dtype() == at::kBool) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::kBool);
  }
  return result;
}

at::Tensor& NPUNativeFunctions::index_select_out(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    at::Tensor& result) {
  at::Tensor indexTmp(index);
  if (indexTmp.ndimension() == 0) {
    indexTmp = index.unsqueeze(0);
  }
  return index_select_out(
      self, dimname_to_position(self, dim), indexTmp, result);
}

at::Tensor NPUNativeFunctions::index_select(
    const at::Tensor& self, 
    at::Dimname dim, 
    const at::Tensor& index) {
  return index_select(self, dimname_to_position(self, dim), index);
}
} // namespace native
} // namespace at_npu
