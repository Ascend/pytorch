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
  if (!c10::npu::OptionsManager::CheckDynamicEnable()){
    OpCommand cmd;
    cmd.Name("ReduceAll")
      .Input(self)
      .Input(dimList, at::kLong)
      .Output(result) 
      .Attr("keep_dims", keepdim)
      .Run();
  } else {
    OpDynamicCommand cmd;
    cmd.Name("ReduceAllD")
      .Input(self)
      .Output(result)
      .Attr("axes", dimList)
      .Attr("keep_dims", keepdim);
    //  DYNAMIC
    Tensor dimTensor_cpu = from_blob((void*)dimList.data(), {dimList.size()}, at::kLong).to(at::kInt);
    Tensor dimTensor_npu = CalcuOpUtil::copy_tensor_host_to_device(dimTensor_cpu);
    cmd.DynamicName("ReduceAll")
       .DynamicInput(self)
       .DynamicInput(dimTensor_npu)
       .DynamicOutput(result, "", FIXED_NONE, false)
       .DynamicAttr("keep_dims", keepdim)
       .DynamicOpRun();

  }
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
  if (self.numel() == 0) {
    Tensor res = at::empty_with_format({}, self.options().dtype(kInt), CalcuOpUtil::get_tensor_npu_format(self)).fill_(1).to(ScalarType::Bool);
    return res;
  }

  // calculate the output size
  IntArrayRef dims(dim);
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  all_out_npu_nocheck(result, self, {dim}, keepdim);

  return result;
}

Tensor all_npu(const Tensor& self) {
  TORCH_CHECK(self.scalar_type() == ScalarType::Bool || self.scalar_type() == ScalarType::Byte,
          "all only supports torch.uint8 and torch.bool dtypes");
  if (self.numel() == 0) {
    Tensor res = at::empty_with_format(
      {}, 
      self.options().dtype(kInt), 
      CalcuOpUtil::get_tensor_npu_format(self)).fill_(1).to(ScalarType::Bool);
    return res;
  }

  // calculate the output size
  IntArrayRef dims;
  auto outputSize = reduce_ops_npu_output_size(self, dims, false);
  
  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  all_out_npu_nocheck(
      result,
      self,
      CalcuOpUtil::get_dimlist_for_tensor(self),
      false);

  return result;
}

} // namespace native
} // namespace at