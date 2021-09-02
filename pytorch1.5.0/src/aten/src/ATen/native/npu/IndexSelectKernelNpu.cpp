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

Tensor& index_select_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  if (self.scalar_type() == at::kLong) {
    TORCH_WARN_ONCE("The oprator of index_select is executed, Currently High Accuracy but Low Performance OP with 64-bit has been used,"
      "Please Do Some Cast at Python Functions with 32-bit for Better Performance!");
  }
  SmallVector<int64_t, N> dimVec = {dim};
  if (!c10::npu::OptionsManager::CheckDynamicEnable()){
    OpCommand cmd;
    cmd.Name("GatherV2")
        .Input(self)
        .Input(index)
        .Input(dimVec, at::kInt)
        .Output(result)    
        .Run();
  } else {
    OpDynamicCommand cmd;
    cmd.Name("GatherV2D")
        .Input(self)
        .Input(index)
        .Output(result)
        .Attr("axis", dim);
    // DYNAMIC
    Tensor dimTensor_cpu = from_blob((void*)dimVec.data(), {1}, at::kLong).to(at::kInt);
    Tensor dimTensor_npu = CalcuOpUtil::copy_tensor_host_to_device(dimTensor_cpu);
    cmd.DynamicName("GatherV2")
       .DynamicInput(self)
       .DynamicInput(index)
       .DynamicInput(dimTensor_npu, "axis")
       .DynamicOutput(result)
       .DynamicOpRun();
  }

  return result;
}

Tensor& index_select_out_npu(
    Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  Tensor indexTmp(index);
  if (indexTmp.ndimension() == 0) {
    indexTmp = index.unsqueeze(0);
  }
  // calculate the output size
  auto outputSize = index_select_npu_output_size(self, dim, indexTmp);

  int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(self);
  // scalar scene no support nz
  if (outputSize.empty()) {
    npu_format = ACL_FORMAT_ND;
  }

  Tensor input = self;
  if (self.dtype() == kBool) {
    // bool to int dtype
    input = input.npu_dtype_cast(at::kInt);
  }

  OpPreparation::CheckOut(
      {input},
      result,
      npu_format,
      input.scalar_type(),
      outputSize);

  OpPipeWithDefinedOut pipe;
  result = pipe.CheckMemory({input, indexTmp}, {result})
      .Func([&input, &dim, &indexTmp](Tensor& result)
      {index_select_out_npu_nocheck(result, input, dim, indexTmp);})
      .Call(result);

  if (self.dtype() == kBool) {
    result = result.to(kBool);
  }

  return result;
}

Tensor index_select_npu(const Tensor& self, int64_t dim, const Tensor& index) {
  Tensor indexTmp(index);
  if (indexTmp.ndimension() == 0) {
    indexTmp = index.unsqueeze(0);
  }
  // calculate the output size
  auto outputSize = index_select_npu_output_size(self, dim, indexTmp);

  int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(self);
  // scalar scene no support nz
  if (outputSize.empty()) {
    npu_format = ACL_FORMAT_ND;
  }

  Tensor input = self;
  if (self.dtype() == kBool) {
    // bool to int dtype
    input = input.npu_dtype_cast(at::kInt);
  }

  // construct the output tensor of the NPU
  Tensor result =
      at::empty_with_format(outputSize, input.options(), npu_format);

  // calculate the output result of the NPU
  index_select_out_npu_nocheck(result, input, dim, indexTmp);

  if (self.dtype() == kBool) {
    // int to bool dtype  这里不转变回bool也能通过测试的比较
    result = result.to(kBool);
  }

  return result;
}

Tensor& index_select_out_npu(
    Tensor& result,
    const Tensor& self,
    Dimname dim,
    const Tensor& index) {
  Tensor indexTmp(index);
  if (indexTmp.ndimension() == 0) {
    indexTmp = index.unsqueeze(0);
  }
  return index_select_out_npu(
      result, self, dimname_to_position(self, dim), indexTmp);
}

Tensor index_select_npu(const Tensor& self, Dimname dim, const Tensor& index) {
  return index_select_npu(self, dimname_to_position(self, dim), index);
}

} // namespace native
} // namespace at
