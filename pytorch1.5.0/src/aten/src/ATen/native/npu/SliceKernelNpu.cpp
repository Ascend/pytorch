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

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& slice_out_npu(
    Tensor& result,
    const Tensor& self,
    IntArrayRef offsets,
    IntArrayRef size) {

  SmallVector<int64_t, N> offsetVec = array_to_small_vector(offsets);
  SmallVector<int64_t, N> sizeVec = array_to_small_vector(size);
  
  if (!c10::npu::OptionsManager::CheckDynamicEnable()) {
    OpCommand cmd;
    cmd.Name("Slice")
        .Input(self)
        .Input(offsetVec)
        .Input(sizeVec)
        .Output(result)
        .Run();
  } else {
    SmallVector<int64_t, N> offsetsList = array_to_small_vector(offsets);
    SmallVector<int64_t, N> sizeList = array_to_small_vector(size);
    OpDynamicCommand cmd;
    cmd.Name("SliceD")
        .Input(self)
        .Output(result)
        .Attr("offsets", offsets)
        .Attr("size", size);
    Tensor offsetCpuTensor = from_blob((void*)offsetVec.data(), {offsetVec.size()}, at::kLong).to(at::kInt);
    Tensor offsetNpuTensor = CalcuOpUtil::copy_tensor_host_to_device(offsetCpuTensor);
    Tensor sizeCpuTensor = from_blob((void*)sizeVec.data(), {sizeVec.size()}, at::kLong);
    Tensor sizeNpuTensor = CalcuOpUtil::copy_tensor_host_to_device(sizeCpuTensor);
    cmd.DynamicName("Slice")
        .DynamicInput(self)
        .DynamicInput(offsetsList, at::kLong, at::kInt, "offsets", true, FIXED_CONST_VALUE)
        .DynamicInput(sizeList, at::kLong, at::kInt, "size", true, FIXED_CONST_VALUE)
        .DynamicOutput(result)
        .DynamicOpRun();
  }
  return result;
}

Tensor slice_npu(const Tensor& self, IntArrayRef offsets, IntArrayRef size) {
  // calculate the output size
  SmallVector<int64_t, SIZE> outputSize = 
      CalcuOpUtil::ConvertIntArrayRefToSmallVector(size);
  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  // calculate the output result of the NPU
  slice_out_npu(result, self, offsets, size);

  return result;
}

} // namespace native
} // namespace at