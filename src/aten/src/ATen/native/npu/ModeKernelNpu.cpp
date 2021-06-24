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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<int64_t, SIZE> mode_npu_output_size(
  const Tensor& self,
  int64_t dim, 
  bool keepdim) {
  SmallVector<int64_t, SIZE> outputSize;
  if(dim==0){
    outputSize={self.size(1)};
  };
  if(dim==-1 || dim==1){
    outputSize={self.size(0)};
  };
  return outputSize;
}

SmallVector<NPUTensorDesc, N> mode_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> mode_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  auto outputs = CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);

  string indicesRealType = "int64";
  outputs[outputs.size() - 1].realDataType = indicesRealType;
  return outputs;
}

SmallVector<NPUAttrDesc, N> mode_npu_attr(
     int64_t dim, bool keepdim) {
  NPUAttrDesc npuAttrDim = NPUAttrDesc("dim", dim);
  NPUAttrDesc npuAttrKeepdim = NPUAttrDesc("keepdim", keepdim);

  SmallVector<NPUAttrDesc, N> attrs = {npuAttrDim,
                                       npuAttrKeepdim};
  return attrs;
}


tuple<Tensor&, Tensor&> mode_out_npu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim, 
    bool keepdim) {
  // constructs the input and output NPUTensorDesc
  auto inputs = mode_npu_input({self});
  auto outputs = mode_npu_output({values, indices});

  // constructs the attr of the NPUAttrDesc
  auto attrs = mode_npu_attr(dim,keepdim);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate(
      "Mode", inputs, outputs, attrs);
  
  return tuple<Tensor&, Tensor&>(values, indices);
}

tuple<Tensor&, Tensor&> _mode_out_npu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim, 
    bool keepdim) {
    
  return mode_out_npu(values,indices,self, dim, keepdim);
}

tuple<Tensor, Tensor> mode_npu(
    const Tensor& self,
    int64_t dim, 
    bool keepdim
) {
  // calculate the output size
  auto outputSize = mode_npu_output_size(self, dim,keepdim);

  // construct the output tensor of the NPU
  Tensor values= at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  Tensor indices = at::empty_with_format(
      outputSize, self.options().dtype(at::kLong), CalcuOpUtil::get_tensor_npu_format(self));


  // calculate the output result of the NPU
  mode_out_npu(
      values, indices, self, dim,keepdim);
  return tuple<Tensor, Tensor>(values, indices);

}

tuple<Tensor, Tensor> _mode_npu(
    const Tensor& self,
    int64_t dim, 
    bool keepdim
) {
  return mode_npu(self,dim, keepdim);
  }

} // namespace native
} // namespace at