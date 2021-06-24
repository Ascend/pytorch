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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUTensorDesc, N> embedding_renorm_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> embedding_renorm_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> embedding_renorm_npu_attr(
      double max_norm, 
      double norm_type){
  int64_t dim = 0;
  float max_norm_float = (float) max_norm;
  float norm_type_float = (float) norm_type;
  NPUAttrDesc npuAttrScalarP = NPUAttrDesc("p", norm_type_float);
  NPUAttrDesc npuAttrScalarMaxnorm = NPUAttrDesc("maxnorm", max_norm_float);
  NPUAttrDesc npuAttrDim = NPUAttrDesc("dim", dim);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrScalarP, npuAttrDim, npuAttrScalarMaxnorm};
  return attrs;
}
SmallVector<NPUAttrDesc, N> embedding_gather2d_npu_attr() {
  NPUAttrDesc npuAttrAxis = NPUAttrDesc("axis", (int64_t)0);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrAxis};
  return attrs;
}

SmallVector<NPUAttrDesc, N> embedding_renorm_scatter_update_npu_attr(){
  NPUAttrDesc npuAttrAxis = NPUAttrDesc("use_locking", false);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrAxis};
  return attrs;
}

Tensor& embedding_renorm_gather2d_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& indices
    ){
// execute the NPU operate  GatherV2D
  auto inputs = embedding_renorm_npu_input({self, indices});
  auto outputs = embedding_renorm_npu_output({result});
  auto attrs = embedding_gather2d_npu_attr();
  CalcuOpUtil::execute_npu_operate("GatherV2D", inputs, outputs, attrs);
  return result;
}

Tensor& embedding_renorm_execute_out_npu(
    Tensor& result,
    const Tensor& self,
    double max_norm, 
    double norm_type){
//execute the NPU operate  Renorm
  auto inputs = embedding_renorm_npu_input({self});
  auto outputs = embedding_renorm_npu_output({result});
  auto attrs = embedding_renorm_npu_attr(max_norm, norm_type);
  CalcuOpUtil::execute_npu_operate("Renorm", inputs, outputs, attrs);
  return result;
}


Tensor& embedding_renorm_scatter_update_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& indices,
    const Tensor& update){
  auto inputs = embedding_renorm_npu_input({self, indices, update});
  auto outputs = embedding_renorm_npu_output({result});
  auto attrs = embedding_renorm_scatter_update_npu_attr();
  CalcuOpUtil::execute_npu_operate("ScatterUpdate", inputs, outputs, attrs);
  return result;
}


Tensor& embedding_renorm_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& indices,
    Tensor& mid_input,
    Tensor& mid_output,
    double max_norm, 
    double norm_type){
// execute the NPU operate  GatherV2D,generate  new tensor by indices 
  embedding_renorm_gather2d_out_npu(
        mid_input,
        self,
        indices);
//execute the NPU operate  Renorm
  embedding_renorm_execute_out_npu(
        mid_output,
        mid_input,
        max_norm, 
        norm_type);
// executing the NPU operator ScatterUpdate
  embedding_renorm_scatter_update_out_npu(
        result,
        self,
        indices,
        mid_output); 
  return result;
}

Tensor& embedding_renorm_npu_(
    Tensor& self,
    const Tensor& indices,
    double max_norm, 
    double norm_type) {

//check dim and type
  auto self_arg = TensorArg(self, "self", 1);
  auto indices_arg = TensorArg(indices, "indices", 2);
  checkDim("embedding_renorm_", self_arg, 2);
  checkScalarType("embedding_renorm_", indices_arg, kLong);

// indices must be int64 in pytorch, but npu can only support int32
  auto indices_int32 = indices.to("cpu");
  indices_int32 = indices_int32.to(at::kInt);
  indices_int32 = indices_int32.to("npu");   

//resize indices to 1D
  Tensor indices_copy = indices.clone();
  auto num_indices = indices.numel();
  resize_npu_(indices_copy, num_indices);
    
  SmallVector<Tensor, N> inputs = {self};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

//get the  outSize of  GatherV2 , the middle tensor
  auto midSize = embedding_renorm_mid_npu_output_size(self, indices_copy);
  Tensor mid = at::empty_with_format(midSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  Tensor mid1 = at::empty_with_format(midSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
    
//inplace operate
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = embedding_renorm_out_npu(contiguousSelf, contiguousSelf, indices_copy, mid, mid1, max_norm, norm_type);
  NpuUtils::format_fresh_view(self, result);
  } else {
    embedding_renorm_out_npu(self, self, indices_copy, mid, mid1, max_norm, norm_type);
  }
  return self;
}

}
}