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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& embedding_renorm_gather2d_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& indices) {
  OpCommand cmd;
  cmd.Name("GatherV2D")
    .Input(self)
    .Input(indices)
    .Output(result)
    .Attr("axis", (int64_t)0)
    .Run();
  return result;
}

Tensor& embedding_renorm_execute_out_npu(
    Tensor& result,
    const Tensor& self,
    double max_norm,
    double norm_type) {
  OpCommand cmd;
  cmd.Name("Renorm")
    .Input(self)
    .Output(result)
    .Attr("p", (float)norm_type)
    .Attr("dim", (int64_t)0)
    .Attr("maxnorm", (float)max_norm)
    .Run();
  return result;
}


Tensor& embedding_renorm_scatter_update_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& indices,
    const Tensor& update) {
  OpCommand cmd;
  cmd.Name("ScatterUpdate")
    .Input(self)
    .Input(indices)
    .Input(update)
    .Output(result)
    .Attr("use_locking", false)
    .Run();
  return result;
}

Tensor& embedding_renorm_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& indices,
    double max_norm,
    double norm_type){

  // get the  outSize of  GatherV2 , the middle tensor
  SmallVector<int64_t, SIZE> midSize = {indices.size(0), self.size(1)};
  Tensor mid_input = OpPreparation::ApplyTensor(self, midSize);
  Tensor mid_output = OpPreparation::ApplyTensor(self, midSize);

  // execute the NPU operate  GatherV2D, generate  new tensor by indices
  embedding_renorm_gather2d_out_npu(mid_input,self,indices);

  // execute the NPU operate  Renorm
  embedding_renorm_execute_out_npu(mid_output, mid_input, max_norm, norm_type);

  // execute the NPU operate  ZerosLike or RangeD, generate new tensor by indices.numel()
  Tensor mid_output_copy = mid_output.clone();
  auto num_indices = indices.numel();
  Tensor input_indices;
  
  // RangeD not support range(0,0)
  if (num_indices - 1 == 0) {
    input_indices = at::zeros({1}, self.options()).to(at::kLong);
  } else {
    input_indices = at::range(0, num_indices-1, self.options()).to(at::kLong);
  }

  // execute the NPU operate  MUL, generate change result
  auto num_mid_output = mid_output.numel();
  resize_npu_(mid_output_copy, num_mid_output);
  Tensor scalar_out = OpPreparation::ApplyTensor(self, {num_indices, 1});
  embedding_renorm_gather2d_out_npu(scalar_out, mid_output_copy, input_indices);
  Tensor out_res = mid_input * scalar_out;

  // executing the NPU operator ScatterUpdate
  embedding_renorm_scatter_update_out_npu(result, self, indices, out_res);

  return result;
}

Tensor& embedding_renorm_npu_(
    Tensor& self,
    const Tensor& indices,
    double max_norm,
    double norm_type) {

  // check dim and type
  auto self_arg = TensorArg(self, "self", 1);
  auto indices_arg = TensorArg(indices, "indices", 2);
  checkDim("embedding_renorm_", self_arg, 2);
  checkScalarType("embedding_renorm_", indices_arg, kLong);

  // resize indices to 1D
  Tensor indices_copy = indices.clone();
  auto num_indices = indices.numel();
  resize_npu_(indices_copy, num_indices);

  OpPipeWithDefinedOut pipe;
  pipe.CheckMemory({self, indices_copy}, {self})
   .Func([&self, &indices_copy, max_norm, norm_type](Tensor& result){
        embedding_renorm_out_npu(self, self, indices_copy, max_norm, norm_type);})
   .Call(self);

  return self;
}

}
}