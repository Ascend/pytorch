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
  auto num_indices = indices.numel();
  auto indices_1d = indices.clone();  
  resize_npu_(indices_1d, {num_indices});

  // get the  outSize of  GatherV2 , the middle tensor
  SmallVector<int64_t, SIZE> midSize = {num_indices, self.size(1)};

  auto mid_input = OpPreparation::ApplyTensor(self, midSize);
  auto mid_output = OpPreparation::ApplyTensor(self, midSize);

  // execute the NPU operate  GatherV2D, generate  new tensor by indices
  embedding_renorm_gather2d_out_npu(mid_input, self, indices_1d);

  // execute the NPU operate  Renorm
  embedding_renorm_execute_out_npu(mid_output, mid_input, max_norm, norm_type);

  // execute the NPU operate  ZerosLike or RangeD, generate new tensor by indices.numel()  
  Tensor input_indices;
  if (num_indices == 1) {
    input_indices = npu_dtype_cast(at::zeros({1}, self.options()), at::kLong);
  } else {
    input_indices = npu_dtype_cast(at::range(0, num_indices - 1, self.options()), at::kLong);
    resize_npu_(mid_output, {mid_output.numel(), 1}); 
  }

  // execute the NPU operate MUL, generate change result
  auto scalar_out = OpPreparation::ApplyTensor(self, {num_indices, 1});
  embedding_renorm_gather2d_out_npu(scalar_out, mid_output, input_indices);

  auto out_res = mid_input * scalar_out;

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

  OpPipeWithDefinedOut pipe;
  pipe.CheckMemory({self, indices}, {self})
   .Func([&self, &indices, max_norm, norm_type](Tensor& result){
        embedding_renorm_out_npu(self, self, indices, max_norm, norm_type);})
   .Call(self);

  return self;
}

}
}