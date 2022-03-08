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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& embedding_renorm_gather2d_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& indices) {
  OpCommand cmd;
  cmd.Name("GatherV2D")
    .Input(self)
    .Input(indices)
    .Output(result)
    .Attr("axis", (int64_t)0)
    .Run();
  return result;
}

at::Tensor& embedding_renorm_execute_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
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

at::Tensor& embedding_renorm_scatter_update_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& indices,
    const at::Tensor& update) {
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

at::Tensor& embedding_renorm_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& indices,
    double max_norm,
    double norm_type){

  at::SmallVector<int64_t, SIZE> midSize = {indices.size(0), self.size(1)};
  at::Tensor mid_input = OpPreparation::ApplyTensor(self, midSize);
  at::Tensor mid_output = OpPreparation::ApplyTensor(self, midSize);

  embedding_renorm_gather2d_out_npu(mid_input,self,indices);

  embedding_renorm_execute_out_npu(mid_output, mid_input, max_norm, norm_type);

  at::Tensor mid_output_copy = mid_output.clone();
  auto num_indices = indices.numel();
  at::Tensor input_indices;

  if (num_indices - 1 == 0) {
    input_indices = NPUNativeFunctions::npu_dtype_cast(at::zeros({1}, self.options()), at::kLong);
  } else {
    input_indices = NPUNativeFunctions::npu_dtype_cast(at::range(0, num_indices-1, self.options()), at::kLong);
  }

  auto num_mid_output = mid_output.numel();
  at::native::resize_(mid_output_copy, num_mid_output);
  at::Tensor scalar_out = OpPreparation::ApplyTensor(self, {num_indices, 1});
  embedding_renorm_gather2d_out_npu(scalar_out, mid_output_copy, input_indices);
  at::Tensor out_res = mid_input * scalar_out;

  embedding_renorm_scatter_update_out_npu(result, self, indices, out_res);

  return result;
}

at::Tensor& NPUNativeFunctions::embedding_renorm_(
    at::Tensor& self,
    const at::Tensor& indices,
    double max_norm,
    double norm_type) {

  auto self_arg = at::TensorArg(self, "self", 1);
  auto indices_arg = at::TensorArg(indices, "indices", 2);
  at::checkDim("embedding_renorm_", self_arg, 2);
  at::checkScalarType("embedding_renorm_", indices_arg, kLong);

  at::Tensor indices_copy = indices.clone();
  auto num_indices = indices.numel();
  at::native::resize_(indices_copy, num_indices);

  OpPipeWithDefinedOut pipe;
  pipe.CheckMemory({self, indices_copy}, {self})
   .Func([&self, &indices_copy, max_norm, norm_type](at::Tensor& result){
        embedding_renorm_out_npu(self, self, indices_copy, max_norm, norm_type);})
   .Call(self);

  return self;
}

}  // namespace native
}  // namespace at_npu