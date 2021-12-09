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
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

tuple<Tensor&, Tensor&> min_v1_out_npu_nocheck(
    Tensor& output,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  OpCommand cmd;
  cmd.Name("ArgMinWithValue")
    .Input(self)
    .Output(indices)      
    .Output(output)
    .Attr("dimension", dim)
    .Attr("keep_dims", keepdim)
    .Run();
  
  return std::tie(output, indices);
}

tuple<Tensor, Tensor> min_v1_npu(const Tensor& self, int64_t dim, bool keepdim) {
  SmallVector<int64_t, SIZE> dims = {dim};
  SmallVector<int64_t, SIZE> outputSize =
      reduce_ops_npu_output_size(self, dims, keepdim);
  SmallVector<int64_t, SIZE> indicesSize =
      reduce_ops_npu_output_size(self, dims, keepdim);
  
  int64_t npuFormat = CalcuOpUtil::get_tensor_npu_format(self);
  if (outputSize.empty()) {
    npuFormat = ACL_FORMAT_NCHW;
  }
  
  Tensor outputs = OpPreparation::ApplyTensorWithFormat(outputSize, self.options(), npuFormat);
  Tensor indices = OpPreparation::ApplyTensorWithFormat(indicesSize, self.options().dtype(kInt), npuFormat);
      
  min_v1_out_npu_nocheck(outputs, indices, self, dim, keepdim);
  return std::tie(outputs, indices);
}

tuple<Tensor, Tensor> min_v1_npu(const Tensor& self, Dimname dim, bool keepdim) {
  return min_v1_npu(self, dimname_to_position(self, dim), keepdim);
}

} // namespace native
} // namespace at
