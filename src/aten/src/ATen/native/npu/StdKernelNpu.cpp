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

namespace at { 
namespace native {
using namespace at::native::npu;

Tensor& std_out_npu(
    Tensor& result, 
    const Tensor& self, 
    DimnameList dim, 
    bool unbiased, 
    bool keepdim) {
  return std_out_npu(result, self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor& std_out_npu(
    Tensor& result, 
    const Tensor& self, 
    IntArrayRef dim, 
    bool unbiased, 
    bool keepdim) {
  auto outputSize = std_npu_output_size(self, dim, keepdim);
  Tensor meanResult = OpPreparation::ApplyTensor(self, std::get<1>(outputSize));

  // executing the NPU operator
  if (!c10::npu::OptionsManager::CheckDynamicEnable()) {
    OpCommand cmd;
    cmd.Name("ReduceStd")
        .Input(self)
        .Output(result)
        .Output(meanResult)
        .Attr("dim", dim)
        .Attr("unbiased", unbiased)
        .Attr("keepdim", keepdim)
        .Run();
  } else {
    OpCommand cmd1;
    cmd1.Name("ReduceMeanD")
        .Input(self)
        .Output(meanResult)
        .Attr("axes", dim)
        .Attr("keep_dims", keepdim)
        .Run();
    if (meanResult.dim() != 0 && keepdim == false) {
      meanResult = meanResult.unsqueeze(dim[0]);
    }
    Tensor meanResult2 = meanResult.expand(self.sizes());
    OpCommand cmd2;
    cmd2.Name("ReduceStdWithMean")
        .Input(self)
        .Input(meanResult2)
        .Output(result)
        .Attr("dim", dim)
        .Attr("unbiased", unbiased)
        .Attr("keepdim", keepdim)
        .Run();
  }

  return result;
}

tuple<Tensor&, Tensor&> std_mean_out_npu(
    Tensor& result1, 
    Tensor& result2, 
    const Tensor& self, 
    IntArrayRef dim, 
    bool unbiased, 
    bool keepdim) {
  // executing the NPU operator 
  if (!c10::npu::OptionsManager::CheckDynamicEnable()) { 
    OpCommand cmd;
    cmd.Name("ReduceStd")
        .Input(self)
        .Output(result1)
        .Output(result2)
        .Attr("dim", dim)
        .Attr("unbiased", unbiased)
        .Attr("keepdim", keepdim)
        .Run();
  } else {
    OpCommand cmd1;
    cmd1.Name("ReduceMeanD")
        .Input(self)
        .Output(result2)
        .Attr("axes", dim)
        .Attr("keep_dims", keepdim)
        .Run();
    Tensor result2_copy = result2;
    if (result2.dim() != 0 && keepdim == false) {
      result2_copy = result2.unsqueeze(dim[0]);
    }
    result2_copy = result2_copy.expand(self.sizes());
    OpCommand cmd2;
    cmd2.Name("ReduceStdWithMean")
        .Input(self)
        .Input(result2_copy)
        .Output(result1)
        .Attr("dim", dim)
        .Attr("unbiased", unbiased)
        .Attr("keepdim", keepdim)
        .Run();
  }

  return std::tie(result1, result2);
}

Tensor std_dim_npu(
    const Tensor & self, 
    IntArrayRef dim, 
    bool unbiased, 
    bool keepdim) {
  // calculate the output size
  auto outputSize = std_npu_output_size(self, dim, keepdim);

  // construct the output tensor of the NPU
  Tensor result1 = OpPreparation::ApplyTensor(self, std::get<0>(outputSize));
  Tensor result2 = OpPreparation::ApplyTensor(self, std::get<1>(outputSize));

  // calculate the output result of the NPU
  std_mean_out_npu(result1, result2, self, dim, unbiased, keepdim);
  return result1;
}

Tensor std_npu(
    const Tensor & self, 
    bool unbiased) {
  SmallVector<int64_t, SIZE> dims = CalcuOpUtil::get_dimlist_for_tensor(self);
  return std_dim_npu(self, dims, unbiased, false);
}

tuple <Tensor, Tensor> std_mean_npu(
    const Tensor & self, 
    bool unbiased) {
  SmallVector<int64_t, SIZE> dims = CalcuOpUtil::get_dimlist_for_tensor(self);
  return std_mean_dim_npu(self, dims, unbiased, false);
}

tuple <Tensor, Tensor> std_mean_dim_npu(
    const Tensor & self, 
    IntArrayRef dim, 
    bool unbiased, 
    bool keepdim) {
  // calculate the output size
  auto outputSize = std_npu_output_size(self, dim, keepdim);

  // construct the output tensor of the NPU
  Tensor result1 = OpPreparation::ApplyTensor(self, std::get<0>(outputSize));
  Tensor result2 = OpPreparation::ApplyTensor(self, std::get<1>(outputSize));

  // calculate the output result of the NPU
  std_mean_out_npu(result1, result2, self, dim, unbiased, keepdim);
  return std::tie(result1, result2);
}

tuple <Tensor, Tensor> std_mean_names_npu(
    const Tensor & self, 
    DimnameList dim, 
    bool unbiased, 
    bool keepdim) {
  return std_mean_dim_npu(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor std_names_npu(
    const Tensor & self, 
    DimnameList dim, 
    bool unbiased, 
    bool keepdim) {
  return std_dim_npu(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

} // namespace native
} // namespace at::native
