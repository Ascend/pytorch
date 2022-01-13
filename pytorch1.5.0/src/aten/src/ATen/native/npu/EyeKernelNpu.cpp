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

Tensor& eye_out_npu_nocheck(Tensor& result, int64_t n, int64_t m){
  OpCommand cmd;
  cmd.Name("Eye")
    .Output(result)      
    .Attr("num_rows", n)
    .Attr("num_columns", m)
    .Run();
    
  return result;
}

Tensor& eye_out_npu(Tensor& result, int64_t n) {
  return eye_out_npu(result, n, -1);
}

Tensor& eye_out_npu(Tensor& result, int64_t n, int64_t m) {
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);

  if (m < 0) {
    m = n;
  }

  result.resize_({n, m});
  eye_out_npu_nocheck(result, n, m);  
  return result;
}

Tensor eye_npu(int64_t n, const TensorOptions& options) {
  // get the output size
  SmallVector<int64_t, N> outputSize = {n, n};

  // The operator does not support the bool type and needs to be converted to an integer.
  Tensor result = (options.dtype() == at::kBool) 
      ? OpPreparation::ApplyTensorWithFormat(outputSize, options.dtype(at::ScalarType::Int), ACL_FORMAT_ND) 
      : OpPreparation::ApplyTensorWithFormat(outputSize, options, ACL_FORMAT_ND);

  eye_out_npu(result, n);
  
  if(options.dtype() == at::kBool){
    result = result.to(at::kBool); 
  }

  return result;
}

Tensor eye_npu(int64_t n, int64_t m, const TensorOptions& options) {
  // get the output size
  SmallVector<int64_t, N> outputSize = {n, m};

  // The operator does not support the bool type and needs to be converted to an integer.
  Tensor result = (options.dtype() == at::kBool) 
      ? OpPreparation::ApplyTensorWithFormat(outputSize, options.dtype(at::ScalarType::Int), ACL_FORMAT_ND) 
      : OpPreparation::ApplyTensorWithFormat(outputSize, options, ACL_FORMAT_ND);

  eye_out_npu_nocheck(result, n, m);
  
  if(options.dtype() == at::kBool){
    result = result.to(at::kBool); 
  }

  return result;
}

} // namespace native
} // namespace at