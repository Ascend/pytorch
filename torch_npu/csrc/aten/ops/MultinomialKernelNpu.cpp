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

at::Tensor& multinomial_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t num_samples, 
    bool replacement,
    c10::optional<at::Generator> gen) {
  
  OpCommand cmd;
  cmd.Name("MultinomialWithReplacementD")
    .Input(self)
    .Output(result)
    .Attr("num_samples", num_samples)
    .Attr("replacement", replacement)
    .Run();

  return result;
}


at::Tensor& NPUNativeFunctions::multinomial_out(
    const at::Tensor& self,
    int64_t num_samples, 
    bool replacement,
    c10::optional<at::Generator> gen,
    at::Tensor& result){

  auto input_dim = self.dim();
  TORCH_CHECK(input_dim==1 || input_dim==2, "dim of input tensor only can be 1 or 2.");

  auto output_dim = result.dim();
  TORCH_CHECK(input_dim==output_dim, "dim of output tensor must equal to input tensor.");

  auto num = result.size(output_dim-1);
  TORCH_CHECK(num == num_samples, "column of output tensor must equal num_samples.");

  multinomial_out_npu_nocheck(result, self, num_samples, replacement, gen);

  return result;
}

at::Tensor NPUNativeFunctions::multinomial(
    const at::Tensor& self, 
    int64_t num_samples, 
    bool replacement, 
    c10::optional<at::Generator> gen){
  
  auto dim = self.dim();
  TORCH_CHECK(dim==1 || dim==2, "dim of input tensor only can be 1 or 2.");

  auto shape = array_to_small_vector(self.sizes());
  shape[dim-1] = num_samples;

  at::Tensor result = OpPreparation::ApplyTensorWithSizes(
      shape, self.options().dtype(at::kLong));
  multinomial_out_npu_nocheck(result, self, num_samples, replacement, gen);
  return result;
}
} // namespace native
} // namespace at_npu
