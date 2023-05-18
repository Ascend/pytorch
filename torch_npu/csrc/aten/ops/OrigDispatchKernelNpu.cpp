// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::_trilinear(
    const at::Tensor& i1_,
    const at::Tensor& i2_,
    const at::Tensor& i3_,
    at::IntArrayRef expand1_,
    at::IntArrayRef expand2_,
    at::IntArrayRef expand3_,
    at::IntArrayRef sumdim_,
    int64_t unroll_dim) {
  return at::native::_trilinear(i1_, i2_, i3_, expand1_, expand2_, expand3_, sumdim_, unroll_dim);
}

at::Tensor NPUNativeFunctions::diag_embed(
    const at::Tensor& self,
    int64_t offset,
    int64_t dim1_,
    int64_t dim2_) {
   return at::native::diag_embed(self, offset, dim1_, dim2_);
}

at::Tensor NPUNativeFunctions::pixel_shuffle(
    const at::Tensor& self,
    int64_t upscale_factor) {
  return at::native::math_pixel_shuffle(self, upscale_factor);
}

at::Tensor NPUNativeFunctions::pixel_unshuffle(
    const at::Tensor& self,
    int64_t downscale_factor) {
  return at::native::math_pixel_unshuffle(self, downscale_factor);
}

at::Tensor NPUNativeFunctions::lift_fresh_copy(const at::Tensor& self) {
  return at::native::lift_fresh_copy(self);
}

at::Tensor NPUNativeFunctions::_reshape_alias(
    const at::Tensor& self,
    at::IntArrayRef size, 
    at::IntArrayRef stride) {
  return at::native::_reshape_alias(self, size, stride);
}

} // namespace native
} // namespace at_npu
