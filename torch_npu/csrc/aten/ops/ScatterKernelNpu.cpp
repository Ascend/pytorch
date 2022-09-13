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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/XLANativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& scatter_npu_src_nocheck(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src) {
  OpCommand cmd;
  cmd.Name("ScatterElements")
     .Input(self)
     .Input(index)
     .Input(src)
     .Output(self)
     .Attr("axis", dim)
     .Run();
  return self;
}

at::Tensor& scatter_npu_src_impl(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index_ex,
    const at::Tensor& src_ex) {
  at::ScalarType selfType = self.scalar_type();
  if (selfType == at::ScalarType::Half) {
    self = XLANativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }

  at::Tensor index(index_ex);
  if (index.scalar_type() == at::ScalarType::Half) {
    index = XLANativeFunctions::npu_dtype_cast(index, at::ScalarType::Float);
  }

  at::Tensor src(src_ex);
  if (src.scalar_type() != self.scalar_type()) {
    src = XLANativeFunctions::npu_dtype_cast(src, self.scalar_type());
  }

  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);

    scatter_npu_src_nocheck(contiguousSelf, dim, index, src);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    scatter_npu_src_nocheck(self, dim, index, src);
  }

  if(self.scalar_type() != selfType){
    self = XLANativeFunctions::npu_dtype_cast(self, at::ScalarType::Half);
  }
  return self;
}

at::Tensor& XLANativeFunctions::scatter_(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index_ex,
    const at::Tensor& src_ex) {
  at::Tensor src(src_ex);
  scatter_npu_src_impl(self, dim, index_ex, src);
  return self;
}

at::Tensor& XLANativeFunctions::scatter_(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index_ex,
    const at::Scalar& src) {
  at::Tensor srcTensor = scalar_to_tensor(src).to(at::ScalarType::Float);
  srcTensor = CalcuOpUtil::copy_tensor_host_to_device(srcTensor);
  at::Tensor srcTensor_broadcast = XLANativeFunctions::npu_broadcast(srcTensor, array_to_small_vector(index_ex.sizes()));
  scatter_npu_src_impl(self, dim, index_ex, srcTensor_broadcast);
  return self;
}
} // namespace native
} // namespace at_npu
