// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace{
using namespace at_npu::native;
inline at::Tensor &norm_out_npu_nocheck_opapi(at::Tensor &out,
                                              const at::Tensor &self,
                                              c10::optional<at::Scalar> p,
                                              at::IntArrayRef dim,
                                              bool keepdim) {
  at::Scalar pvalue = 2;
  if (p.has_value()) {
    pvalue = p.value();
  }
  EXEC_NPU_CMD(aclnnNorm, self, pvalue, dim, keepdim, out);
  return out;
}

inline at::Tensor &norm_out_imp(const at::Tensor &self,
                              const c10::optional<at::Scalar> &p,
                              at::IntArrayRef dim, bool keepdim,
                              at::ScalarType dtype, at::Tensor &out) {
  DO_COMPATIBILITY(aclnnNorm,
                    NPUNativeFunctions::norm_out(self, p, dim, keepdim, out));

  auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
  OpPreparation::CheckOut({self}, out, dtype, outputSize);

  return norm_out_npu_nocheck_opapi(out, self, p, dim, keepdim);
}

inline at::Tensor norm_imp(const at::Tensor &self,
                           const c10::optional<at::Scalar> &p,
                           at::IntArrayRef dim, bool keepdim,
                           at::ScalarType dtype) {
  DO_COMPATIBILITY(aclnnNorm, NPUNativeFunctions::norm(self, p, dim, keepdim));

  auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
  at::Tensor out = OpPreparation::ApplyTensorWithSizes(
      outputSize, self.options().dtype(dtype));

  return norm_out_npu_nocheck_opapi(out, self, p, dim, keepdim);
}
}

namespace at_npu {
namespace native {

// norm.dtype_out
at::Tensor &NPUNativeOpApiFunctions::norm_out(
    const at::Tensor &self, const c10::optional<at::Scalar> &p,
    at::IntArrayRef dim, bool keepdim, at::ScalarType dtype, at::Tensor &out) {
  return norm_out_imp(self, p, dim, keepdim, out.scalar_type(), out);
}

// norm.out
at::Tensor &NPUNativeOpApiFunctions::norm_out(
    const at::Tensor &self, const c10::optional<at::Scalar> &p,
    at::IntArrayRef dim, bool keepdim, at::Tensor &out) {
  return norm_out_imp(self, p, dim, keepdim, out.scalar_type(), out);
}

// norm.ScalarOpt_dim_dtype
at::Tensor NPUNativeOpApiFunctions::norm(const at::Tensor &self,
                                         const c10::optional<at::Scalar> &p,
                                         at::IntArrayRef dim, bool keepdim,
                                         at::ScalarType dtype) {
  return norm_imp(self, p, dim, keepdim, dtype);
}

// norm.ScalarOpt_dtype
at::Tensor NPUNativeOpApiFunctions::norm(const at::Tensor &self,
                                         const c10::optional<at::Scalar> &p,
                                         at::ScalarType dtype) {
  return norm_imp(self, p, {}, false, dtype);
}

// norm.Scalar
at::Tensor NPUNativeOpApiFunctions::norm(const at::Tensor &self,
                                         const at::Scalar &p) {
  return norm_imp(self, p, {}, false, self.scalar_type());
}

// norm.ScalarOpt_dim
at::Tensor NPUNativeOpApiFunctions::norm(const at::Tensor &self,
                                         const c10::optional<at::Scalar> &p,
                                         at::IntArrayRef dim, bool keepdim) {
  return norm_imp(self, p, dim, false, self.scalar_type());
}

} // namespace native
} // namespace at_npu

