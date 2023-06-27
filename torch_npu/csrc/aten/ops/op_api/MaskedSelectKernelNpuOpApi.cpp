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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

static constexpr int OK = 0;

static inline std::tuple<at::Tensor, at::Tensor> expand_outplace_npu(const at::Tensor& to_expand1,
                                                                     const at::Tensor& to_expand2) {
  if (to_expand1.sizes().equals(to_expand2.sizes())) {
    return std::make_tuple(to_expand1, to_expand2);
  }

  auto expanded_size = at::infer_size(to_expand1.sizes(), to_expand2.sizes());
  return std::make_tuple(to_expand1.expand(expanded_size, true), to_expand2.expand(expanded_size, true));
}

static inline at::SmallVector<int64_t, SIZE> masked_select_npu_output_size(const at::Tensor& self,
                                                                           const at::Tensor& mask) {
  at::Tensor maskCast;
  at::Tensor selfCast;
  std::tie(maskCast, selfCast) = expand_outplace_npu(mask, self);
  auto outputSize = {maskCast.numel()};
  return outputSize;
}

at::Tensor exec_aclnn_masked_select(const at::Tensor& self, const at::Tensor& mask, at::Tensor& out) {
  static auto opApiFuncAddr = GetOpApiFuncAddr("aclGetViewShape");
  using aclGetViewShapeFunc = int (*)(const aclTensor* tensor, int64_t** view_dims, uint64_t* view_dims_num);
  auto aclGetViewShape = reinterpret_cast<aclGetViewShapeFunc>(opApiFuncAddr);
  auto npuAclParams = EXEC_NPU_CMD_SYNC(aclnnMaskedSelect, self, mask, out);
  int64_t* view_dims = nullptr;
  uint64_t view_dim_num = 0;
  auto ret = aclGetViewShape(npuAclParams.Get<2>(), &view_dims, &view_dim_num);
  TORCH_CHECK(ret == OK, "aclGetViewShape failed");
  at::SmallVector<int64_t, SIZE> outputShapeSize = {};
  for (uint64_t i = 0; i < view_dim_num; i++) {
    outputShapeSize.push_back(view_dims[i]);
  }
  out.resize_(outputShapeSize);
  delete view_dims;
  view_dims = nullptr;
  return out;
}

at::Tensor NPUNativeOpApiFunctions::masked_select(const at::Tensor& self, const at::Tensor& mask) {
  DO_COMPATIBILITY(aclnnMaskedSelect, NPUNativeFunctions::masked_select(self, mask));

  auto outputSize = masked_select_npu_output_size(self, mask);
  at::Tensor out = OpPreparation::ApplyTensor(self, outputSize);
  return exec_aclnn_masked_select(self, mask, out);
}

at::Tensor& NPUNativeOpApiFunctions::masked_select_out(const at::Tensor& self, const at::Tensor& mask,
                                                       at::Tensor& result) {
  DO_COMPATIBILITY(aclnnMaskedSelect, NPUNativeFunctions::masked_select_out(self, mask, result));

  auto outputSize = masked_select_npu_output_size(self, mask);
  OpPreparation::CheckOut({self, mask}, result, self.scalar_type(), outputSize);

  auto out = result.is_contiguous() ? result : result.contiguous();

  auto output = exec_aclnn_masked_select(self, mask, out);
  if (!result.is_contiguous()) {
    result.copy_(output);
  }
  return result;
}
}  // namespace native
}  // namespace at_npu