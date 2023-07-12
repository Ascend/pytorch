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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::replication_pad1d_out(const at::Tensor& self, at::IntArrayRef padding,
                                                           at::Tensor& out) {
  DO_COMPATIBILITY(aclnnReplicationPad1d,
                   NPUNativeFunctions::replication_pad1d_out(self, padding, out));

  auto output_size = replication_pad1d_npu_out_size(self, padding);
  OpPreparation::CheckOut({self}, out, self, output_size);
  EXEC_NPU_CMD(aclnnReplicationPad1d, self, padding, out);
  return out;
}

at::Tensor NPUNativeOpApiFunctions::replication_pad1d(const at::Tensor& self, at::IntArrayRef padding) {

  DO_COMPATIBILITY(aclnnReplicationPad1d,
                   NPUNativeFunctions::replication_pad1d(self, padding));

  auto output_size = replication_pad1d_npu_out_size(self, padding);
  at::Tensor out = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  EXEC_NPU_CMD(aclnnReplicationPad1d, self, padding, out);
  return out;
}

}  // namespace native
}  // namespace at_npu
