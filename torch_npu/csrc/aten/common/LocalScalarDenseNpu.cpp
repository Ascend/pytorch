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

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

c10::Scalar NPUNativeFunctions::_local_scalar_dense(const at::Tensor& self) {
  c10::Scalar r;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "_local_scalar_dense_npu",
      [&] {
        scalar_t value = 0;
        c10_npu::NPUStream copy_stream = c10_npu::getCurrentNPUStream();
        // Synchronous copy after stream synchronization
        aclError error = c10_npu::acl::AclrtSynchronizeStreamWithTimeout(copy_stream);
        if (error != ACL_ERROR_NONE) {
          C10_NPU_SHOW_ERR_MSG();
          AT_ERROR("ACL stream synchronize failed.");
          return;
        }

        error = CalcuOpUtil::AclrtMemcpyWithModeSwitch(
            &value,
            sizeof(scalar_t),
            std::make_pair(
                self.storage().unsafeGetStorageImpl(), self.storage_offset() * self.itemsize()),
            sizeof(scalar_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (error != ACL_ERROR_NONE) {
          C10_NPU_SHOW_ERR_MSG();
          AT_ERROR("aclrtMemcpy device to host error.");
          return;
        }
        r = c10::Scalar(value);
      });
  return r;
}

} // namespace native
} // namespace at_npu
