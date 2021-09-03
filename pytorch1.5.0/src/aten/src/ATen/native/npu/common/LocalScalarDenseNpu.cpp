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
#include <c10/npu/NPUStream.h>
#include <third_party/acl/inc/acl/acl_base.h>
#include <third_party/acl/inc/acl/acl_rt.h>

namespace at {
namespace native {

Scalar _local_scalar_dense_npu(const Tensor& self) {
  Scalar r;
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      self.scalar_type(),
      "_local_scalar_dense_npu",
      [&] {
        scalar_t value = 0;
        c10::npu::NPUStream copy_stream = c10::npu::getCurrentNPUStream();
        aclError error = aclrtMemcpyAsync(
            &value,
            sizeof(scalar_t),
            self.data_ptr<scalar_t>(),
            sizeof(scalar_t),
            ACL_MEMCPY_DEVICE_TO_HOST,
            copy_stream);
        if (error != ACL_ERROR_NONE) {
          AT_ERROR("aclrtMemcpy device to host error.");
          return;
        }

        error = aclrtSynchronizeStream(copy_stream);
        if (error != ACL_ERROR_NONE) {
          AT_ERROR("ACL stream synchronize failed.");
          return;
        }
        r = Scalar(value);
      });
  return r;
}
} // namespace native
} // namespace at
