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

#pragma once

// This header provides C++ wrappers around commonly used CUDA API functions.
// The benefit of using C++ here is that we can raise an exception in the
// event of an error, rather than explicitly pass around error codes.  This
// leads to more natural APIs.
//
// The naming convention used here matches the naming convention of torch.cuda

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include <c10/npu/NPUException.h>
#include <c10/npu/npu_log.h>
#include <third_party/acl/inc/acl/acl.h>

namespace c10 {
namespace npu {
inline DeviceIndex device_count() noexcept {
  unsigned int count = 1;
  // NB: In the past, we were inconsistent about whether or not this reported
  // an error if there were driver problems are not.  Based on experience
  // interacting with users, it seems that people basically ~never want this
  // function to fail; it should just return zero if things are not working.
  // Oblige them.
  aclError error = aclrtGetDeviceCount(&count);
  if (error != ACL_ERROR_NONE) {
    // Clear out the error state, so we don't spuriously trigger someone else.
    // (This shouldn't really matter, since we won't be running very much CUDA
    // code in this regime.)
    // npuError_t last_err = npuGetLastError();
    //(void)last_err;
    NPU_LOGE("get device count of NPU failed");
    return 0;
  }
  return static_cast<DeviceIndex>(count);
}

inline DeviceIndex current_device() {
  int cur_device = 0;
  C10_NPU_CHECK(aclrtGetDevice(&cur_device));
  return static_cast<DeviceIndex>(cur_device);
}

inline void set_device(DeviceIndex device) {
  // C10_NPU_CHECK(npuSetDevice(static_cast<int>(device)));
}

} // namespace npu
} // namespace c10
