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

#include <ATen/detail/NPUHooksInterface.h>

#include <ATen/Generator.h>
#include <THNPU/THNPUCachingHostAllocator.h>
#include <c10/util/Optional.h>

namespace at {
namespace npu {
namespace detail {

// The real implementation of NPUHooksInterface
struct NPUHooks : public at::NPUHooksInterface {
  NPUHooks(at::NPUHooksArgs) {}
  void initNPU() const override;
  bool isPinnedPtr(void* data) const override;
  Generator* getDefaultNPUGenerator(DeviceIndex device_index = -1) const override;
  bool hasNPU() const override;
  int64_t current_device() const override;
  Allocator* getPinnedMemoryAllocator() const override;
  int getNumNPUs() const override;
};

} // namespace detail
} // namespace npu
} // namespace at
