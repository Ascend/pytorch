// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include <set>

#include "torch_npu/csrc/profiler/npu_activity_profiler.h"

namespace torch_npu {
namespace profiler {

class TORCH_NPU_API AclprofActivityProfiler : public NPUActivityProfiler {
 public:
  explicit AclprofActivityProfiler(std::set<libkineto::ActivityType> activities);

  const char* name() const override {
    return "aclprof";
  }
  void start() override;
  void stop() override;
  std::vector<std::string> errors() override;
  void processTrace(libkineto::ActivityLogger& logger) override;
  void processTrace(
      libkineto::ActivityLogger& logger,
      libkineto::getLinkedActivityCallback getLinkedActivity,
      int64_t startTime,
      int64_t endTime) override;
  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override;
  std::vector<libkineto::ResourceInfo> getResourceInfos() override;
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override;
};

} // namespace profiler
} // namespace torch_npu
