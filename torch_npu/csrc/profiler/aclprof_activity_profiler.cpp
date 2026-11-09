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

#include "torch_npu/csrc/profiler/aclprof_activity_profiler.h"

#include "libkineto.h"
#include <c10/util/Exception.h>

namespace torch_npu {
namespace profiler {

AclprofActivityProfiler::AclprofActivityProfiler(std::set<libkineto::ActivityType>) {}

void AclprofActivityProfiler::start() {
  status_ = libkineto::TraceStatus::RECORDING;
}

void AclprofActivityProfiler::stop() {
  status_ = libkineto::TraceStatus::PROCESSING;
}

std::vector<std::string> AclprofActivityProfiler::errors() {
  return {};
}

void AclprofActivityProfiler::processTrace(libkineto::ActivityLogger&) {
  TORCH_WARN("ACLprof activity collection is not implemented yet");
}

void AclprofActivityProfiler::processTrace(
    libkineto::ActivityLogger& logger,
    libkineto::getLinkedActivityCallback,
    int64_t,
    int64_t) {
  processTrace(logger);
}

std::unique_ptr<libkineto::DeviceInfo> AclprofActivityProfiler::getDeviceInfo() {
  return nullptr;
}

std::vector<libkineto::ResourceInfo> AclprofActivityProfiler::getResourceInfos() {
  return {};
}

std::unique_ptr<libkineto::CpuTraceBuffer> AclprofActivityProfiler::getTraceBuffer() {
  return nullptr;
}

} // namespace profiler
} // namespace torch_npu
