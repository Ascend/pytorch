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
#include <filesystem>
#include <nlohmann/json.hpp>
#include "torch_npu/csrc/profiler/profiler_mgr.h"
#include <c10/util/Exception.h>

namespace torch_npu {
namespace profiler {

AclprofActivityProfiler::AclprofActivityProfiler(std::set<libkineto::ActivityType>) {}

void AclprofActivityProfiler::reportFailure(const char* stage, const char* reason) noexcept {
  status_ = libkineto::TraceStatus::ERROR;
  try {
    errors_.emplace_back(std::string("ACLprof ") + stage + " failed: " + reason);
    TORCH_WARN(errors_.back());
  } catch (...) {
    // A warning handler must not interrupt the Kineto transition.
  }
}

void AclprofActivityProfiler::cleanup() noexcept {
  try {
    ProfilerMgr::GetInstance()->FinishAclprof(this, &ownsNative_);
  } catch (const std::exception& error) {
    reportFailure("cleanup", error.what());
  } catch (...) {
    reportFailure("cleanup", "unknown error");
  }
}

AclprofActivityProfiler::~AclprofActivityProfiler() {
  // Failed CANN teardown has no general retry guarantee. Leave it to explicit cleanup.
  if (ownsNative_ && status_ != libkineto::TraceStatus::ERROR) {
    cleanup();
  }
}

void AclprofActivityProfiler::configure(const libkineto::Config& config) {
  TORCH_CHECK(!ownsNative_, "ACLprof session already owns native resources");
  const auto json = nlohmann::json::parse(config.getCustomConfig());
  NpuTraceConfig trace{};
  trace.trace_level = json.value("profiler_level", std::string("Level0"));
  trace.metrics = json.value("aic_metrics", std::string("ACL_AICORE_NONE"));
  trace.npu_memory = json.value("profile_memory", false);
  trace.record_shapes = json.value("record_shapes", false);
  trace.l2_cache = json.value("l2_cache", false);
  trace.msprof_tx = json.value("mstx", false);
  TORCH_CHECK(!json.value("record_op_args", false), "ACLprof native capture does not collect operator arguments");
  trace.op_attr = json.value("op_attr", false);
  trace.mstx_domain_include = json.value("mstx_domain_include", std::vector<std::string>{});
  trace.mstx_domain_exclude = json.value("mstx_domain_exclude", std::vector<std::string>{});
  trace.host_sys = json.value("host_sys", std::vector<std::string>{});
  trace.sys_io = json.value("sys_io", false);
  trace.sys_interconnection = json.value("sys_interconnection", false);
  auto path = json.value("output_path", std::string{});
  try {
    if (path.empty()) {
      path = std::filesystem::current_path().string();
      TORCH_WARN("ACLprof output_path is missing; writing native capture to the current directory");
    }
    ProfilerMgr::GetInstance()->PrepareAclprof(this, path, config.requestTraceID(), trace, &ownsNative_);
  } catch (const std::exception& error) {
    reportFailure("prepare", error.what());
    if (ownsNative_) {
      cleanup();
    }
  }
}

void AclprofActivityProfiler::start() {
  if (status_ == libkineto::TraceStatus::ERROR) {
    return;
  }
  try {
    TORCH_CHECK(ownsNative_, "ACLprof session is not prepared");
    ProfilerMgr::GetInstance()->StartAclprof(this);
    status_ = libkineto::TraceStatus::RECORDING;
  } catch (const std::exception& error) {
    reportFailure("start", error.what());
    if (ownsNative_) {
      cleanup();
    }
  }
}

void AclprofActivityProfiler::stop() {
  if (status_ == libkineto::TraceStatus::ERROR) {
    return;
  }
  try {
    if (ownsNative_) {
      ProfilerMgr::GetInstance()->FinishAclprof(this, &ownsNative_, status_ == libkineto::TraceStatus::RECORDING);
    }
    status_ = libkineto::TraceStatus::PROCESSING;
  } catch (const std::exception& error) {
    reportFailure("stop", error.what());
  }
}

std::vector<std::string> AclprofActivityProfiler::errors() {
  return errors_;
}

void AclprofActivityProfiler::processTrace(libkineto::ActivityLogger&) {
  // Native device records are written by ACLprof; no Kineto events are materialized.
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
