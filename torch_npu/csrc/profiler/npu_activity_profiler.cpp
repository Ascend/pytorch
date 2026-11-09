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

#include "torch_npu/csrc/profiler/npu_activity_profiler.h"

#include <set>
#include <utility>

#include <nlohmann/json.hpp>
#include <c10/util/Exception.h>

#include "libkineto.h"
#include "torch_npu/csrc/profiler/aclprof_activity_profiler.h"
#include "torch_npu/csrc/profiler/mspti_activity_profiler.h"
#include <torch/csrc/profiler/standalone/privateuse1_profiler.h>

namespace torch_npu {
namespace profiler {
namespace {

bool privateUse1Requested(const std::set<libkineto::ActivityType>& types) {
  using AT = libkineto::ActivityType;
  return types.count(AT::CONCURRENT_KERNEL) != 0 || types.count(AT::GPU_MEMCPY) != 0 ||
      types.count(AT::GPU_MEMSET) != 0 || types.count(AT::GPU_USER_ANNOTATION) != 0;
}

NPUProfilerConfig parseExperimentalConfig(const std::string& customConfig) {
  NPUProfilerConfig result;
  const auto config = nlohmann::json::parse(customConfig, nullptr, false);
  if (!config.is_object()) {
    return result;
  }
  const auto backend = config.find("backend");
  if (backend != config.end() && backend->is_string() && backend->get<std::string>() == "aclprof") {
    result.backend = "aclprof";
  }
  const auto level = config.find("profiler_level");
  if (level != config.end()) {
    TORCH_CHECK(
        level->is_string() &&
            (*level == "Level0" || *level == "Level1" || *level == "Level2" || *level == "Level_none"),
        "Invalid NPU profiler_level");
    result.profilerLevel = level->get<std::string>();
  }
  const auto exports = config.find("export_type");
  if (exports != config.end()) {
    TORCH_CHECK(exports->is_array() && !exports->empty(), "Invalid NPU export_type: expected a nonempty array");
    std::set<std::string> values;
    for (const auto& value : *exports) {
      TORCH_CHECK(value.is_string() && (value == "db" || value == "text"), "Invalid NPU export_type");
      values.insert(value.get<std::string>());
    }
    result.exportTypes.assign(values.begin(), values.end());
  }
  return result;
}

} // namespace

std::unique_ptr<NPUActivityProfiler> NPUActivityProfilerFactory::create(
    const std::set<libkineto::ActivityType>& activities,
    const libkineto::Config& config) {
  if (!privateUse1Requested(activities)) {
    return nullptr;
  }
  auto options = parseExperimentalConfig(config.getCustomConfig());
  std::unique_ptr<NPUActivityProfiler> profiler;
  if (options.backend == "aclprof") {
    profiler = std::make_unique<AclprofActivityProfiler>(activities);
  } else {
    profiler = std::make_unique<MsptiActivityProfiler>(activities);
  }
  profiler->config_ = std::move(options);
  return profiler;
}

void PrivateUse1ProfilerManager::start() {
  profiler_->start();
  status_ = profiler_->status();
}

void PrivateUse1ProfilerManager::stop() {
  profiler_->stop();
  status_ = profiler_->status();
}

std::vector<std::string> PrivateUse1ProfilerManager::errors() {
  return profiler_->errors();
}

void PrivateUse1ProfilerManager::processTrace(libkineto::ActivityLogger& logger) {
  profiler_->processTrace(logger);
}

void PrivateUse1ProfilerManager::processTrace(
    libkineto::ActivityLogger& logger,
    libkineto::getLinkedActivityCallback getLinkedActivity,
    int64_t startTime,
    int64_t endTime) {
  profiler_->processTrace(logger, std::move(getLinkedActivity), startTime, endTime);
}

std::unique_ptr<libkineto::DeviceInfo> PrivateUse1ProfilerManager::getDeviceInfo() {
  return profiler_->getDeviceInfo();
}

std::vector<libkineto::ResourceInfo> PrivateUse1ProfilerManager::getResourceInfos() {
  return profiler_->getResourceInfos();
}

std::unique_ptr<libkineto::CpuTraceBuffer> PrivateUse1ProfilerManager::getTraceBuffer() {
  return profiler_->getTraceBuffer();
}

void PrivateUse1ProfilerManager::pushCorrelationId(uint64_t id) {
  profiler_->pushCorrelationId(id);
}

void PrivateUse1ProfilerManager::popCorrelationId() {
  profiler_->popCorrelationId();
}

std::unique_ptr<libkineto::IActivityProfilerSession> NPUActivityProfilerPlugin::configure(
    const std::set<libkineto::ActivityType>& activityTypes,
    const libkineto::Config& config) {
  auto profiler = NPUActivityProfilerFactory::create(activityTypes, config);
  if (profiler == nullptr) {
    return nullptr;
  }
  return std::make_unique<PrivateUse1ProfilerManager>(std::move(profiler));
}

std::unique_ptr<libkineto::IActivityProfilerSession> NPUActivityProfilerPlugin::configure(
    int64_t,
    int64_t,
    const std::set<libkineto::ActivityType>& activityTypes,
    const libkineto::Config& config) {
  return configure(activityTypes, config);
}

#ifdef REGISTER_PRIVATEUSE1_PROFILER
REGISTER_PRIVATEUSE1_PROFILER(NPUActivityProfilerPlugin);
#else
namespace {
struct NPUProfilerAutoRegister {
  NPUProfilerAutoRegister() {
    libkineto::api().registerProfilerFactory([]() -> std::unique_ptr<libkineto::IActivityProfiler> {
      return std::make_unique<NPUActivityProfilerPlugin>();
    });
  }
};
static NPUProfilerAutoRegister g_npu_profiler_auto_register;
} // namespace
#endif

} // namespace profiler
} // namespace torch_npu
