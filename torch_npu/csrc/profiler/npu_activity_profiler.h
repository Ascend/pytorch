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

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "IActivityProfiler.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace torch_npu {
namespace profiler {

struct NPUProfilerConfig {
  std::string backend = "mspti";
  // Empty fields mean unspecified; neither field affects backend selection.
  std::string profilerLevel;
  std::vector<std::string> exportTypes;
};

class NPUActivityProfiler {
 public:
  virtual ~NPUActivityProfiler() = default;

  virtual const char* name() const = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual std::vector<std::string> errors() = 0;
  virtual void processTrace(libkineto::ActivityLogger& logger) = 0;
  virtual void processTrace(
      libkineto::ActivityLogger& logger,
      libkineto::getLinkedActivityCallback getLinkedActivity,
      int64_t startTime,
      int64_t endTime) = 0;
  virtual std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() = 0;
  virtual std::vector<libkineto::ResourceInfo> getResourceInfos() = 0;
  virtual std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() = 0;
  virtual void pushCorrelationId(uint64_t) {}
  virtual void popCorrelationId() {}

  libkineto::TraceStatus status() const {
    return status_;
  }

  const NPUProfilerConfig& experimentalConfig() const {
    return config_;
  }

 protected:
  libkineto::TraceStatus status_ = libkineto::TraceStatus::READY;

 private:
  friend class NPUActivityProfilerFactory;
  NPUProfilerConfig config_;
};

class TORCH_NPU_API NPUActivityProfilerFactory {
 public:
  static std::unique_ptr<NPUActivityProfiler> create(
      const std::set<libkineto::ActivityType>& activities,
      const libkineto::Config& config);
};

class TORCH_NPU_API PrivateUse1ProfilerManager : public libkineto::IActivityProfilerSession {
 public:
  explicit PrivateUse1ProfilerManager(std::unique_ptr<NPUActivityProfiler> profiler) : profiler_(std::move(profiler)) {}

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
  void pushCorrelationId(uint64_t id) override;
  void popCorrelationId() override;

 private:
  std::unique_ptr<NPUActivityProfiler> profiler_;
};

class TORCH_NPU_API NPUActivityProfilerPlugin : public libkineto::IActivityProfiler {
 public:
  const std::string& name() const override {
    return name_;
  }
  const std::set<libkineto::ActivityType>& availableActivities() const override {
    return avail_;
  }
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t tsMs,
      int64_t durationMs,
      const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;

 private:
  std::string name_{"mspti"};
  std::set<libkineto::ActivityType> avail_{libkineto::ActivityType::CONCURRENT_KERNEL};
};

} // namespace profiler
} // namespace torch_npu
