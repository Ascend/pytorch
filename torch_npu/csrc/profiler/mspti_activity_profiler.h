#pragma once
// mspti NPU activity-profiler session + libkineto bridge.
//
// MsptiActivityProfilerPoc (libkineto::IActivityProfiler) registers for
// ProfilerActivity.PrivateUse1; configure() returns a MsptiActivityProfiler,
// which is the session libkineto drives. The session enables mspti kinds from
// the requested activity-type set and collects NPU kernel and host API records.

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "IActivityProfiler.h"

namespace torch_npu {
namespace profiler {

struct MsptiEvent {
  std::string name;
  int64_t start_ns = 0;
  int64_t end_ns = 0;
  int64_t device = 0;
  int64_t resource = 0;
  int64_t correlation = 0;
  int64_t msptiCorrelation = 0;
  int64_t externalId = 0;
  libkineto::ActivityType atype = libkineto::ActivityType::CONCURRENT_KERNEL;
  std::string kernelType;
  std::string commName;
  int64_t streamId = 0;
};

class MsptiActivityProfiler : public libkineto::IActivityProfilerSession {
 public:
  explicit MsptiActivityProfiler(std::set<libkineto::ActivityType> activities) : activities_(std::move(activities)) {}

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
  // Device kernels need a (device, stream) lane or kineto drops them at export.
  std::vector<libkineto::ResourceInfo> getResourceInfos() override;
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override;

 private:
  void processTraceImpl(libkineto::ActivityLogger& logger, int64_t startTime, int64_t endTime);

  std::set<libkineto::ActivityType> activities_;
  std::vector<std::string> errors_;
};

class MsptiActivityProfilerPoc : public libkineto::IActivityProfiler {
 public:
  const std::string& name() const override {
    return name_;
  }
  const std::set<libkineto::ActivityType>& availableActivities() const override {
    return avail_;
  }
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activity_types,
      const libkineto::Config& config) override;
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t ts_ms,
      int64_t duration_ms,
      const std::set<libkineto::ActivityType>& activity_types,
      const libkineto::Config& config) override;

 private:
  std::string name_{"mspti"};
  std::set<libkineto::ActivityType> avail_{libkineto::ActivityType::CONCURRENT_KERNEL};
};

} // namespace profiler
} // namespace torch_npu
