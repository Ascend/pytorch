#pragma once
// MSPTI-specific NPU activity-profiler backend.

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "torch_npu/csrc/profiler/npu_activity_profiler.h"

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

class MsptiActivityProfiler : public NPUActivityProfiler {
 public:
  explicit MsptiActivityProfiler(std::set<libkineto::ActivityType> activities) : activities_(std::move(activities)) {}

  const char* name() const override {
    return "mspti";
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
  // Device kernels need a (device, stream) lane or kineto drops them at export.
  std::vector<libkineto::ResourceInfo> getResourceInfos() override;
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override;

 private:
  void processTraceImpl(libkineto::ActivityLogger& logger, int64_t startTime, int64_t endTime);

  std::set<libkineto::ActivityType> activities_;
  std::vector<std::string> errors_;
};

} // namespace profiler
} // namespace torch_npu
