#ifndef FXRT_PROFILER_PROFILER_H_
#define FXRT_PROFILER_PROFILER_H_

#include <sys/types.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <tuple>
#include <unordered_map>

#include "nlohmann/json.hpp"
#include "common/visible.h"
#include "common/common.h"
#include "runtime/utils/spinlock.h"

namespace fxrt {
namespace profiler {

static const char kDefaultOpName[] = "Default";
static const size_t kPercent = 100;

// Profiler stages
enum class ProfilerStage {
  kDefault,
  kPython,
  kInfer,
  kRunGraph,
  kRunOp,
  kMemoryAlloc,
  kMemoryFree,
  kCopyData,
  kStreamSync,
};

// Profiler modules
enum class ProfilerModule { kDefault, kRuntime, kKernel, kPython, kMemory, kOther };

// Profiler events
enum class ProfilerEvent {
  kDefault,
  kInferShape,
  kCalcWorkspace,
  kKernelLaunch,
  kKernelExecute,
  kGraphLaunch,
  kInputProcess,
  kOutputProcess,
  kMemoryAlloc,
  kMemoryFree,
  kCopyData,
  kStreamSync,
  kWaitTaskFinish,
};

// String mappings for enum values
static const std::unordered_map<ProfilerStage, std::string> kProfilerStageString = {
    {ProfilerStage::kDefault, "Default"},
    {ProfilerStage::kPython, "Python"},
    {ProfilerStage::kInfer, "Infer"},
    {ProfilerStage::kRunGraph, "RunGraph"},
    {ProfilerStage::kRunOp, "RunOp"},
    {ProfilerStage::kMemoryAlloc, "MemoryAlloc"},
    {ProfilerStage::kMemoryFree, "MemoryFree"},
    {ProfilerStage::kCopyData, "CopyData"},
    {ProfilerStage::kStreamSync, "StreamSync"},
};

static const std::unordered_map<ProfilerModule, std::string> kProfilerModuleString = {
    {ProfilerModule::kDefault, "Default"},
    {ProfilerModule::kRuntime, "Runtime"},
    {ProfilerModule::kKernel, "Kernel"},
    {ProfilerModule::kPython, "Python"},
    {ProfilerModule::kMemory, "Memory"},
    {ProfilerModule::kOther, "Other"},
};

static const std::unordered_map<ProfilerEvent, std::string> kProfilerEventString = {
    {ProfilerEvent::kDefault, "Default"},
    {ProfilerEvent::kInferShape, "InferShape"},
    {ProfilerEvent::kCalcWorkspace, "CalcWorkspace"},
    {ProfilerEvent::kKernelLaunch, "KernelLaunch"},
    {ProfilerEvent::kKernelExecute, "KernelExecute"},
    {ProfilerEvent::kGraphLaunch, "GraphLaunch"},
    {ProfilerEvent::kInputProcess, "InputProcess"},
    {ProfilerEvent::kOutputProcess, "OutputProcess"},
    {ProfilerEvent::kMemoryAlloc, "MemoryAlloc"},
    {ProfilerEvent::kMemoryFree, "MemoryFree"},
    {ProfilerEvent::kCopyData, "CopyData"},
    {ProfilerEvent::kStreamSync, "StreamSync"},
    {ProfilerEvent::kWaitTaskFinish, "WaitTaskFinish"},
};

// Profiler data statistics information
struct ProfilerStatisticsInfo {
  explicit ProfilerStatisticsInfo(const std::string& name, bool isInnerEvent = false)
      : name(name), totalTime(0), count(0), minTime(UINT64_MAX), maxTime(0), isInnerEvent(isInnerEvent) {}

  void AccumulateTime(uint64_t time) {
    totalTime += time;
    ++count;
    if (time < minTime) {
      minTime = time;
    }
    if (time > maxTime) {
      maxTime = time;
    }
  }

  std::string name;
  uint64_t totalTime;
  uint64_t count;
  uint64_t minTime;
  uint64_t maxTime;
  bool isInnerEvent;
};

using ProfilerStatisticsInfoPtr = std::shared_ptr<ProfilerStatisticsInfo>;

// Profiler data structure
struct ProfilerData {
  ProfilerData(
      ProfilerModule module,
      ProfilerEvent event,
      const std::string& opName,
      bool isInnerEvent,
      uint64_t startTime,
      uint64_t endTime,
      uint64_t flowId = UINT64_MAX)
      : isStage(false),
        stage(ProfilerStage::kDefault),
        module(module),
        event(event),
        opName(opName),
        isInnerEvent(isInnerEvent),
        startTime(startTime),
        endTime(endTime),
        durTime(endTime - startTime),
        tid(std::hash<std::thread::id>{}(std::this_thread::get_id())),
        pid(getpid()),
        flowId(flowId) {}

  ProfilerData(ProfilerStage stage, uint64_t startTime, uint64_t endTime)
      : isStage(true),
        stage(stage),
        module(ProfilerModule::kDefault),
        event(ProfilerEvent::kDefault),
        opName(),
        isInnerEvent(false),
        startTime(startTime),
        endTime(endTime),
        durTime(endTime - startTime),
        tid(std::hash<std::thread::id>{}(std::this_thread::get_id())),
        pid(getpid()),
        flowId(UINT64_MAX) {}

  ProfilerData(const ProfilerData& other) = default;
  ProfilerData& operator=(const ProfilerData& other) = default;

  bool isStage;
  ProfilerStage stage;
  ProfilerModule module;
  ProfilerEvent event;
  std::string opName;
  bool isInnerEvent;
  uint64_t startTime;
  uint64_t endTime;
  uint64_t durTime;
  uint64_t tid;
  uint32_t pid;
  uint64_t flowId;
};

using ProfilerDataPtr = std::shared_ptr<ProfilerData>;
using ProfilerDataSpan = std::vector<ProfilerDataPtr>;

// Profiler event information
struct ProfilerEventInfo {
  ProfilerEventInfo() = default;
  ProfilerStatisticsInfoPtr eventStatisticsInfo;
  std::unordered_map<std::string, ProfilerStatisticsInfoPtr> opInfos;
};

using ProfilerEventInfoPtr = std::shared_ptr<ProfilerEventInfo>;

// Profiler module information
struct ProfilerModuleInfo {
  ProfilerModuleInfo() = default;
  ProfilerStatisticsInfoPtr moduleStatisticsInfo;
  std::unordered_map<ProfilerEvent, ProfilerEventInfoPtr> eventInfos;
};

using ProfilerModuleInfoPtr = std::shared_ptr<ProfilerModuleInfo>;

// Step information
struct StepInfo {
  StepInfo(size_t step, uint64_t stepTime) : step(step), stepTime(stepTime) {}
  const size_t step;
  const uint64_t stepTime;
};

using StepInfoPtr = std::shared_ptr<StepInfo>;

// Profiler macros
#define FXRT_PROFILER_START(startTime)                                                        \
  do {                                                                                        \
    if (FXRT_UNLIKELY(fxrt::profiler::ProfilerAnalyzer::GetInstance().IsProfilerEnabled())) { \
      startTime = fxrt::profiler::ProfilerAnalyzer::GetInstance().GetTimeStamp();             \
    }                                                                                         \
  } while (0);

#define FXRT_PROFILER_END(startTime, module, event, opName, isInnerEvent)                                           \
  do {                                                                                                              \
    if (FXRT_UNLIKELY(fxrt::profiler::ProfilerAnalyzer::GetInstance().IsProfilerEnabled())) {                       \
      auto endTime = fxrt::profiler::ProfilerAnalyzer::GetInstance().GetTimeStamp();                                \
      fxrt::profiler::ProfilerAnalyzer::GetInstance().RecordData(                                                   \
          std::make_shared<fxrt::profiler::ProfilerData>(module, event, opName, isInnerEvent, startTime, endTime)); \
    }                                                                                                               \
  } while (0);

#define FXRT_PROFILER_STAGE_END(startTime, stage)                                             \
  do {                                                                                        \
    if (FXRT_UNLIKELY(fxrt::profiler::ProfilerAnalyzer::GetInstance().IsProfilerEnabled())) { \
      auto endTime = fxrt::profiler::ProfilerAnalyzer::GetInstance().GetTimeStamp();          \
      fxrt::profiler::ProfilerAnalyzer::GetInstance().RecordData(                             \
          std::make_shared<fxrt::profiler::ProfilerData>(stage, startTime, endTime));         \
    }                                                                                         \
  } while (0);

// Profiler recorder class for automatic profiling using RAII
class FXRT_EXPORT ProfilerRecorder {
 public:
  ProfilerRecorder(
      ProfilerModule module,
      ProfilerEvent event,
      const std::string& opName,
      bool isInnerEvent = false,
      uint64_t flowId = UINT64_MAX);
  ~ProfilerRecorder();

 private:
  struct Data {
    Data(
        ProfilerModule module,
        ProfilerEvent event,
        std::string opName,
        uint64_t startTime,
        uint64_t flowId,
        bool isInnerEvent)
        : module(module),
          event(event),
          opName(std::move(opName)),
          startTime(startTime),
          flowId(flowId),
          isInnerEvent(isInnerEvent) {}
    ProfilerModule module;
    ProfilerEvent event;
    std::string opName;
    uint64_t startTime;
    uint64_t flowId;
    bool isInnerEvent;
  };

  std::unique_ptr<Data> data_;
};

// Profiler stage recorder
class FXRT_EXPORT ProfilerStageRecorder {
 public:
  explicit ProfilerStageRecorder(ProfilerStage stage);
  ~ProfilerStageRecorder();

 private:
  ProfilerStage stage_;
  uint64_t startTime_;
};

// Profiler analyzer singleton class
class FXRT_EXPORT ProfilerAnalyzer {
 public:
  static ProfilerAnalyzer& GetInstance() noexcept;

  // Initialize profiler
  void Initialize();

  // Check if profiler is enabled
  bool IsProfilerEnabled() const;

  // Enable/disable profiler
  void Enable();
  void Disable();

  // Record data
  void RecordData(const ProfilerDataPtr& data) noexcept;

  // Step management
  void StartStep();
  void EndStep();
  // Get current timestamp
  uint64_t GetTimeStamp() const noexcept;

  // Process and dump data
  void ProcessData();
  void Clear();
  void Reset();

 private:
  ProfilerAnalyzer() = default;
  ~ProfilerAnalyzer() {
    Clear();
  }

  ProfilerAnalyzer(const ProfilerAnalyzer&) = delete;
  ProfilerAnalyzer& operator=(const ProfilerAnalyzer&) = delete;

  // Helper methods
  void SaveJsonData(const ProfilerDataPtr& data);
  void AnalyzeSummaryData(const ProfilerDataPtr& data);
  void AnalyzeStageSummaryData(const ProfilerDataPtr& data);
  void AnalyzeModuleSummaryData(const ProfilerDataPtr& data);
  void AnalyzeEventSummaryData(const ProfilerDataPtr& data);
  void AnalyzeOpSummaryData(
      std::unordered_map<std::string, ProfilerStatisticsInfoPtr>* opInfos,
      const ProfilerDataPtr& data);
  void DumpJsonData();
  void DumpSummaryData(size_t step);
  void DumpDetailData(size_t step, const ProfilerDataSpan& span);

  // Member variables
  bool init_{false};
  bool profilerEnabled_{false};
  size_t step_{0};
  uint64_t stepStartTime_{0};
  uint64_t stepTime_{0};
  uint64_t moduleTotalTime_{0};
  size_t showTopNum_{10};

  // Data storage
  std::vector<ProfilerDataPtr> data_;
  std::vector<std::pair<StepInfoPtr, ProfilerDataSpan>> dataLine_;
  std::vector<nlohmann::json> jsonInfos_;

  // Summary information
  std::unordered_map<ProfilerModule, ProfilerModuleInfoPtr> moduleInfos_;
  std::unordered_map<ProfilerStage, ProfilerStatisticsInfoPtr> stageInfos_;

  // Thread management
  fxrt::runtime::SpinLock dataMutex_;

  // File names
  std::string jsonFileName_;
  std::string summaryInfoFileName_;
  std::string detailInfoFileName_;
};

} // namespace profiler
} // namespace fxrt

#endif // FXRT_PROFILER_PROFILER_H_
