#include "profiler/profiler.h"
#include <sys/types.h>
#include <unistd.h>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace fxrt {
namespace profiler {
static constexpr int kPrecisionDigits = 2;
static constexpr double kNsToUs = 1000;

// File name constants
static const char kJsonFileName[] = "RuntimeProfilerJson";
static const char kSummaryInfoFileName[] = "RuntimeProfilerSummary";
static const char kDetailInfoFileName[] = "RuntimeProfilerDetail";

// JSON field constants
static const char kJsonName[] = "name";
static const char kJsonPh[] = "ph";
static const char kJsonPid[] = "pid";
static const char kJsonTid[] = "tid";
static const char kJsonTs[] = "ts";
static const char kJsonDur[] = "dur";
static const char kJsonPhX[] = "X";
static const char kJsonArgs[] = "args";
static const char kJsonFlowId[] = "flow_id";

// Environment variables
static const char kEnableRuntimeProfiler[] = "FXRT_ENABLE_RUNTIME_PROFILER";
static const char kRuntimeProfilerTopNum[] = "FXRT_ENABLE_PROFILER_TOP_NUM";

// Get real path for saving files
std::string GetRealPathName(const std::string& name) {
  // Simple implementation, can be enhanced
  return "./" + name;
}

// Get current timestamp in nanoseconds
static uint64_t GetClockTimeNs() {
  auto ts = std::chrono::system_clock::now();
  int64_t system_t = std::chrono::duration_cast<std::chrono::nanoseconds>(ts.time_since_epoch()).count();
  return static_cast<uint64_t>(system_t);
}

// ProfilerRecorder implementation
ProfilerRecorder::ProfilerRecorder(
    ProfilerModule module,
    ProfilerEvent event,
    const std::string& opName,
    bool isInnerEvent,
    uint64_t flowId) {
  auto& profiler = ProfilerAnalyzer::GetInstance();
  if (!profiler.IsProfilerEnabled()) {
    return;
  }

  data_ = std::make_unique<Data>(module, event, opName, profiler.GetTimeStamp(), flowId, isInnerEvent);
}

ProfilerRecorder::~ProfilerRecorder() {
  auto& profiler = ProfilerAnalyzer::GetInstance();
  if (!profiler.IsProfilerEnabled()) {
    return;
  }
  if (data_ == nullptr) {
    return;
  }
  profiler.RecordData(std::make_shared<ProfilerData>(
      data_->module,
      data_->event,
      data_->opName,
      data_->isInnerEvent,
      data_->startTime,
      profiler.GetTimeStamp(),
      data_->flowId));
}

// ProfilerStageRecorder implementation
ProfilerStageRecorder::ProfilerStageRecorder(ProfilerStage stage) {
  if (!ProfilerAnalyzer::GetInstance().IsProfilerEnabled()) {
    return;
  }
  startTime_ = ProfilerAnalyzer::GetInstance().GetTimeStamp();
  this->stage_ = stage;
}

ProfilerStageRecorder::~ProfilerStageRecorder() {
  if (!ProfilerAnalyzer::GetInstance().IsProfilerEnabled()) {
    return;
  }
  ProfilerAnalyzer::GetInstance().RecordData(
      std::make_shared<ProfilerData>(stage_, startTime_, ProfilerAnalyzer::GetInstance().GetTimeStamp()));
}

// ProfilerAnalyzer implementation
ProfilerAnalyzer& ProfilerAnalyzer::GetInstance() noexcept {
  static ProfilerAnalyzer instance{};
  return instance;
}

void ProfilerAnalyzer::Initialize() {
  if (init_) {
    return;
  }
  std::unique_lock<fxrt::runtime::SpinLock> lock(dataMutex_);
  init_ = true;

  // Check environment variable to enable profiler. GetEnv copies the value out
  // of the environment, so a concurrent setenv can not invalidate it under us.
  const std::string enableProfiler = GetEnv(kEnableRuntimeProfiler);
  if (enableProfiler == "1") {
    profilerEnabled_ = true;
  }

  // Get top number from environment variable
  const std::string topNumEnv = GetEnv(kRuntimeProfilerTopNum);
  if (!topNumEnv.empty()) {
    try {
      showTopNum_ = std::stoi(topNumEnv);
    } catch (const std::exception& e) {
      // Fall back to default
      showTopNum_ = 10;
    }
  }

  // Generate file names with timestamp
  auto nowTime = std::to_string(GetTimeStamp());
  jsonFileName_ = GetRealPathName(kJsonFileName + nowTime + ".json");
  summaryInfoFileName_ = GetRealPathName(kSummaryInfoFileName + nowTime + ".csv");
  detailInfoFileName_ = GetRealPathName(kDetailInfoFileName + nowTime + ".csv");
}

bool ProfilerAnalyzer::IsProfilerEnabled() const {
  return profilerEnabled_;
}

void ProfilerAnalyzer::Enable() {
  profilerEnabled_ = true;
}

void ProfilerAnalyzer::Disable() {
  profilerEnabled_ = false;
}

uint64_t ProfilerAnalyzer::GetTimeStamp() const noexcept {
  return GetClockTimeNs();
}

void ProfilerAnalyzer::RecordData(const ProfilerDataPtr& data) noexcept {
  if (!data) {
    return;
  }
  std::unique_lock<fxrt::runtime::SpinLock> lock(dataMutex_);
  if (profilerEnabled_) {
    this->data_.emplace_back(data);
  }
}

void ProfilerAnalyzer::StartStep() {
  Initialize();
  if (!IsProfilerEnabled()) {
    return;
  }

  std::unique_lock<fxrt::runtime::SpinLock> lock(dataMutex_);
  ++step_;
  data_.clear();
  stepStartTime_ = GetTimeStamp();
}

void ProfilerAnalyzer::EndStep() {
  if (!IsProfilerEnabled()) {
    return;
  }

  std::unique_lock<fxrt::runtime::SpinLock> lock(dataMutex_);
  if (data_.empty()) {
    return;
  }

  stepTime_ = GetTimeStamp() - stepStartTime_;
  dataLine_.emplace_back(std::make_shared<StepInfo>(step_, stepTime_), std::move(data_));
}

void ProfilerAnalyzer::ProcessData() {
  for (const auto& [stepInfoPtr, span] : dataLine_) {
    stepTime_ = stepInfoPtr->stepTime;
    // Process data
    for (auto& data : span) {
      SaveJsonData(data);
      AnalyzeSummaryData(data);
    }
    // Dump data
    DumpDetailData(stepInfoPtr->step, span);
    DumpSummaryData(stepInfoPtr->step);
    // Clear temp data
    moduleTotalTime_ = 0;
    moduleInfos_.clear();
    stageInfos_.clear();
  }
}

void ProfilerAnalyzer::Clear() {
  std::unique_lock<fxrt::runtime::SpinLock> lock(dataMutex_);
  if (!init_ || !profilerEnabled_ || dataLine_.empty()) {
    return;
  }
  ProcessData();

  // Dump JSON data
  DumpJsonData();
  data_.clear();
  dataLine_.clear();
}

void ProfilerAnalyzer::Reset() {
  init_ = false;
  step_ = 0;
  stepStartTime_ = 0;
  stepTime_ = 0;
  moduleTotalTime_ = 0;
  showTopNum_ = 10;
  data_.clear();
  dataLine_.clear();
  jsonInfos_.clear();
}

void ProfilerAnalyzer::SaveJsonData(const ProfilerDataPtr& data) {
  if (!data) {
    return;
  }
  nlohmann::json jsonData;
  if (data->isStage) {
    jsonData[kJsonName] = kProfilerStageString.at(data->stage);
  } else {
    jsonData[kJsonName] =
        kProfilerModuleString.at(data->module) + "::" + kProfilerEventString.at(data->event) + "::" + data->opName;
  }
  jsonData[kJsonPh] = kJsonPhX;
  jsonData[kJsonPid] = std::to_string(data->pid);
  jsonData[kJsonTid] = std::to_string(data->tid);
  jsonData[kJsonTs] = static_cast<double>(data->startTime) / kNsToUs;
  jsonData[kJsonDur] = static_cast<double>(data->durTime) / kNsToUs;
  nlohmann::json args;
  args[kJsonFlowId] = data->flowId;
  jsonData[kJsonArgs] = args;

  jsonInfos_.emplace_back(jsonData);
}

void ProfilerAnalyzer::AnalyzeSummaryData(const ProfilerDataPtr& data) {
  if (!data) {
    return;
  }
  if (data->isStage) {
    AnalyzeStageSummaryData(data);
  } else {
    AnalyzeEventSummaryData(data);
  }
}

void ProfilerAnalyzer::AnalyzeStageSummaryData(const ProfilerDataPtr& data) {
  if (!data) {
    return;
  }
  if (stageInfos_.count(data->stage) == 0) {
    auto stageInfo = std::make_shared<ProfilerStatisticsInfo>(kProfilerStageString.at(data->stage), false);
    stageInfos_[data->stage] = stageInfo;
  }
  stageInfos_[data->stage]->AccumulateTime(data->durTime);
}

void ProfilerAnalyzer::AnalyzeModuleSummaryData(const ProfilerDataPtr& data) {
  if (!data) {
    return;
  }
  if (moduleInfos_.count(data->module) == 0) {
    auto moduleInfoPtr = std::make_shared<ProfilerModuleInfo>();
    moduleInfoPtr->moduleStatisticsInfo =
        std::make_shared<ProfilerStatisticsInfo>(kProfilerModuleString.at(data->module));
    moduleInfos_[data->module] = moduleInfoPtr;
  }
  moduleInfos_[data->module]->moduleStatisticsInfo->AccumulateTime(data->durTime);
  moduleTotalTime_ += data->durTime;
}

void ProfilerAnalyzer::AnalyzeEventSummaryData(const ProfilerDataPtr& data) {
  if (!data) {
    return;
  }
  AnalyzeModuleSummaryData(data);

  if (moduleInfos_.count(data->module) == 0) {
    return;
  }

  auto& moduleInfoPtr = moduleInfos_[data->module];
  auto& eventInfosPtr = moduleInfoPtr->eventInfos;
  if (eventInfosPtr.count(data->event) == 0) {
    auto eventInfoPtr = std::make_shared<ProfilerEventInfo>();
    eventInfoPtr->eventStatisticsInfo =
        std::make_shared<ProfilerStatisticsInfo>(kProfilerEventString.at(data->event), data->isInnerEvent);
    eventInfosPtr[data->event] = eventInfoPtr;
  }

  auto& eventInfoPtr = eventInfosPtr[data->event];
  eventInfoPtr->eventStatisticsInfo->AccumulateTime(data->durTime);
  AnalyzeOpSummaryData(&eventInfoPtr->opInfos, data);
}

void ProfilerAnalyzer::AnalyzeOpSummaryData(
    std::unordered_map<std::string, ProfilerStatisticsInfoPtr>* opInfos,
    const ProfilerDataPtr& data) {
  if (!opInfos || !data) {
    return;
  }
  if (opInfos->count(data->opName) == 0) {
    auto opInfoPtr = std::make_shared<ProfilerStatisticsInfo>(data->opName, data->isInnerEvent);
    (*opInfos)[data->opName] = opInfoPtr;
  }
  (*opInfos)[data->opName]->AccumulateTime(data->durTime);
}

void ProfilerAnalyzer::DumpJsonData() {
  std::ofstream jsonFile(jsonFileName_);
  if (!jsonFile.is_open()) {
    RT_GLOG(ERROR) << "Failed to open json file: " << jsonFileName_;
    return;
  }

  jsonFile << "[\n";
  for (size_t i = 0; i < jsonInfos_.size(); ++i) {
    jsonFile << jsonInfos_[i];
    if (i != jsonInfos_.size() - 1) {
      jsonFile << ",\n";
    }
  }
  jsonFile << "\n]";
  jsonFile.close();
}

void ProfilerAnalyzer::DumpSummaryData(size_t step) {
  std::ofstream summaryFile(summaryInfoFileName_, std::ios::app);
  if (!summaryFile.is_open()) {
    RT_GLOG(ERROR) << "Failed to open summary file: " << summaryInfoFileName_;
    return;
  }

  // Write header if file is empty
  if (summaryFile.tellp() == 0) {
    summaryFile << "Step,Module,Event,OpName,Count,TotalTime(us),AvgTime(us),MinTime(us),MaxTime(us),Percent(%)\n";
  }

  // Write module summary
  for (const auto& [module, moduleInfo] : moduleInfos_) {
    auto totalTime = moduleInfo->moduleStatisticsInfo->totalTime / kNsToUs;
    auto percent =
        (stepTime_ > 0) ? (static_cast<double>(moduleInfo->moduleStatisticsInfo->totalTime) / stepTime_ * kPercent) : 0;
    summaryFile << step << "," << moduleInfo->moduleStatisticsInfo->name << ",,,"
                << moduleInfo->moduleStatisticsInfo->count << "," << std::fixed << std::setprecision(kPrecisionDigits)
                << totalTime << ",,," << std::fixed << std::setprecision(kPrecisionDigits) << percent << "%\n";

    // Write event summary
    for (const auto& [event, eventInfo] : moduleInfo->eventInfos) {
      auto eventTotalTime = eventInfo->eventStatisticsInfo->totalTime / kNsToUs;
      auto eventPercent =
          (stepTime_ > 0) ? (static_cast<double>(eventInfo->eventStatisticsInfo->totalTime) / stepTime_ * kPercent) : 0;
      auto eventAvgTime = (eventInfo->eventStatisticsInfo->count > 0)
          ? (eventInfo->eventStatisticsInfo->totalTime / eventInfo->eventStatisticsInfo->count / kNsToUs)
          : 0;
      auto eventMinTime = eventInfo->eventStatisticsInfo->minTime / kNsToUs;
      auto eventMaxTime = eventInfo->eventStatisticsInfo->maxTime / kNsToUs;

      summaryFile << step << "," << moduleInfo->moduleStatisticsInfo->name << ","
                  << eventInfo->eventStatisticsInfo->name << ",," << eventInfo->eventStatisticsInfo->count << ","
                  << std::fixed << std::setprecision(kPrecisionDigits) << eventTotalTime << "," << std::fixed
                  << std::setprecision(kPrecisionDigits) << eventAvgTime << "," << std::fixed
                  << std::setprecision(kPrecisionDigits) << eventMinTime << "," << std::fixed
                  << std::setprecision(kPrecisionDigits) << eventMaxTime << "," << std::fixed
                  << std::setprecision(kPrecisionDigits) << eventPercent << "%\n";

      // Write op summary (top N)
      std::vector<std::pair<std::string, ProfilerStatisticsInfoPtr>> opList;
      for (const auto& [opName, opInfo] : eventInfo->opInfos) {
        opList.emplace_back(opName, opInfo);
      }

      // Sort by total time
      std::sort(opList.begin(), opList.end(), [](const auto& a, const auto& b) {
        return a.second->totalTime > b.second->totalTime;
      });

      // Write top N ops
      size_t count = 0;
      for (const auto& [opName, opInfo] : opList) {
        if (count >= showTopNum_) {
          break;
        }
        auto opTotalTime = opInfo->totalTime / kNsToUs;
        auto opPercent = (stepTime_ > 0) ? (static_cast<double>(opInfo->totalTime) / stepTime_ * kPercent) : 0;
        auto opAvgTime = (opInfo->count > 0) ? (opInfo->totalTime / opInfo->count / kNsToUs) : 0;
        auto opMinTime = opInfo->minTime / kNsToUs;
        auto opMaxTime = opInfo->maxTime / kNsToUs;

        summaryFile << step << "," << moduleInfo->moduleStatisticsInfo->name << ","
                    << eventInfo->eventStatisticsInfo->name << "," << opName << "," << opInfo->count << ","
                    << std::fixed << std::setprecision(kPrecisionDigits) << opTotalTime << "," << std::fixed
                    << std::setprecision(kPrecisionDigits) << opAvgTime << "," << std::fixed
                    << std::setprecision(kPrecisionDigits) << opMinTime << "," << std::fixed
                    << std::setprecision(kPrecisionDigits) << opMaxTime << "," << std::fixed
                    << std::setprecision(kPrecisionDigits) << opPercent << "%\n";
        ++count;
      }
    }
  }

  summaryFile.close();
}

void ProfilerAnalyzer::DumpDetailData(size_t step, const ProfilerDataSpan& span) {
  std::ofstream detailFile(detailInfoFileName_, std::ios::app);
  if (!detailFile.is_open()) {
    RT_GLOG(ERROR) << "Failed to open detail file: " << detailInfoFileName_;
    return;
  }

  // Write header if file is empty
  if (detailFile.tellp() == 0) {
    detailFile << "Step,Type,Module/Stage,Event,OpName,StartTime(us),EndTime(us),Duration(us),ThreadId\n";
  }

  for (const auto& data : span) {
    if (data->isStage) {
      detailFile << step << ",Stage," << kProfilerStageString.at(data->stage) << ",,," << data->startTime / kNsToUs
                 << "," << data->endTime / kNsToUs << "," << data->durTime / kNsToUs << "," << data->tid << "\n";
    } else {
      detailFile << step << ",Event," << kProfilerModuleString.at(data->module) << ","
                 << kProfilerEventString.at(data->event) << "," << data->opName << "," << data->startTime / kNsToUs
                 << "," << data->endTime / kNsToUs << "," << data->durTime / kNsToUs << "," << data->tid << "\n";
    }
  }

  detailFile.close();
}

} // namespace profiler
} // namespace fxrt
