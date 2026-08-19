#include "torch_npu/csrc/profiler/mspti_activity_profiler.h"

#include <ATen/record_function.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "libkineto.h"
#include "GenericTraceActivity.h"
#include "torch_npu/csrc/framework/interface/MsptiInterface.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
#include <torch/csrc/profiler/standalone/privateuse1_profiler.h>

namespace torch_npu {
namespace profiler {
namespace {

constexpr size_t kBufferSize = 8 * 1024 * 1024; // 8 MiB

std::mutex g_mtx;
std::vector<MsptiEvent> g_events;
msptiSubscriberHandle g_subscriber = nullptr;

std::unordered_map<uint64_t, uint64_t> g_externalByCorrelation;

struct LaunchSite {
  int64_t timestamp_ns = 0;
  int64_t processId = 0;
  int64_t threadId = 0;
};
std::unordered_map<uint64_t, LaunchSite> g_launchSites;

at::CallbackHandle g_recordFunctionHandle = 0;
std::atomic<uint64_t> g_pushCount{0};
std::atomic<uint64_t> g_popCount{0};
bool g_sessionActive = false;

int64_t g_corrCounter = 1LL << 30; // 1,073,741,824  (< 2^32 - 1)

bool privateUse1Requested(const std::set<libkineto::ActivityType>& types) {
  using AT = libkineto::ActivityType;
  return types.count(AT::CONCURRENT_KERNEL) != 0 || types.count(AT::GPU_MEMCPY) != 0 ||
      types.count(AT::GPU_MEMSET) != 0 || types.count(AT::GPU_USER_ANNOTATION) != 0;
}

std::vector<msptiActivityKind> computeEnabledKinds(const std::set<libkineto::ActivityType>& acts) {
  using AT = libkineto::ActivityType;
  std::vector<msptiActivityKind> ks;
  if (acts.count(AT::CONCURRENT_KERNEL) != 0) {
    ks.push_back(MSPTI_ACTIVITY_KIND_KERNEL);
    ks.push_back(MSPTI_ACTIVITY_KIND_COMMUNICATION);
    ks.push_back(MSPTI_ACTIVITY_KIND_ACL_API);
    ks.push_back(MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
  }
  if (acts.count(AT::GPU_MEMCPY) != 0) {
    ks.push_back(MSPTI_ACTIVITY_KIND_MEMCPY);
  }
  if (acts.count(AT::GPU_MEMSET) != 0) {
    ks.push_back(MSPTI_ACTIVITY_KIND_MEMSET);
  }
  return ks;
}

void bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  void* p = nullptr;
  if (posix_memalign(&p, 8, kBufferSize) != 0 || p == nullptr) {
    ASCEND_LOGE("mspti bufferRequested: failed to allocate %zu bytes", kBufferSize);
    *buffer = nullptr;
    *size = 0;
    *maxNumRecords = 0;
    return;
  }
  *buffer = static_cast<uint8_t*>(p);
  *size = kBufferSize;
  *maxNumRecords = 0;
}

int64_t nextCorrelationId() {
  return g_corrCounter++;
}

void bufferCompleted(uint8_t* buffer, size_t /*size*/, size_t validSize) {
  msptiActivity* record = nullptr;
  std::lock_guard<std::mutex> lk(g_mtx);
  while (at_npu::native::MsptiActivityGetNextRecord(buffer, validSize, &record) && record) {
    switch (record->kind) {
      case MSPTI_ACTIVITY_KIND_KERNEL: {
        auto* k = reinterpret_cast<msptiActivityKernel*>(record);
        MsptiEvent e;
        e.name = (k->name != nullptr) ? k->name : (k->type != nullptr) ? k->type : "mspti_kernel";
        e.start_ns = static_cast<int64_t>(k->start);
        e.end_ns = static_cast<int64_t>(k->end);
        e.device = k->ds.deviceId;
        e.resource = k->ds.streamId;
        e.correlation = nextCorrelationId();
        e.msptiCorrelation = static_cast<int64_t>(k->correlationId);
        e.atype = libkineto::ActivityType::CONCURRENT_KERNEL;
        e.kernelType = (k->type != nullptr) ? k->type : "";
        e.streamId = static_cast<int64_t>(k->ds.streamId);
        g_events.push_back(std::move(e));
        break;
      }
      case MSPTI_ACTIVITY_KIND_COMMUNICATION: {
        auto* c = reinterpret_cast<msptiActivityCommunication*>(record);
        MsptiEvent e;
        e.name = (c->name != nullptr) ? c->name : "mspti_communication";
        e.start_ns = static_cast<int64_t>(c->start);
        e.end_ns = static_cast<int64_t>(c->end);
        e.device = c->ds.deviceId;
        e.resource = c->ds.streamId;
        e.correlation = nextCorrelationId();
        e.atype = libkineto::ActivityType::COLLECTIVE_COMM;
        e.commName = (c->commName != nullptr) ? c->commName : "";
        e.streamId = static_cast<int64_t>(c->ds.streamId);
        g_events.push_back(std::move(e));
        break;
      }
      case MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
        auto* x = reinterpret_cast<msptiActivityExternalCorrelation*>(record);
        g_externalByCorrelation[x->correlationId] = x->externalId;
        break;
      }
      default:
        break;
    }
  }
  ::free(buffer);
}

struct CorrelationContext : public at::ObserverContext {
  explicit CorrelationContext(uint64_t identifier) : id(identifier) {}
  uint64_t id;
};

void registerCorrelationCallback() {
  if (g_recordFunctionHandle != 0) {
    return;
  }
  g_recordFunctionHandle = at::addGlobalCallback(
      at::RecordFunctionCallback(
          [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
            {
              std::lock_guard<std::mutex> lk(g_mtx);
              if (!g_sessionActive) {
                return nullptr;
              }
            }
            const uint64_t id = static_cast<uint64_t>(fn.handle());
            if (!at_npu::native::MsptiActivityPushExternalCorrelationId(MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id)) {
              return nullptr;
            }
            g_pushCount.fetch_add(1, std::memory_order_relaxed);
            {
              std::lock_guard<std::mutex> lk(g_mtx);
              g_launchSites[id] = LaunchSite{
                  static_cast<int64_t>(torch_npu::toolkit::profiler::Utils::GetClockTime()),
                  static_cast<int64_t>(torch_npu::toolkit::profiler::Utils::GetPid()),
                  static_cast<int64_t>(torch_npu::toolkit::profiler::Utils::GetTid())};
            }
            return std::make_unique<CorrelationContext>(id);
          },
          [](const at::RecordFunction&, at::ObserverContext* ctx) {
            auto* correlation = static_cast<CorrelationContext*>(ctx);
            if (correlation == nullptr) {
              return; // nothing was pushed for this operation
            }
            uint64_t popped = 0;
            if (at_npu::native::MsptiActivityPopExternalCorrelationId(
                    MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &popped)) {
              g_popCount.fetch_add(1, std::memory_order_relaxed);
              if (popped != correlation->id) {
                // The stack is per thread, so this can only happen if
                // something else pushes onto the same correlation kind.
                ASCEND_LOGW(
                    "mspti external correlation stack out of step: popped %llu, expected %llu",
                    static_cast<unsigned long long>(popped),
                    static_cast<unsigned long long>(correlation->id));
              }
            }
          })
          .needsInputs(false)
          .needsOutputs(false)
          .needsIds(true));
}

void removeCorrelationCallback() {
  if (g_recordFunctionHandle != 0) {
    at::removeCallback(g_recordFunctionHandle);
    g_recordFunctionHandle = 0;
  }
}

} // namespace

void MsptiActivityProfiler::start() {
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_events.clear();
    g_externalByCorrelation.clear();
    g_launchSites.clear();
  }
  bool support = at_npu::native::IsSupportMsptiFunc();
  if (!support) {
    errors_.emplace_back("libmspti.so not available - mspti session inactive");
    return;
  }
  bool sub = at_npu::native::MsptiSubscribe(&g_subscriber);
  bool reg = at_npu::native::MsptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
  if (!sub)
    errors_.emplace_back("msptiSubscribe failed");
  if (!reg)
    errors_.emplace_back("msptiActivityRegisterCallbacks failed");
  for (auto k : computeEnabledKinds(activities_)) {
    if (!at_npu::native::MsptiActivityEnable(k)) {
      errors_.emplace_back(std::string("msptiActivityEnable failed for kind ") + std::to_string(static_cast<int>(k)));
    }
  }
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_sessionActive = true;
  }
  g_pushCount.store(0, std::memory_order_relaxed);
  g_popCount.store(0, std::memory_order_relaxed);
  registerCorrelationCallback();
  status_ = libkineto::TraceStatus::RECORDING;
}

void MsptiActivityProfiler::stop() {
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_sessionActive = false;
  }
  // Removed before the drain, so no push arrives while the session is tearing
  // down.
  removeCorrelationCallback();
  const uint64_t pushes = g_pushCount.load(std::memory_order_relaxed);
  const uint64_t pops = g_popCount.load(std::memory_order_relaxed);
  if (pushes != pops) {
    ASCEND_LOGW(
        "mspti external correlation unbalanced: %llu pushes, %llu pops",
        static_cast<unsigned long long>(pushes),
        static_cast<unsigned long long>(pops));
  }

  constexpr auto kDrainInterval = std::chrono::milliseconds(25);
  constexpr int kRequiredEmptyStreak = 3;
  constexpr int kSafetyIterations = 400;

  size_t prev = 0;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    prev = g_events.size();
  }
  int emptyStreak = 0;
  int iterations = 0;
  while (iterations++ < kSafetyIterations) {
    at_npu::native::MsptiActivityFlushAll(1);
    std::this_thread::sleep_for(kDrainInterval);
    size_t now = 0;
    {
      std::lock_guard<std::mutex> lk(g_mtx);
      now = g_events.size();
    }
    if (now == prev) {
      if (++emptyStreak >= kRequiredEmptyStreak) {
        break;
      }
    } else {
      emptyStreak = 0;
      prev = now;
    }
  }
  if (iterations >= kSafetyIterations) {
    ASCEND_LOGW(
        "mspti: drain hit the safety bound after %d polls with %zu events; records may still have been arriving",
        iterations,
        prev);
  }

  for (auto k : computeEnabledKinds(activities_)) {
    at_npu::native::MsptiActivityDisable(k);
  }
  if (g_subscriber != nullptr) {
    at_npu::native::MsptiUnsubscribe(g_subscriber);
    g_subscriber = nullptr;
  }
  // Flush once more after unsubscribe to drain the final batch of records.
  at_npu::native::MsptiActivityFlushAll(1);
  std::this_thread::sleep_for(kDrainInterval);
  status_ = libkineto::TraceStatus::PROCESSING;
}

std::vector<std::string> MsptiActivityProfiler::errors() {
  return errors_;
}

void MsptiActivityProfiler::processTrace(libkineto::ActivityLogger& logger) {
  processTraceImpl(logger, 0, 0);
}

void MsptiActivityProfiler::processTrace(
    libkineto::ActivityLogger& logger,
    libkineto::getLinkedActivityCallback /*getLinkedActivity*/,
    int64_t startTime,
    int64_t endTime) {
  processTraceImpl(logger, startTime, endTime);
}

void MsptiActivityProfiler::processTraceImpl(libkineto::ActivityLogger& logger, int64_t startTime, int64_t endTime) {
  std::lock_guard<std::mutex> lk(g_mtx);
  const bool haveWindow = (startTime > 0 && endTime > startTime);
  size_t emitted = 0;
  size_t dropped = 0;
  size_t linked = 0;
  std::set<int64_t> startedFlows;

  for (const auto& e : g_events) {
    if (haveWindow && (e.end_ns < startTime || e.start_ns > endTime)) {
      ++dropped;
      continue;
    }
    libkineto::GenericTraceActivity act;
    act.activityType = e.atype;
    act.activityName = e.name;
    act.startTime = e.start_ns;
    act.endTime = e.end_ns;
    act.id = static_cast<int32_t>(e.correlation);
    act.device = static_cast<int32_t>(e.device);
    act.resource = static_cast<int32_t>(e.resource);

    // Link the kernel back to the operation that launched it. kineto turns
    // flow.type / flow.id / flow.start into the ph:"s" and ph:"f" pairs a
    // trace viewer draws as arrows, so setting those three fields is enough.
    int64_t externalId = e.externalId;
    if (externalId == 0 && e.msptiCorrelation != 0) {
      auto found = g_externalByCorrelation.find(static_cast<uint64_t>(e.msptiCorrelation));
      if (found != g_externalByCorrelation.end()) {
        externalId = static_cast<int64_t>(found->second);
      }
    }
    if (externalId != 0) {
      act.flow.id = static_cast<uint32_t>(externalId);
      act.flow.type = libkineto::kLinkAsyncCpuGpu;
      act.flow.start = 0;
      ++linked;

      auto site = g_launchSites.find(static_cast<uint64_t>(externalId));
      if (site != g_launchSites.end() && startedFlows.find(externalId) == startedFlows.end()) {
        libkineto::GenericTraceActivity tail;
        tail.activityType = libkineto::ActivityType::PRIVATEUSE1_RUNTIME;
        tail.activityName = e.name;
        tail.startTime = e.start_ns;
        tail.endTime = e.start_ns;
        tail.id = static_cast<int32_t>(externalId);
        tail.device = static_cast<int32_t>(site->second.processId);
        tail.resource = static_cast<int32_t>(site->second.threadId);
        tail.flow.id = static_cast<uint32_t>(externalId);
        tail.flow.type = libkineto::kLinkAsyncCpuGpu;
        tail.flow.start = 1;
        tail.log(logger);
        startedFlows.insert(externalId);
      }
    }

    if (e.atype == libkineto::ActivityType::CONCURRENT_KERNEL) {
      if (!e.kernelType.empty()) {
        act.addMetadataQuoted("type", e.kernelType);
      }
      act.addMetadata("streamId", e.streamId);
    } else if (e.atype == libkineto::ActivityType::COLLECTIVE_COMM) {
      if (!e.commName.empty()) {
        act.addMetadataQuoted("commName", e.commName);
      }
      act.addMetadata("streamId", e.streamId);
    }
    act.log(logger);
    ++emitted;
  }
  ASCEND_LOGI(
      "mspti: emitted %zu activities, %zu linked to a launching operation across %zu flows "
      "(%zu outside capture window)",
      emitted,
      linked,
      startedFlows.size(),
      dropped);
}

std::unique_ptr<libkineto::DeviceInfo> MsptiActivityProfiler::getDeviceInfo() {
  // Above any real process id, so the NPU lane renders below the CPU lane.
  constexpr int64_t kNpuSortIndex = 0x1000000;
  return std::make_unique<libkineto::DeviceInfo>(
      libkineto::DeviceInfo{0, kNpuSortIndex, std::string("NPU"), std::string("Ascend NPU")});
}

std::vector<libkineto::ResourceInfo> MsptiActivityProfiler::getResourceInfos() {
  std::lock_guard<std::mutex> lk(g_mtx);
  std::set<std::pair<int64_t, int64_t>> seen;
  std::vector<libkineto::ResourceInfo> out;
  for (const auto& e : g_events) {
    if (e.atype != libkineto::ActivityType::CONCURRENT_KERNEL) {
      continue;
    }
    auto key = std::make_pair(static_cast<int64_t>(e.device), static_cast<int64_t>(e.resource));
    if (seen.insert(key).second) {
      out.push_back(libkineto::ResourceInfo{
          static_cast<int64_t>(e.resource),
          static_cast<int64_t>(e.resource),
          static_cast<int64_t>(e.device),
          std::string("NPU stream ") + std::to_string(e.resource)});
    }
  }
  return out;
}

std::unique_ptr<libkineto::CpuTraceBuffer> MsptiActivityProfiler::getTraceBuffer() {
  return nullptr;
}

std::unique_ptr<libkineto::IActivityProfilerSession> MsptiActivityProfilerPoc::configure(
    const std::set<libkineto::ActivityType>& types,
    const libkineto::Config&) {
  if (!privateUse1Requested(types)) {
    return nullptr;
  }
  return std::make_unique<MsptiActivityProfiler>(types);
}

std::unique_ptr<libkineto::IActivityProfilerSession> MsptiActivityProfilerPoc::configure(
    int64_t,
    int64_t,
    const std::set<libkineto::ActivityType>& types,
    const libkineto::Config& cfg) {
  return configure(types, cfg);
}

// ------------------------------ Registration -------------------------------
// Prefer the upstream macro (stores the factory in PrivateUse1ProfilerRegistry);
#ifdef REGISTER_PRIVATEUSE1_PROFILER
REGISTER_PRIVATEUSE1_PROFILER(MsptiActivityProfilerPoc);
#else
namespace {
struct MsptiAutoRegister {
  MsptiAutoRegister() {
    libkineto::api().registerProfilerFactory(
        []() -> std::unique_ptr<libkineto::IActivityProfiler> { return std::make_unique<MsptiActivityProfilerPoc>(); });
  }
};
static MsptiAutoRegister g_mspti_auto_register;
} // namespace
#endif

} // namespace profiler
} // namespace torch_npu
