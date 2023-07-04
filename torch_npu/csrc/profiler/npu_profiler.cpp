#include <unistd.h>
#include <sys/syscall.h>
#include <torch/csrc/autograd/profiler_legacy.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/profiler/npu_profiler.h"

#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
#include "torch_npu/csrc/toolkit/profiler/inc/data_reporter.h"
namespace torch_npu {
namespace profiler {
using torch_npu::toolkit::profiler::Utils;

static const int64_t g_pid = getpid();

struct NpuProfilerThreadLocalState : public c10::MemoryReportingInfoBase {
  explicit NpuProfilerThreadLocalState(
    const NpuProfilerConfig &config,
    std::set<NpuActivityType> activities)
      : config_(config),
        activities_(std::move(activities)) {}
  ~NpuProfilerThreadLocalState() override = default;

  static NpuProfilerThreadLocalState *getTLS() {
    return static_cast<NpuProfilerThreadLocalState *>(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE)
    );
  }

  const NpuProfilerConfig &config() const {
    return config_;
  }

  const std::set<NpuActivityType> &activities() const {
    return activities_;
  }

  torch_npu::toolkit::profiler::OpRangeData& newOpEvent() {
    std::lock_guard<std::mutex> guard(state_mutex_);
    op_events_.emplace_back(torch_npu::toolkit::profiler::OpRangeData(0, "torch.op_range"));
    return op_events_.back();
  }

  torch_npu::toolkit::profiler::OpRangeData& getOpEvent() {
    std::lock_guard<std::mutex> guard(state_mutex_);
    return op_events_.back();
  }

  void removeOpEvent() {
    std::lock_guard<std::mutex> guard(state_mutex_);
    op_events_.pop_back();
  }

  void finalizeTrace() {
    std::lock_guard<std::mutex> guard(state_mutex_);
    op_events_.clear();
  }

  bool memoryProfilingEnabled() const {
    return config_.profile_memory;
  }

  bool tracePython() {
    return config_.with_stack && activities_.count(NpuActivityType::CPU);
  }

  void setCallbackHandle(at::CallbackHandle handle) {
    handle_ = handle;
  }

  at::CallbackHandle callbackHandle() const {
    return handle_;
  }

  bool hasCallbackHandle() {
    return handle_ > 0;
  }

  void reportMemoryUsage(
    void *ptr,
    int64_t alloc_size,
    c10::Device device) {
    if (config_.profile_memory) {
      static thread_local uint64_t tid = syscall(SYS_gettid);
      std::unique_ptr<torch_npu::toolkit::profiler::MemoryData> data = std::make_unique<torch_npu::toolkit::profiler::MemoryData>(
        0, "torch.memory_usage",
        reinterpret_cast<int64_t>(ptr),
        Utils::GetClockMonotonicRawNs(),
        alloc_size,
        0,
        0,
        static_cast<int8_t>(device.type()),
        device.index(),
        tid,
        g_pid
      );
      reportData(std::move(data));
    }
  }

protected:
  NpuProfilerConfig config_;
  std::set<NpuActivityType> activities_;
  std::deque<torch_npu::toolkit::profiler::OpRangeData> op_events_;
  std::mutex state_mutex_;
  at::CallbackHandle handle_ = 0;
};

bool profDataReportEnable() {
  return ProfilerMgr::GetInstance()->ReportEnable();
}

void initNpuProfiler(const std::string &path, const std::set<NpuActivityType> &activities) {
  if (path.empty()) {
    return;
  }
  std::string absPath = Utils::RelativeToAbsPath(path);
  if (!Utils::IsFileExist(absPath) && !Utils::CreateDir(absPath)) {
    NPU_LOGE("Path %s not exist and create failed.", absPath.c_str());
    return;
  }
  if (!Utils::IsDir(absPath) || !Utils::IsFileWritable(absPath)) {
    NPU_LOGE("%s is not a directory or is not writable.", absPath.c_str());
    return;
  }
  bool npu_trace = false;
  if (activities.count(NpuActivityType::NPU)) {
    npu_trace = true;
  }
  ProfilerMgr::GetInstance()->Init(Utils::RealPath(absPath), npu_trace);
}

static void registerCallback(const std::unordered_set<at::RecordScope> &scopes) {
  auto registeration_state_ptr = NpuProfilerThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(registeration_state_ptr, "Expected profiler state set");
  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(
          [](const at::RecordFunction &fn) -> std::unique_ptr<at::ObserverContext> {
            auto state_ptr = NpuProfilerThreadLocalState::getTLS();
            if (!state_ptr) {
              return nullptr;
            }
            const auto &config = state_ptr->config();
            auto data_ptr = &state_ptr->newOpEvent();
            data_ptr->process_id = g_pid;
            data_ptr->start_ns = Utils::GetClockMonotonicRawNs();
            static thread_local uint64_t tid = syscall(SYS_gettid);
            data_ptr->start_thread_id = tid;
            data_ptr->sequence_number = fn.seqNr();
            data_ptr->forward_thread_id = fn.forwardThreadId();
            data_ptr->is_async = false;
            data_ptr->name = std::string(fn.name().str());
            if (config.record_shapes) {
              data_ptr->input_shapes = torch::autograd::profiler::inputSizes(fn);
            }
            if (config.with_stack && fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
              auto cs = torch::autograd::profiler::prepareCallstack(torch::jit::currentCallstack());
              cs = cs.empty() ? torch::autograd::profiler::prepareCallstack(torch::jit::tracer::pythonCallstack()) : cs;
              data_ptr->stack = torch::autograd::profiler::callstackStr(cs);
            }
            if (config.with_flops) {
              data_ptr->extra_args = torch::autograd::profiler::saveExtraArgs(fn);
            }
            return nullptr;
          },
          [](const at::RecordFunction &fn, at::ObserverContext *) {
            auto state_ptr = NpuProfilerThreadLocalState::getTLS();
            if (!state_ptr) {
              return;
            }
            auto data_ptr = &state_ptr->getOpEvent();
            data_ptr->end_ns = Utils::GetClockMonotonicRawNs();
            static thread_local uint64_t tid = syscall(SYS_gettid);
            data_ptr->end_thread_id = tid;
            std::unique_ptr<torch_npu::toolkit::profiler::OpRangeData> data =
              std::make_unique<torch_npu::toolkit::profiler::OpRangeData>(*data_ptr);
            reportData(std::move(data));
            state_ptr->removeOpEvent();
          }
      )
      .needsInputs(registeration_state_ptr->config().record_shapes)
      .scopes(scopes)
  );
  registeration_state_ptr->setCallbackHandle(handle);
}

void startNpuProfiler(const NpuProfilerConfig &config,
  const std::set<NpuActivityType> &activities,
  const std::unordered_set<at::RecordScope> &scopes) {
  auto state = std::make_shared<NpuProfilerThreadLocalState>(config, activities);
  c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);
  bool cpu_trace = activities.count(NpuActivityType::CPU);
  ExperimentalConfig experimental_config = config.experimental_config;
  NpuTraceConfig npu_config = {experimental_config.trace_level, experimental_config.metrics, config.profile_memory, experimental_config.l2_cache};
  ProfilerMgr::GetInstance()->Start(npu_config, cpu_trace);
  if (cpu_trace) {
    registerCallback(scopes);
  }
}

void stopNpuProfiler() {
  auto state = c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE);
  auto state_ptr = static_cast<NpuProfilerThreadLocalState *>(state.get());
  if (state_ptr->hasCallbackHandle()) {
    at::removeCallback(state_ptr->callbackHandle());
  }
  state_ptr->finalizeTrace();
  ProfilerMgr::GetInstance()->Stop();
}

void finalizeNpuProfiler() {
  ProfilerMgr::GetInstance()->Finalize();
}

void reportData(std::unique_ptr<torch_npu::toolkit::profiler::BaseReportData> data) {
  if (!ProfilerMgr::GetInstance()->ReportEnable()) {
    return;
  }
  ProfilerMgr::GetInstance()->Upload(std::move(data));
}

void reportMarkDataToNpuProfiler(uint32_t category, const std::string &msg, uint64_t correlation_id) {
  if (!ProfilerMgr::GetInstance()->ReportEnable()) {
    return;
  }
  static thread_local uint64_t tid = syscall(SYS_gettid);
  std::unique_ptr<torch_npu::toolkit::profiler::OpMarkData> data = std::make_unique<torch_npu::toolkit::profiler::OpMarkData>(
    0, "torch.op_mark",
    Utils::GetClockMonotonicRawNs(),
    category,
    correlation_id,
    tid,
    g_pid,
    msg
  );
  reportData(std::move(data));
}
} // profiler
} // torch_npu
