#include <c10/util/Exception.h>
#include <torch/csrc/profiler/util.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/profiler/npu_profiler.h"

#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
#include "torch_npu/csrc/toolkit/profiler/inc/data_reporter.h"
namespace torch_npu {
namespace profiler {
using torch_npu::toolkit::profiler::Utils;

struct NpuObserverContext : public at::ObserverContext {
    explicit NpuObserverContext(std::unique_ptr<torch_npu::toolkit::profiler::OpRangeData> data) : data_(std::move(data)) {}
    std::unique_ptr<torch_npu::toolkit::profiler::OpRangeData> data_;
};

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

  std::unique_ptr<NpuObserverContext> newOpEvent(const at::RecordFunction &fn) {
    return std::make_unique<NpuObserverContext>(
      std::make_unique<torch_npu::toolkit::profiler::OpRangeData>(
        static_cast<int64_t>(Utils::GetClockTime()),
        0,
        fn.seqNr(),
        Utils::GetPid(),
        Utils::GetTid(),
        0,
        fn.forwardThreadId(),
        fn.isAsync(),
        fn.name())
    );
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
    size_t total_allocated,
    size_t total_reserved,
    c10::Device device) {
    if (config_.profile_memory && ProfilerMgr::GetInstance()->ReportEnable().load(std::memory_order_relaxed)) {
      ProfilerMgr::GetInstance()->Upload(std::make_unique<torch_npu::toolkit::profiler::MemoryData>(
        reinterpret_cast<int64_t>(ptr),
        static_cast<int64_t>(Utils::GetClockTime()),
        alloc_size,
        static_cast<int64_t>(total_allocated),
        static_cast<int64_t>(total_reserved),
        static_cast<int8_t>(device.type()),
        device.index(),
        Utils::GetTid(),
        Utils::GetPid()
      ));
    }
  }

protected:
  NpuProfilerConfig config_;
  std::set<NpuActivityType> activities_;
  at::CallbackHandle handle_ = 0;
};

std::atomic<bool>& profDataReportEnable() {
  return ProfilerMgr::GetInstance()->ReportEnable();
}

void initNpuProfiler(const std::string &path, const std::set<NpuActivityType> &activities) {
  if (path.empty()) {
    return;
  }
  std::string absPath = Utils::RelativeToAbsPath(path);
  if (Utils::IsSoftLink(absPath)) {
    ASCEND_LOGE("Path %s is soft link.", absPath.c_str());
    return;
  }
  if (!Utils::IsFileExist(absPath) && !Utils::CreateDir(absPath)) {
    ASCEND_LOGE("Path %s not exist and create failed.", absPath.c_str());
    return;
  }
  if (!Utils::IsDir(absPath) || !Utils::IsFileWritable(absPath)) {
    ASCEND_LOGE("%s is not a directory or is not writable.", absPath.c_str());
    return;
  }
  bool npu_trace = false;
  if (activities.count(NpuActivityType::NPU)) {
    npu_trace = true;
  }
  std::string realPath = Utils::RealPath(absPath);
  TORCH_CHECK(!realPath.empty(), "Invalid path", path);
  ProfilerMgr::GetInstance()->Init(realPath, npu_trace);
}

static void parseInputShapesAndDtypes(const at::RecordFunction &fn,
                                      std::vector<std::string> &dtypes,
                                      std::vector<std::vector<int64_t>> &shapes)
{
    auto inputs = fn.inputs();
    for (const auto &value : inputs) {
        std::vector<int64_t> shape;
        std::string dtype = "";
        if (value.isTensor()) {
            const at::Tensor &t = value.toTensor();
            if (t.defined() && !t.is_nested() && !t.unsafeGetTensorImpl()->has_symbolic_sizes_strides()) {
                dtype = std::string(scalarTypeToTypeMeta(t.scalar_type()).name());
                for (auto i: t.sizes()) {
                    shape.emplace_back(i);
                }
            }
        } else if (value.isTensorList()) {
            dtype = "TensorList";
        } else if (value.isScalar()) {
            dtype = "Scalar";
        } else if (value.isList()) {
            auto listRef = value.toListRef();
            if (listRef.empty() || listRef[0].isScalar()) {
                dtype = "ScalarList";
            }
        }
        dtypes.emplace_back(std::move(dtype));
        shapes.emplace_back(std::move(shape));
    }
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
            auto ctx_ptr = state_ptr->newOpEvent(fn);
            auto &data_ptr = ctx_ptr->data_;
            if ((C10_UNLIKELY(config.record_shapes))) {
                parseInputShapesAndDtypes(fn, data_ptr->input_dtypes, data_ptr->input_shapes);
            }
            if (C10_UNLIKELY(config.with_stack && fn.scope() != at::RecordScope::BACKWARD_FUNCTION)) {
              auto cs = torch::profiler::impl::prepareCallstack(torch::jit::currentCallstack());
              cs = cs.empty() ? torch::profiler::impl::prepareCallstack(torch::jit::tracer::pythonCallstack()) : cs;
              data_ptr->stack = torch::profiler::impl::callstackStr(cs);
            }
            if (C10_UNLIKELY(config.with_modules && fn.scope() != at::RecordScope::BACKWARD_FUNCTION)) {
              data_ptr->module_hierarchy = torch::jit::currentModuleHierarchy();
            }
            return ctx_ptr;
          },
          [](const at::RecordFunction &fn, at::ObserverContext *ctx_ptr) {
            auto state_ptr = NpuProfilerThreadLocalState::getTLS();
            if (!state_ptr) {
              return;
            }
            auto *npu_ctx_ptr = static_cast<NpuObserverContext *>(ctx_ptr);
            TORCH_INTERNAL_ASSERT(npu_ctx_ptr != nullptr);
            auto data_ptr = std::move(npu_ctx_ptr->data_);
            data_ptr->end_ns = static_cast<int64_t>(Utils::GetClockTime());
            data_ptr->end_thread_id = Utils::GetTid();
            if (ProfilerMgr::GetInstance()->ReportEnable().load(std::memory_order_relaxed)) {
              ProfilerMgr::GetInstance()->Upload(std::move(data_ptr));
            }
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
  if (c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE) != nullptr) {
    ASCEND_LOGE("Profiler is already enabled.");
    return;
  }
  c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);
  bool cpu_trace = activities.count(NpuActivityType::CPU);
  ExperimentalConfig experimental_config = config.experimental_config;
  NpuTraceConfig npu_config = {experimental_config.trace_level, experimental_config.metrics,
      config.profile_memory, experimental_config.l2_cache, experimental_config.record_op_args};
  ProfilerMgr::GetInstance()->Start(npu_config, cpu_trace);
  if (cpu_trace) {
    registerCallback(scopes);
  }
}

void stopNpuProfiler() {
  auto state = c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE);
  auto state_ptr = static_cast<NpuProfilerThreadLocalState *>(state.get());
  if (state_ptr == nullptr) {
    ASCEND_LOGE("Can't disable Ascend Pytorch Profiler when it's not running.");
    return;
  }
  if (state_ptr->hasCallbackHandle()) {
    at::removeCallback(state_ptr->callbackHandle());
  }
  ProfilerMgr::GetInstance()->Stop();
}

void finalizeNpuProfiler() {
  ProfilerMgr::GetInstance()->Finalize();
}

void reportMarkDataToNpuProfiler(uint32_t category, const std::string &msg, uint64_t correlation_id) {
  ProfilerMgr::GetInstance()->Upload(std::make_unique<torch_npu::toolkit::profiler::OpMarkData>(
    static_cast<int64_t>(Utils::GetClockTime()),
    category,
    correlation_id,
    Utils::GetTid(),
    Utils::GetPid(),
    msg
  ));
}
} // profiler
} // torch_npu
