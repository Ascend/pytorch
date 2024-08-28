#include <c10/util/Exception.h>
#include <torch/csrc/profiler/util.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/profiler/npu_profiler.h"

#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
#include "torch_npu/csrc/toolkit/profiler/inc/data_reporter.h"
namespace torch_npu {
namespace profiler {
namespace python_tracer {
namespace {
CallFn call_fn;
}

void registerFunctions(CallFn call)
{
    call_fn = call;
}

void call(Command c)
{
    if (call_fn != nullptr) {
        call_fn(c);
    }
}
} // python_tracer
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

    std::unique_ptr<NpuObserverContext> newOpEvent(const at::RecordFunction &fn)
    {
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
    return (config_.with_stack || config_.with_modules) && activities_.count(NpuActivityType::CPU);
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

  // Only CPU
    void reportMemoryUsage(
        void *ptr,
        int64_t alloc_size,
        int64_t total_allocated,
        int64_t total_reserved,
        c10::Device device)
    {
        if (config_.profile_memory && ProfilerMgr::GetInstance()->ReportEnable().load(std::memory_order_relaxed)) {
            ProfilerMgr::GetInstance()->UploadWithLock(std::make_unique<torch_npu::toolkit::profiler::MemoryData>(
                reinterpret_cast<int64_t>(ptr),
                static_cast<int64_t>(Utils::GetClockTime()),
                alloc_size,
                total_allocated,
                total_reserved,
                0,
                0,
                static_cast<int8_t>(device.type()),
                device.index(),
                0,
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

std::atomic<bool>& profDataReportEnable()
{
    return ProfilerMgr::GetInstance()->ReportEnable();
}

void initNpuProfiler(const std::string &path, const std::set<NpuActivityType> &activities) {
  if (path.empty()) {
    return;
  }
  std::string absPath = Utils::RelativeToAbsPath(path);
  if (Utils::IsSoftLink(absPath)) {
    NPU_LOGE("Path %s is soft link.", absPath.c_str());
    return;
  }
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

static void registerCallback(const std::unordered_set<at::RecordScope> &scopes)
{
    auto registeration_state_ptr = NpuProfilerThreadLocalState::getTLS();
    TORCH_INTERNAL_ASSERT(registeration_state_ptr, "Expected profiler state set", PROF_ERROR(ErrCode::PTR));
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
                if (C10_UNLIKELY(config.record_shapes)) {
                    data_ptr->input_dtypes = torch::profiler::impl::inputTypes(fn);
                    data_ptr->input_shapes = torch::profiler::impl::inputSizes(fn);
                }
                if (C10_UNLIKELY(config.with_stack && fn.scope() != at::RecordScope::BACKWARD_FUNCTION)) {
                    auto cs = torch::profiler::impl::prepareCallstack(torch::jit::currentCallstack());
                    cs = cs.empty() ? torch::profiler::impl::prepareCallstack(torch::jit::tracer::pythonCallstack()) : cs;
                    data_ptr->stack = torch::profiler::impl::callstackStr(cs);
                }
                if (C10_UNLIKELY(config.with_modules && fn.scope() != at::RecordScope::BACKWARD_FUNCTION)) {
                    data_ptr->module_hierarchy = torch::jit::currentModuleHierarchy();
                }
                if (C10_UNLIKELY(config.with_flops)) {
                    data_ptr->extra_args = torch::profiler::impl::saveExtraArgs(fn);
                }
                return ctx_ptr;
            },
            [](const at::RecordFunction &fn, at::ObserverContext *ctx_ptr) {
                auto state_ptr = NpuProfilerThreadLocalState::getTLS();
                if (!state_ptr) {
                    return;
                }
                auto *npu_ctx_ptr = static_cast<NpuObserverContext*>(ctx_ptr);
                TORCH_INTERNAL_ASSERT(npu_ctx_ptr != nullptr, PROF_ERROR(ErrCode::PTR));
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
    const std::unordered_set<at::RecordScope> &scopes)
{
    auto state = std::make_shared<NpuProfilerThreadLocalState>(config, activities);
    if (c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE) != nullptr) {
        NPU_LOGE("Profiler is already enabled.");
        return;
    }
    c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);
    bool cpu_trace = activities.count(NpuActivityType::CPU);
    ExperimentalConfig experimental_config = config.experimental_config;
    NpuTraceConfig npu_config = {experimental_config.trace_level, experimental_config.metrics,
        config.profile_memory, experimental_config.l2_cache, experimental_config.record_op_args,
        experimental_config.msprof_tx, experimental_config.op_attr};
    ProfilerMgr::GetInstance()->Start(npu_config, cpu_trace);
    if (state->tracePython()) {
        python_tracer::call(python_tracer::Command::kStartOne);
    }
    if (cpu_trace) {
        registerCallback(scopes);
    }
}

void stopNpuProfiler()
{
    auto state = c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE);
    auto state_ptr = static_cast<NpuProfilerThreadLocalState *>(state.get());
    if (state_ptr == nullptr) {
        NPU_LOGE("Can't disable Ascend Pytorch Profiler when it's not running.");
        return;
    }
    if (state_ptr->hasCallbackHandle()) {
        at::removeCallback(state_ptr->callbackHandle());
    }
    if (state_ptr->tracePython()) {
        python_tracer::call(python_tracer::Command::kStop);
        python_tracer::call(python_tracer::Command::kClear);
    }
    ProfilerMgr::GetInstance()->Stop();
}

void finalizeNpuProfiler() {
  ProfilerMgr::GetInstance()->Finalize();
}

void reportMarkDataToNpuProfiler(uint32_t category, const std::string &msg, uint64_t correlation_id)
{
    ProfilerMgr::GetInstance()->UploadWithLock(std::make_unique<torch_npu::toolkit::profiler::OpMarkData>(
        static_cast<int64_t>(Utils::GetClockTime()),
        category,
        correlation_id,
        Utils::GetTid(),
        Utils::GetPid(),
        msg
    ));
}

void reportMemoryDataToNpuProfiler(const MemoryUsage& data)
{
    if (!ProfilerMgr::GetInstance()->ReportMemEnable().load()) {
        return;
    }
    ProfilerMgr::GetInstance()->UploadWithLock(std::make_unique<torch_npu::toolkit::profiler::MemoryData>(
        data.ptr,
        static_cast<int64_t>(Utils::GetClockTime()),
        data.alloc_size,
        data.total_allocated,
        data.total_reserved,
        data.total_active,
        data.stream_ptr,
        data.device_type,
        data.device_index,
        data.data_type,
        Utils::GetTid(),
        Utils::GetPid()
    ));
}
} // profiler
} // torch_npu
