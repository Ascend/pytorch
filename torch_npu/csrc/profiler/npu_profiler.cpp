#include <c10/util/Exception.h>
#include <torch/csrc/profiler/util.h>
#include <torch/csrc/profiler/api.h>
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
using torch_npu::toolkit::profiler::OpRangeData;
using torch_npu::toolkit::profiler::TensorMetadata;
using torch::autograd::profiler::ProfilerConfig;
using torch::autograd::profiler::ProfilerState;
using torch::profiler::impl::ProfilerStateBase;
using torch::profiler::impl::ActiveProfilerType;

struct NpuObserverContext : public at::ObserverContext {
    explicit NpuObserverContext(std::unique_ptr<torch_npu::toolkit::profiler::OpRangeData> data) : data_(std::move(data)) {}
    std::unique_ptr<torch_npu::toolkit::profiler::OpRangeData> data_;
};

struct NpuProfilerThreadLocalState : public ProfilerStateBase {
    explicit NpuProfilerThreadLocalState(
        const NpuProfilerConfig &config,
        std::set<NpuActivityType> activities)
        : ProfilerStateBase(ProfilerConfig(ProfilerState::CPU)), npu_config_(config), activities_(std::move(activities))
    {
        // copy value from NpuProfilerConfig to ProfilerConfig for compatibility
        config_.report_input_shapes = config.record_shapes;
        config_.profile_memory = config.profile_memory;
        config_.with_stack = config.with_stack;
        config_.with_flops = config.with_flops;
        config_.with_modules = config.with_modules;
    }
    ~NpuProfilerThreadLocalState() override = default;

    static NpuProfilerThreadLocalState *getTLS()
    {
        return static_cast<NpuProfilerThreadLocalState *>(
            c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE)
        );
    }

    const std::set<NpuActivityType> &activities() const
    {
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

    bool memoryProfilingEnabled() const
    {
        return config_.profile_memory;
    }

    bool tracePython()
    {
        return (config_.with_stack || config_.with_modules) && activities_.count(NpuActivityType::CPU);
    }

    void setCallbackHandle(at::CallbackHandle handle)
    {
        handle_ = handle;
    }

    at::CallbackHandle callbackHandle() const
    {
        return handle_;
    }

    bool hasCallbackHandle()
    {
        return handle_ > 0;
    }

    // Only CPU
    void reportMemoryUsage(
        void *ptr,
        int64_t alloc_size,
        size_t total_allocated,
        size_t total_reserved,
        c10::Device device)
    {
        if (config_.profile_memory && ProfilerMgr::GetInstance()->ReportEnable().load(std::memory_order_relaxed)) {
            ProfilerMgr::GetInstance()->UploadWithLock(std::make_unique<torch_npu::toolkit::profiler::MemoryData>(
                reinterpret_cast<int64_t>(ptr),
                static_cast<int64_t>(Utils::GetClockTime()),
                alloc_size,
                static_cast<int64_t>(total_allocated),
                static_cast<int64_t>(total_reserved),
                0,
                0,
                static_cast<int8_t>(device.type()),
                device.index(),
                0,
                0,
                0,
                Utils::GetTid(),
                Utils::GetPid()
            ));
        }
    }

    ActiveProfilerType profilerType() override
    {
        return ActiveProfilerType::NONE;
    }

protected:
    NpuProfilerConfig npu_config_;
    std::set<NpuActivityType> activities_;
    at::CallbackHandle handle_ = 0;
};

std::atomic<bool>& profDataReportEnable()
{
    return ProfilerMgr::GetInstance()->ReportEnable();
}

void initNpuProfiler(const std::string &path, const std::set<NpuActivityType> &activities)
{
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
    TORCH_CHECK(!realPath.empty(), "Invalid path", path, PROF_ERROR(ErrCode::PARAM));
    ProfilerMgr::GetInstance()->Init(realPath, npu_trace);
}

static std::string parseTypeName(const c10::IValue &value)
{
    if (value.isBool()) {
        return std::string("Bool");
    } else if (value.isInt()) {
        return std::string("Int");
    } else if (value.isDouble()) {
        return std::string("Double");
    } else if (value.isComplexDouble()) {
        return std::string("Complex");
    }
    return std::string("Others");
}

static bool isValidTensor(const at::Tensor& t)
{
    return t.defined() && !t.is_nested() && !t.unsafeGetTensorImpl()->has_symbolic_sizes_strides();
}

static std::vector<TensorMetadata> getTensorList(const c10::IValue value)
{
    std::vector<TensorMetadata> tensor_list;
    for (const auto& t: value.toTensorList()) {
        if (isValidTensor(t)) {
            tensor_list.emplace_back(t);
        }
    }
    return tensor_list;
}

static void parseInputShapesAndDtypes(const at::RecordFunction &fn,
                                      OpRangeData *data,
                                      bool report_details)
{
    auto inputs = fn.inputs();
    for (const auto &value : inputs) {
        std::vector<int64_t> shape;
        std::string dtype = "None";
        if (value.isTensor()) {
            const at::Tensor &t = value.toTensor();
            if (isValidTensor(t)) {
                dtype = std::string(scalarTypeToTypeMeta(t.scalar_type()).name());
                for (auto i: t.sizes()) {
                    shape.emplace_back(i);
                }
                if (report_details) {
                    data->input_tensors.emplace_back(t);
                }
            }
        } else if (value.isTensorList()) {
            dtype = "TensorList";
            if (report_details) {
                data->input_tensorlists.emplace_back(std::move(getTensorList(value)));
            }
        } else if (value.isScalar()) {
            dtype = "Scalar";
            if (report_details) {
                data->input_scalars.emplace_back(std::move(parseTypeName(value)));
            }
        } else if (value.isList()) {
            auto listRef = value.toListRef();
            if (listRef.empty() || listRef[0].isScalar()) {
                dtype = "ScalarList";
            }
        }
        data->input_dtypes.emplace_back(std::move(dtype));
        data->input_shapes.emplace_back(std::move(shape));
    }
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
                data_ptr->scope = static_cast<uint8_t>(fn.scope());
                if ((C10_UNLIKELY(config.report_input_shapes))) {
                    bool report_details = config.report_input_shapes
                                          && (config.with_stack || config.with_modules)
                                          && config.profile_memory;
                    parseInputShapesAndDtypes(fn, data_ptr.get(), report_details);
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
                TORCH_INTERNAL_ASSERT(npu_ctx_ptr != nullptr, PROF_ERROR(ErrCode::PTR));
                auto data_ptr = std::move(npu_ctx_ptr->data_);
                data_ptr->end_ns = static_cast<int64_t>(Utils::GetClockTime());
                data_ptr->end_thread_id = Utils::GetTid();
                if (ProfilerMgr::GetInstance()->ReportEnable().load(std::memory_order_relaxed)) {
                    ProfilerMgr::GetInstance()->Upload(std::move(data_ptr));
                }
            }
        )
        .needsInputs(registeration_state_ptr->config().report_input_shapes)
        .scopes(scopes)
    );
    registeration_state_ptr->setCallbackHandle(handle);
}

void warmupNpuProfiler(const NpuProfilerConfig &config,
    const std::set<NpuActivityType> &activities)
{
    bool cpu_trace = activities.count(NpuActivityType::CPU);
    ExperimentalConfig experimental_config = config.experimental_config;
    NpuTraceConfig npu_config = {experimental_config.trace_level, experimental_config.metrics,
        config.profile_memory, experimental_config.l2_cache, experimental_config.record_op_args,
        experimental_config.msprof_tx, experimental_config.op_attr, experimental_config.host_sys, experimental_config.mstx_domain_include,
        experimental_config.mstx_domain_exclude, experimental_config.sys_io, experimental_config.sys_interconnection};
    ProfilerMgr::GetInstance()->Warmup(npu_config, cpu_trace);
}

void startNpuProfiler(const NpuProfilerConfig &config,
    const std::set<NpuActivityType> &activities,
    const std::unordered_set<at::RecordScope> &scopes)
{
    auto state = std::make_shared<NpuProfilerThreadLocalState>(config, activities);
    if (c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE) != nullptr) {
        ASCEND_LOGE("Profiler is already enabled.");
        return;
    }
    c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);
    bool cpu_trace = activities.count(NpuActivityType::CPU);
    ExperimentalConfig experimental_config = config.experimental_config;
    NpuTraceConfig npu_config = {experimental_config.trace_level, experimental_config.metrics,
        config.profile_memory, experimental_config.l2_cache, experimental_config.record_op_args,
        experimental_config.msprof_tx, experimental_config.op_attr, experimental_config.host_sys, experimental_config.mstx_domain_include,
        experimental_config.mstx_domain_exclude, experimental_config.sys_io, experimental_config.sys_interconnection};
    ProfilerMgr::GetInstance()->Start(npu_config, cpu_trace);
    if (state->tracePython()) {
        python_tracer::call(python_tracer::Command::kStartAll);
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
        ASCEND_LOGE("Can't disable Ascend Pytorch Profiler when it's not running.");
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

void finalizeNpuProfiler()
{
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
        data.component_type,
        data.data_type,
        data.allocator_type,
        Utils::GetTid(),
        Utils::GetPid()
    ));
}
} // profiler
} // torch_npu
