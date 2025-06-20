#include "torch_npu/csrc/profiler/containers.h"
#include "torch_npu/csrc/profiler/profiler_python.h"

#include <deque>
#include <memory>
#include <unordered_map>
#include <utility>

#include <Python.h>
#include <frameobject.h>

#include "torch_npu/csrc/profiler/npu_profiler.h"
#include "torch_npu/csrc/profiler/profiler_mgr.h"
#include "torch_npu/csrc/profiler/utils.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
#include "torch_npu/csrc/toolkit/profiler/inc/data_reporter.h"

#include <c10/util/hash.h>
#include <ATen/record_function.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/profiler/api.h>

namespace torch_npu {
namespace profiler {
namespace python_tracer {

const std::string EXIT_EVENT_DESC = "__torch_npu_profiler_python_tracer_exit";  // Special hash value for exit event
const size_t EXIT_EVENT_HASH_ID = c10::get_hash(EXIT_EVENT_DESC);               // Special hash key for exit event
const std::string MODULE_NAME_DELIMITER = "######";
constexpr size_t TRACE_DUMP_THRESHOLD = 1024 * DEFAULT_BLOCK_SIZE;

using TensorMetadata = torch_npu::toolkit::profiler::TensorMetadata;
using ModuleParam = torch_npu::toolkit::profiler::ModuleParam;
using OptimizerParam = torch_npu::toolkit::profiler::OptimizerParam;

std::vector<PyThreadState*> getInterpreterThreads(PyInterpreterState* interpreter)
{
    pybind11::gil_scoped_acquire gil;
    std::vector<PyThreadState*> threads;
    if (interpreter != nullptr) {
        auto* thread_state = PyInterpreterState_ThreadHead(interpreter);
        while (thread_state != nullptr) {
            threads.push_back(thread_state);
            thread_state = PyThreadState_Next(thread_state);
        }
    }
    return threads;
}

class GilAndRestoreThread {
public:
    GilAndRestoreThread() : gil_(), initial_thread_state_{PyThreadState_Get()} {}
    ~GilAndRestoreThread()
    {
        PyThreadState_Swap(initial_thread_state_);
        if (!Py_IsInitialized()) {
            gil_.disarm();
        }
    }

    PyThreadState* initial_thread_state() const
    {
        return initial_thread_state_;
    }

private:
    pybind11::gil_scoped_acquire gil_;
    PyThreadState* initial_thread_state_{nullptr};
};

struct ThreadLocalResult;
struct TraceContext {
    PyObject_HEAD
    ThreadLocalResult* thread_local_result_;
};

static PyTypeObject TraceContextType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "TraceContext",             /* tp_name */
    sizeof(TraceContext),       /* tp_basicsize */
    0,                          /* tp_itemsize */
    nullptr,                    /* tp_dealloc */
    0,                          /* tp_vectorcall_offset */  // NOLINT: modernize-use-nullptr
    nullptr,                    /* tp_getattr */
    nullptr,                    /* tp_setattr */
    nullptr,                    /* tp_reserved */
    nullptr,                    /* tp_repr */
    nullptr,                    /* tp_as_number */
    nullptr,                    /* tp_as_sequence */
    nullptr,                    /* tp_as_mapping */
    nullptr,                    /* tp_hash  */
    nullptr,                    /* tp_call */
    nullptr,                    /* tp_str */
    nullptr,                    /* tp_getattro */
    nullptr,                    /* tp_setattro */
    nullptr,                    /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,         /* tp_flags */
    "Python tracer TLS",        /* tp_doc */
    nullptr,                    /* tp_traverse */
    nullptr,                    /* tp_clear */
    nullptr,                    /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    nullptr,                    /* tp_iter */
    nullptr,                    /* tp_iternext */
    nullptr,                    /* tp_methods */
    nullptr,                    /* tp_members */
    nullptr,                    /* tp_getset */
    nullptr,                    /* tp_base */
    nullptr,                    /* tp_dict */
    nullptr,                    /* tp_descr_get */
    nullptr,                    /* tp_descr_set */
    0,                          /* tp_dictoffset */
    nullptr,                    /* tp_init */
    nullptr,                    /* tp_alloc */
    PyType_GenericNew,          /* tp_new */
    nullptr                     /* tp_free */
};

class PythonTracer;
struct ThreadLocalResult {
    explicit ThreadLocalResult(PythonTracer* active_tracer)
        : active_tracer_(active_tracer),
          ctx_((TraceContext*) TraceContextType.tp_alloc(&TraceContextType, 0))
    {
        if (ctx_) {
            ctx_->thread_local_result_ = this;
        }
    }

    ThreadLocalResult() = delete;
    ThreadLocalResult(const ThreadLocalResult&) = delete;
    ThreadLocalResult(ThreadLocalResult&&) = delete;
    ThreadLocalResult& operator=(const ThreadLocalResult&) = delete;
    ThreadLocalResult& operator=(const ThreadLocalResult&&) = delete;

    ~ThreadLocalResult()
    {
        if (ctx_) {
            Py_DECREF((PyObject*)ctx_);
        }
        active_tracer_ = nullptr;
    }

    PythonTracer* active_tracer_{nullptr};
    TraceContext* ctx_{nullptr};
};

struct PyCallInfo {
    PyCallInfo() = default;
    explicit PyCallInfo(PyFrameObject* frame) : line_no_(PyFrame_GetLineNumber(frame))
    {
        auto f_code = PyFrame_GetCode_NPU(frame);
        file_name_ = THPUtils_unpackStringView(f_code->co_filename).data();
        func_name_ = THPUtils_unpackStringView(f_code->co_name).data();
    }

    size_t get_hash_id()
    {
        return c10::get_hash(line_no_, file_name_, func_name_);
    }

    std::string get_name()
    {
        std::stringstream name_stream;
        name_stream << file_name_ << "(" << line_no_ << "): " << func_name_;
        return name_stream.str();
    }

    int line_no_{0};
    const char* file_name_{nullptr};
    const char* func_name_{nullptr};
};

struct ModuleInfo {
    ModuleInfo() = default;
    explicit ModuleInfo(PyObject* module_class) : moudle_id_(reinterpret_cast<uintptr_t>(module_class))
    {
        auto py_class_name = py::handle(module_class).attr("__class__").attr("__name__");
        module_name_ = at::StringView(py::str(py_class_name));
    }

    std::string get_name()
    {
        std::stringstream name_stream;
        name_stream << module_name_ << MODULE_NAME_DELIMITER << moudle_id_;
        return name_stream.str();
    }

    uintptr_t moudle_id_{0};
    at::StringView module_name_;
};

constexpr size_t max_py_threads = std::numeric_limits<uint8_t>::max() + 1;

class PythonTracer final {
public:
    static void call(Command c);
    static int pyProfileFn(
        PyObject* obj,
        PyFrameObject* frame,
        int what,
        PyObject* arg);

    struct StartPyCall {
        explicit StartPyCall(size_t key)
            : hash_id_(static_cast<uint64_t>(key)), ts_(torch_npu::toolkit::profiler::Utils::GetClockTime()) {}

        uint64_t hash_id_{0};
        uint64_t ts_{0};
    };

private:
    PythonTracer();
    static PythonTracer& singleton();

    void start(size_t max_threads = max_py_threads);
    void stop();
    void clear();
    size_t genPyCallHashId(PyFrameObject* frame);
    void recordPyCall(TraceContext* ctx, PyFrameObject* frame);
    void recordCCall(TraceContext* ctx, PyFrameObject* frame, PyObject* arg);
    void recordReturn(TraceContext* ctx, PyFrameObject* frame, TraceTag tag);
    void recordEvent(TraceTag tag, size_t hash_key);
    void reportTraceData();
    void reportHashData();
    void reportParamData();
    std::string trimPrefix(std::string s);

private:
    std::atomic<bool> active_{false};
    bool record_params_{false};
    PyInterpreterState* interpreter_{nullptr};
    std::deque<ThreadLocalResult> thread_local_results_;
    PyObject* module_call_code_{nullptr};
    PyObject* optimizer_call_code_{nullptr};
    std::vector<std::string> func_name_prefixes_;
    std::unordered_map<size_t, PyCallInfo> py_call_cache_;
    std::unordered_map<size_t, at::StringView> pyc_call_cache_;
    std::unordered_map<size_t, ModuleInfo> module_info_cache_;
    std::vector<std::pair<size_t, std::vector<ModuleParam>>> module_param_cache_;
    std::vector<std::pair<size_t, std::vector<OptimizerParam>>> optimizer_param_cache_;
    AppendOnlyList<TraceEvent> events_;
    std::unordered_map<uintptr_t, std::vector<StartPyCall>> start_py_call_info_;
    std::unordered_map<uintptr_t, uint64_t> ctx_tid_map_;
};

PythonTracer& PythonTracer::singleton()
{
    static PythonTracer singleton_;
    return singleton_;
}

PythonTracer::PythonTracer() : active_(false)
{
    pybind11::gil_scoped_acquire gil;
    module_call_code_ = py::module::import("torch.nn")
        .attr("Module")
        .attr("__call__")
        .attr("__code__")
        .ptr();
    optimizer_call_code_ = py::module::import("torch.optim")
        .attr("Optimizer")
        .attr("_optimizer_step_code")
        .attr("__code__")
        .ptr();
    func_name_prefixes_ = py::module::import("torch.profiler.python_tracer")
        .attr("_prefix_regex")()
        .cast<std::vector<std::string>>();
}

void PythonTracer::start(size_t max_threads)
{
    TORCH_CHECK(thread_local_results_.empty(), "PythonTracer should not have active contexts", PROF_ERROR(ErrCode::INTERNAL));
    TORCH_CHECK(max_threads > 0, "max_threads must be positive, got ", max_threads, PROF_ERROR(ErrCode::VALUE));
    TORCH_CHECK(max_threads <= max_py_threads, "max_threads must be less equal to ", max_py_threads, PROF_ERROR(ErrCode::VALUE));

    bool expected{false};
    bool active = active_.compare_exchange_strong(expected, true);
    if (!active) {
        ASCEND_LOGW("There is already an active PythonTracer. Refusing to register profile functions.");
        return;
    }

    GilAndRestoreThread gil;
    interpreter_ = PyInterpreterState_Get();
    if (!gil.initial_thread_state()) {
        ASCEND_LOGW("Failed to get main thread state, PythonTracer will not start.");
        return;
    }

    std::vector<PyThreadState*> thread_states = getInterpreterThreads(interpreter_);
    if (thread_states.empty()) {
        ASCEND_LOGW("There is no active thread, PythonTracer will not start.")
        return;
    }
    if (thread_states.size() > max_threads) {
        ASCEND_LOGW("Can only trace %zu thread. %zu are currently active.", max_threads, thread_states.size());
        thread_states.resize(max_threads);
    }

    const size_t STACK_MAX_DEPTH = 128;
    // Register the tracer in each thread.
    for (const auto thread_state : thread_states) {
        PyThreadState_Swap(thread_state);
        thread_local_results_.emplace_back(this);
        auto* ctx = thread_local_results_.back().ctx_;

        std::vector<THPFrameObjectPtr> current_stack;
        auto frame = PyEval_GetFrame_NPU();
        size_t depth = 0;  // Make sure we can't infinite loop.
        while (frame != nullptr && depth <= STACK_MAX_DEPTH) {
            current_stack.emplace_back(frame);
            frame = PyFrame_GetBack(frame);
            ++depth;
        }
        // record py call before proflier start
        for (auto it = current_stack.rbegin(); it != current_stack.rend(); it++) {
            start_py_call_info_[reinterpret_cast<uintptr_t>(ctx)].emplace_back(genPyCallHashId(*it));
        }
        PyEval_SetProfile(PythonTracer::pyProfileFn, (PyObject*)ctx);
    }

    auto config = torch::autograd::profiler::getProfilerConfig();
    if (config.report_input_shapes
        && (config.with_stack || config.with_modules)
        && config.profile_memory) {
        record_params_ = true;
    }
}

void PythonTracer::stop()
{
    TORCH_INTERNAL_ASSERT(active_.load(), "PythonTracer is not running.", PROF_ERROR(ErrCode::INTERNAL));

    GilAndRestoreThread gil;
    for (const auto thread_state : getInterpreterThreads(interpreter_)) {
        if (thread_state->c_profilefunc == &PythonTracer::pyProfileFn) {
            PyThreadState_Swap(thread_state);
            PyEval_SetProfile(nullptr, nullptr);
        }
    }
    for (const auto& start_py_call : start_py_call_info_) {
        auto ctx_tid = ctx_tid_map_.find(start_py_call.first);
        if (ctx_tid != ctx_tid_map_.end()) {
            for (const auto& py_call : start_py_call.second) {
                events_.emplace_back(
                    ctx_tid->second,
                    py_call.ts_,
                    py_call.hash_id_,
                    TraceTag::kPy_Call);
            }
        }
    }
    active_ = false;
    reportTraceData();
    reportHashData();
    reportParamData();
}

void PythonTracer::clear()
{
    TORCH_CHECK(!active_.load(), "Cannot clear state while PythonTracer is active.", PROF_ERROR(ErrCode::INTERNAL));
    py_call_cache_.clear();
    pyc_call_cache_.clear();
    module_info_cache_.clear();
    module_param_cache_.clear();
    optimizer_param_cache_.clear();
    events_.clear();
    ctx_tid_map_.clear();
    start_py_call_info_.clear();
    thread_local_results_.clear();
    interpreter_ = nullptr;
}

std::string PythonTracer::trimPrefix(std::string s)
{
    for (const auto& p : func_name_prefixes_) {
        if (s.compare(0, p.size(), p) == 0) {
            s.erase(0, p.size());
            return s;
        }
    }
    return s;
}

void PythonTracer::reportTraceData()
{
    if (events_.size() > 0) {
        ProfilerMgr::GetInstance()->UploadTraceEventData(
            std::make_unique<torch_npu::toolkit::profiler::PythonTracerFuncData>(
                torch_npu::toolkit::profiler::Utils::GetPid(),
                std::move(events_)
            )
        );
        events_.clear();
    }
}

void PythonTracer::reportHashData()
{
    std::vector<std::pair<uint64_t, std::string>> hash_data;
    hash_data.resize(py_call_cache_.size() + pyc_call_cache_.size() + module_info_cache_.size() + 1);
    size_t idx = 0;
    for (auto& item : py_call_cache_) {
        hash_data[idx++] = std::make_pair(item.first, trimPrefix(std::move(item.second.get_name())));
    }
    for (auto& item : pyc_call_cache_) {
        hash_data[idx++] = std::make_pair(item.first, std::string(item.second.str()));
    }
    for (auto& item : module_info_cache_) {
        hash_data[idx++] = std::make_pair(item.first, item.second.get_name());
    }
    hash_data[idx] = std::make_pair(EXIT_EVENT_HASH_ID, EXIT_EVENT_DESC);

    ProfilerMgr::GetInstance()->UploadTraceHashData(
        std::make_unique<torch_npu::toolkit::profiler::PythonTracerHashData>(
            hash_data
        )
    );
}

void PythonTracer::reportParamData()
{
    if (module_param_cache_.size() > 0 || optimizer_param_cache_.size() > 0) {
        ProfilerMgr::GetInstance()->UploadParamData(
            std::make_unique<torch_npu::toolkit::profiler::ParamTensorData>(
                std::move(module_param_cache_),
                std::move(optimizer_param_cache_)
            )
        );
    }
    module_param_cache_.clear();
    optimizer_param_cache_.clear();
}

void PythonTracer::recordEvent(TraceTag tag, size_t hash_key)
{
    events_.emplace_back(
        torch_npu::toolkit::profiler::Utils::GetTid(),
        torch_npu::toolkit::profiler::Utils::GetClockTime(),
        hash_key,
        tag);
    if (events_.size() >= TRACE_DUMP_THRESHOLD) {
        reportTraceData();
    }
}

static TensorMetadata toTensorMetadata(PyObject* self)
{
    if (!THPVariable_CheckExact(self)) {
        TensorMetadata m;
        return m;
    }
    const auto& t = THPVariable_Unpack(self);
    TensorMetadata m{t};
    return m;
}

static c10::optional<TensorMetadata> recordIfTensor(py::handle p)
{
    return THPVariable_CheckExact(p.ptr())
        ? c10::optional<TensorMetadata>(toTensorMetadata(p.ptr()))
        : c10::nullopt;
}

static std::vector<std::pair<std::string, TensorMetadata>> unpackTensorMap(const py::dict& tensor_map)
{
    std::vector<std::pair<std::string, TensorMetadata>> out;
    for (auto& it : tensor_map) {
        auto* value = it.second.ptr();
        if (py::isinstance<py::str>(it.first) && THPVariable_CheckExact(value)) {
            out.emplace_back(py::cast<std::string>(it.first), toTensorMetadata(value));
        }
    }
    return out;
}

static void parseModuleParams(
    std::vector<std::pair<size_t, std::vector<ModuleParam>>> &module_param_cache,
    PyObject* cls,
    size_t hash_id)
{
    std::vector<ModuleParam> module_params;
    py::dict params = py::handle(cls).attr("_parameters");
    for (auto& it : params) {
        auto* p = it.second.ptr();
        if (py::isinstance<py::str>(it.first) && THPVariable_CheckExact(p)) {
            module_params.emplace_back(
                ModuleParam{
                    it.first.cast<std::string>(),
                    toTensorMetadata(p),
                    recordIfTensor(py::getattr(it.second, "grad", py::none()))});
        }
    }
    if (module_params.size() > 0) {
        module_param_cache.emplace_back(std::move(std::make_pair(hash_id, module_params)));
    }
}

static void parseOptimizerParams(
    std::vector<std::pair<size_t, std::vector<OptimizerParam>>> &optimizer_param_cache,
    PyObject* cls,
    size_t hash_id)
{
    std::vector<OptimizerParam> optimizer_params;
    const py::handle self{cls};
    for (const auto& it : (py::list)self.attr("param_groups")) {
        for (auto& param : py::cast<py::dict>(it).attr("get")("params")) {
            if (THPVariable_CheckExact(param.ptr())) {
                optimizer_params.emplace_back(
                    OptimizerParam{
                        toTensorMetadata(param.ptr()),
                        recordIfTensor(py::getattr(param, "grad", py::none())),
                        unpackTensorMap(py::cast<py::dict>(self.attr("state")).attr("get")(param, py::dict()))});
            }
        }
    }
    if (optimizer_params.size() > 0) {
        optimizer_param_cache.emplace_back(std::move(std::make_pair(hash_id, optimizer_params)));
    }
}

size_t PythonTracer::genPyCallHashId(PyFrameObject* frame)
{
    TORCH_INTERNAL_ASSERT(frame != nullptr, "frame can not be nullptr.", PTA_ERROR(ErrCode::PARAM));

    auto call_info = PyCallInfo(frame);
    auto hash_id = call_info.get_hash_id();
    auto f_code = PyFrame_GetCode_NPU(frame);
    if (py_call_cache_.find(hash_id) == py_call_cache_.end()) {
        py_call_cache_.insert({hash_id, call_info});

        // check optim.Optimizer call
        if (record_params_ && ((PyObject*)f_code.get() == optimizer_call_code_)) {
            auto f_locals = PyFrame_GetLocals_NPU(frame);
            auto optimizer_class = PyDict_GetItemString(f_locals, "self");
            parseOptimizerParams(optimizer_param_cache_, (PyObject*)optimizer_class, hash_id);
        }
    }

    // check nn.Module call
    if ((PyObject*)f_code.get() == module_call_code_) {
        auto f_locals = PyFrame_GetLocals_NPU(frame);
        auto module_class = PyDict_GetItemString(f_locals, "self");
        hash_id = c10::get_hash(module_class);
        if (module_info_cache_.find(hash_id) == module_info_cache_.end()) {
            module_info_cache_.insert({hash_id, ModuleInfo(module_class)});

            if (record_params_) {
                parseModuleParams(module_param_cache_, (PyObject*)module_class, hash_id);
            }
        }
    }
    return hash_id;
}

void PythonTracer::recordPyCall(TraceContext* ctx, PyFrameObject* frame)
{
    auto hash_id = genPyCallHashId(frame);
    recordEvent(TraceTag::kPy_Call, hash_id);
}

void PythonTracer::recordCCall(TraceContext* ctx, PyFrameObject* frame, PyObject* arg)
{
    std::string call_info = py::repr(arg);
    auto hash_id = c10::get_hash(call_info);
    if (pyc_call_cache_.find(hash_id) == pyc_call_cache_.end()) {
        pyc_call_cache_.insert({hash_id, at::StringView(std::move(call_info))});
    }
    recordEvent(TraceTag::kC_Call, hash_id);
}

void PythonTracer::recordReturn(TraceContext* ctx, PyFrameObject* frame, TraceTag tag)
{
    recordEvent(tag, EXIT_EVENT_HASH_ID);

    // record ctx to thread id map
    auto ctx_addr = reinterpret_cast<uintptr_t>(ctx);
    if (ctx_tid_map_.find(ctx_addr) == ctx_tid_map_.end()) {
        ctx_tid_map_.insert({ctx_addr, torch_npu::toolkit::profiler::Utils::GetTid()});
    }
}

int PythonTracer::pyProfileFn(
    PyObject* obj,
    PyFrameObject* frame,
    int what,
    PyObject* arg)
{
    auto ctx = reinterpret_cast<TraceContext*>(obj);
    auto thread_local_result = ctx->thread_local_result_;
    switch (what) {
        case PyTrace_CALL:
            thread_local_result->active_tracer_->recordPyCall(ctx, frame);
            break;

        case PyTrace_C_CALL:
            thread_local_result->active_tracer_->recordCCall(ctx, frame, arg);
            break;

        case PyTrace_EXCEPTION:
        case PyTrace_RETURN:
            thread_local_result->active_tracer_->recordReturn(ctx, frame, TraceTag::kPy_Return);
            break;

        case PyTrace_C_EXCEPTION:
        case PyTrace_C_RETURN:
            thread_local_result->active_tracer_->recordReturn(ctx, frame, TraceTag::kC_Return);
            break;

        default:
            break;
    }
    return 0;
}

void PythonTracer::call(Command c)
{
    switch (c) {
        case Command::kStartOne:
            PythonTracer::singleton().start(1);
            break;

        case Command::kStartAll:
            PythonTracer::singleton().start();
            break;

        case Command::kStop:
            PythonTracer::singleton().stop();
            break;

        case Command::kClear:
            PythonTracer::singleton().clear();
            break;

        default:
            break;
    }
};

void init()
{
    pybind11::gil_scoped_acquire gil;
    TORCH_CHECK(PyType_Ready(&TraceContextType) == 0, PROF_ERROR(ErrCode::INTERNAL));
    registerFunctions(
        &PythonTracer::call
    );
}
} // python_tracer
} // profiler
} // torch_npu
