#include "torch_npu/csrc/profiler/containers.h"
#include "torch_npu/csrc/profiler/profiler_python.h"

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

std::string trimPrefix(std::string s)
{
    static std::vector<std::string> prefixes = py::module::import("torch.profiler.python_tracer")
        .attr("_prefix_regex")().cast<std::vector<std::string>>();
    for (const auto& p : prefixes) {
        if (s.compare(0, p.size(), p) == 0) {
            s.erase(0, p.size());
            return s;
        }
    }
    return s;
}

struct TraceContext {
    PyObject_HEAD
    PyThreadState* thread_state_;
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

private:
    PythonTracer();
    static PythonTracer& singleton();

    void start(size_t max_threads = max_py_threads);
    void stop();
    void clear();
    void recordPyCall(TraceContext* ctx, PyFrameObject* frame);
    void recordCCall(TraceContext* ctx, PyFrameObject* frame, PyObject* arg);
    void recordReturn(TraceContext* ctx, PyFrameObject* frame, TraceTag tag);
    void recordEvent(TraceTag tag, size_t hash_key);
    void reportTraceData();
    void reportHashData();
    void reportParamData();

    bool active_{false};
    bool record_params_{false};
    PyObject* module_call_code_{nullptr};
    PyObject* optimizer_call_code_{nullptr};
    std::vector<TraceContext*> trace_contexts_;
    std::unordered_map<size_t, PyCallInfo> py_call_cache_;
    std::unordered_map<size_t, at::StringView> pyc_call_cache_;
    std::unordered_map<size_t, ModuleInfo> module_info_cache_;
    std::vector<std::pair<size_t, std::vector<ModuleParam>>> module_param_cache_;
    std::vector<std::pair<size_t, std::vector<OptimizerParam>>> optimizer_param_cache_;
    AppendOnlyList<TraceEvent> events_;
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
}

void PythonTracer::start(size_t max_threads)
{
    TORCH_CHECK(!active_, "PythonTracer is already active", PROF_ERROR(ErrCode::INTERNAL));
    TORCH_CHECK(!trace_contexts_.size(), "PythonTracer should not have active contexts", PROF_ERROR(ErrCode::INTERNAL));
    TORCH_CHECK(max_threads > 0, "max_threads must be positive, got ", max_threads, PROF_ERROR(ErrCode::VALUE));
    TORCH_CHECK(max_threads <= max_py_threads, "max_threads must be less equal to ", max_py_threads, PROF_ERROR(ErrCode::VALUE));

    pybind11::gil_scoped_acquire gil;

    std::vector<PyThreadState*> thread_states { PyThreadState_Get() };
    if (max_threads > 1) {
        auto thread_state = thread_states[0];
        while (thread_state != nullptr) {
            if (thread_state != thread_states[0]) {
                thread_states.push_back(thread_state);
            }
            thread_state = PyThreadState_Next(thread_state);
        }
        if (thread_states.size() > max_threads) {
            ASCEND_LOGW("Warning: can only trace %zu thread. %zu are currently active.", max_threads, thread_states.size());
            thread_states.resize(max_threads);
        }
    }

    const size_t STACK_MAX_DEPTH = 128;
    for (const auto i : c10::irange(thread_states.size())) {
        PyThreadState* thread_state = thread_states[i];
        PyThreadState_Swap(thread_state);
        auto ctx = (TraceContext*) TraceContextType.tp_alloc(&TraceContextType, 0);
        ctx->thread_state_ = thread_state;
        trace_contexts_.push_back(ctx);

        std::vector<THPFrameObjectPtr> current_stack;
        auto frame = PyEval_GetFrame_NPU();
        size_t depth = 0;  // Make sure we can't infinite loop.
        while (frame != nullptr && depth <= STACK_MAX_DEPTH) {
            current_stack.emplace_back(frame);
            frame = PyFrame_GetBack(frame);
            ++depth;
        }
        for (auto it = current_stack.rbegin(); it != current_stack.rend(); it++) {
            recordPyCall(ctx, *it);
        }
        PyEval_SetProfile(PythonTracer::pyProfileFn, (PyObject*)ctx);
    }
    PyThreadState_Swap(thread_states[0]);
    active_ = true;

    auto config = torch::autograd::profiler::getProfilerConfig();
    if (config.report_input_shapes
        && (config.with_stack || config.with_modules)
        && config.profile_memory) {
        record_params_ = true;
    }
}

void PythonTracer::stop()
{
    TORCH_INTERNAL_ASSERT(active_, "PythonTracer is not running.")

    pybind11::gil_scoped_acquire gil;

    PyThreadState* initial_thread_state = PyThreadState_Get();
    for (const auto i : trace_contexts_) {
        PyThreadState_Swap(i->thread_state_);
        PyEval_SetProfile(nullptr, nullptr);
    }
    PyThreadState_Swap(initial_thread_state);
    active_ = false;
    reportTraceData();
    reportHashData();
    reportParamData();
}

void PythonTracer::clear()
{
    TORCH_CHECK(!active_, "Cannot clear state while PythonTracer is active.", PROF_ERROR(ErrCode::INTERNAL));
    for (auto i : trace_contexts_) {
        Py_DECREF((PyObject*) i);
    }
    trace_contexts_.clear();
    py_call_cache_.clear();
    pyc_call_cache_.clear();
    module_info_cache_.clear();
    module_param_cache_.clear();
    optimizer_param_cache_.clear();
    events_.clear();
}

void PythonTracer::reportTraceData()
{
    if (events_.size() > 0) {
        ProfilerMgr::GetInstance()->UploadTraceEventData(
            std::make_unique<torch_npu::toolkit::profiler::PythonTracerFuncData>(
                torch_npu::toolkit::profiler::Utils::GetTid(),
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
        hash_data[idx++] = std::make_pair(item.first, trimPrefix(item.second.get_name()));
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
    events_.emplace_back(torch_npu::toolkit::profiler::Utils::GetClockTime(), hash_key, tag);
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

void PythonTracer::recordPyCall(TraceContext* ctx, PyFrameObject* frame)
{
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
}

int PythonTracer::pyProfileFn(
    PyObject* obj,
    PyFrameObject* frame,
    int what,
    PyObject* arg)
{
    auto ctx = reinterpret_cast<TraceContext*>(obj);
    switch (what) {
        case PyTrace_CALL:
            PythonTracer::singleton().recordPyCall(ctx, frame);
            break;

        case PyTrace_C_CALL:
            PythonTracer::singleton().recordCCall(ctx, frame, arg);
            break;

        case PyTrace_EXCEPTION:
        case PyTrace_RETURN:
            PythonTracer::singleton().recordReturn(ctx, frame, TraceTag::kPy_Return);
            break;

        case PyTrace_C_EXCEPTION:
        case PyTrace_C_RETURN:
            PythonTracer::singleton().recordReturn(ctx, frame, TraceTag::kC_Return);
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
