#include "torch_npu/csrc/profiler/profiler_python.h"

#include <memory>

#include <Python.h>
#include <frameobject.h>

#include "torch_npu/csrc/profiler/npu_profiler.h"
#include "torch_npu/csrc/profiler/profiler_mgr.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"

#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/pybind.h>

namespace torch_npu {
namespace profiler {
namespace python_tracer {

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

enum class TraceTag {
    kPy_Call = 0,
    kPy_Return,
    kC_Call,
    kC_Return
};

struct RawEvent {
    RawEvent(TraceTag tag, PyFrameObject* frame)
        : tag_(tag),
          frame_(frame),
          t_(torch_npu::toolkit::profiler::Utils::GetClockTime()),
          misc_() {}

    RawEvent(TraceTag tag, PyFrameObject* frame, PyObject* arg)
        : RawEvent(tag, frame)
    {
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tag == TraceTag::kC_Call);
        misc_.arg_ = arg;
    }

    TraceTag tag_{};
    PyFrameObject* frame_{nullptr};
    uint64_t t_{0};
    union {
        PyObject* arg_;  // kC_Call
        void* null_; // Unused (placeholder), kPy_Call, kPy_Return, kC_Return
    } misc_{};

    uint8_t tag() const
    {
        return static_cast<uint8_t>(tag_);
    }

    std::string get_func_name() const
    {
        if (tag_ == TraceTag::kC_Call) {
            return py::repr(misc_.arg_);
        } else if (tag_ == TraceTag::kPy_Call) {
            auto f_code = frame_->f_code;
            auto line_no = PyFrame_GetLineNumber(frame_);
            auto file_name = trimPrefix(THPUtils_unpackString(f_code->co_filename));
            auto func_name = THPUtils_unpackString(f_code->co_name);
            std::stringstream name_stream;
            name_stream << file_name << "(" << line_no << "): " << func_name;
            return name_stream.str();
        }
        return "";
    }
};

void reportPythonFuncCallDataToNpuProfiler(const RawEvent& event)
{
    ProfilerMgr::GetInstance()->UploadTraceData(std::make_unique<torch_npu::toolkit::profiler::PythonFuncCallData>(
        event.t_,
        torch_npu::toolkit::profiler::Utils::GetTid(),
        torch_npu::toolkit::profiler::Utils::GetPid(),
        event.tag(),
        event.get_func_name()
    ));
}

void reportPythonModuleCallDataToNpuProfiler(PyObject* mod_class, uint64_t idx)
{
    auto py_class_name = py::handle(mod_class).attr("__class__").attr("__name__");
    std::string module_name = "nn.Module: " + std::string(py::str(py_class_name));
    ProfilerMgr::GetInstance()->UploadTraceData(std::make_unique<torch_npu::toolkit::profiler::PythonModuleCallData>(
        idx,
        torch_npu::toolkit::profiler::Utils::GetTid(),
        torch_npu::toolkit::profiler::Utils::GetPid(),
        std::to_string(reinterpret_cast<uintptr_t>(mod_class)),
        module_name
    ));
}

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
    void trackModule(PyFrameObject* frame);

    bool active_{false};
    int64_t event_count_{0};
    PyObject* module_call_code_{nullptr};
    std::vector<TraceContext*> trace_contexts_;
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
}

void PythonTracer::start(size_t max_threads)
{
    TORCH_CHECK(!active_, "PythonTracer is already active", PROF_ERROR(ErrCode::PARAM))
    TORCH_CHECK(!trace_contexts_.size(), "PythonTracer should not have active contexts", PROF_ERROR(ErrCode::PARAM));
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
        std::vector<PyFrameObject*> current_stack;
        auto frame = PyEval_GetFrame();
        size_t depth = 0;  // Make sure we can't infinite loop.
        while (frame != nullptr && depth <= STACK_MAX_DEPTH) {
            current_stack.push_back(frame);
            frame = frame->f_back;
            ++depth;
        }
        for (auto it = current_stack.rbegin(); it != current_stack.rend(); it++) {
            recordPyCall(ctx, *it);
        }
        PyEval_SetProfile(PythonTracer::pyProfileFn, (PyObject*)ctx);
    }
    PyThreadState_Swap(thread_states[0]);
    active_ = true;
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
}

void PythonTracer::clear()
{
    TORCH_CHECK(!active_, "Cannot clear state while PythonTracer is active.", PROF_ERROR(ErrCode::INTERNAL));
    event_count_ = 0;
    for (auto i : trace_contexts_) {
        Py_DECREF((PyObject*) i);
    }
    trace_contexts_.clear();
}

void PythonTracer::recordPyCall(TraceContext* ctx, PyFrameObject* frame)
{
    ++event_count_;
    trackModule(frame);
    auto event = RawEvent(TraceTag::kPy_Call, frame);
    reportPythonFuncCallDataToNpuProfiler(event);
}

void PythonTracer::recordCCall(TraceContext* ctx, PyFrameObject* frame, PyObject* arg)
{
    ++event_count_;
    auto event = RawEvent(TraceTag::kC_Call, frame, arg);
    reportPythonFuncCallDataToNpuProfiler(event);
}

void PythonTracer::recordReturn(TraceContext* ctx, PyFrameObject* frame, TraceTag tag)
{
    ++event_count_;
    auto event = RawEvent(tag, frame);
    reportPythonFuncCallDataToNpuProfiler(event);
}

void PythonTracer::trackModule(PyFrameObject* frame)
{
    auto f_code = (PyObject*)frame->f_code;
    if (f_code == module_call_code_) {
        PyFrame_FastToLocals(frame);
        auto self = PyDict_GetItemString(frame->f_locals, "self");
        PyFrame_LocalsToFast(frame, 0);
        reportPythonModuleCallDataToNpuProfiler(self, event_count_ - 1);
    }
};

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
