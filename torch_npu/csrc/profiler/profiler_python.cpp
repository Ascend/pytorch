#include "torch_npu/csrc/profiler/containers.h"
#include "torch_npu/csrc/profiler/profiler_python.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Python.h>
#include <frameobject.h>

#include "torch_npu/csrc/profiler/profiler_callbacks.h"
#include "torch_npu/csrc/profiler/profiler_mgr.h"
#include "torch_npu/csrc/profiler/python/python_frame_utils.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
#include "torch_npu/csrc/toolkit/profiler/inc/data_reporter.h"

#include <c10/util/hash.h>
#include <ATen/record_function.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/profiler/api.h>

namespace torch_npu {
namespace profiler {
namespace python_tracer {

const std::string EXIT_EVENT_DESC = "__torch_npu_profiler_python_tracer_exit"; // Special hash value for exit
                                                                               // event
const size_t EXIT_EVENT_HASH_ID = c10::get_hash(EXIT_EVENT_DESC); // Special hash key for exit event
const std::string MODULE_NAME_DELIMITER = "######";
constexpr size_t TRACE_DUMP_THRESHOLD = 1024 * DEFAULT_BLOCK_SIZE;
constexpr size_t STACK_MAX_DEPTH = 128;

using TensorMetadata = torch_npu::toolkit::profiler::TensorMetadata;
using ModuleParam = torch_npu::toolkit::profiler::ModuleParam;
using OptimizerParam = torch_npu::toolkit::profiler::OptimizerParam;

std::vector<PyThreadState*> getInterpreterThreads(PyInterpreterState* interpreter) {
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

#if defined(Py_GIL_DISABLED) && PY_VERSION_HEX >= 0x030E0000
extern "C" void _PyEval_StopTheWorld(PyInterpreterState*);
extern "C" void _PyEval_StartTheWorld(PyInterpreterState*);
#endif

class StopTheWorldGuard {
 public:
  explicit StopTheWorldGuard([[maybe_unused]] PyInterpreterState* interpreter) {
#if defined(Py_GIL_DISABLED) && PY_VERSION_HEX >= 0x030E0000
    interpreter_ = interpreter;
    _PyEval_StopTheWorld(interpreter_);
#endif
  }
  ~StopTheWorldGuard() {
#if defined(Py_GIL_DISABLED) && PY_VERSION_HEX >= 0x030E0000
    _PyEval_StartTheWorld(interpreter_);
#endif
  }
  StopTheWorldGuard(const StopTheWorldGuard&) = delete;
  StopTheWorldGuard& operator=(const StopTheWorldGuard&) = delete;

 private:
#if defined(Py_GIL_DISABLED) && PY_VERSION_HEX >= 0x030E0000
  PyInterpreterState* interpreter_;
#endif
};

class PythonTracer;
// A single context is passed to every thread. The callback looks up the
// calling thread's caches instead of switching the current thread state.
struct TraceContext {
  PyObject_HEAD
  PythonTracer* tracer_;
};

static PyTypeObject TraceContextType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "TraceContext", /* tp_name */
    sizeof(TraceContext), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0,
    /* tp_vectorcall_offset */ // NOLINT: modernize-use-nullptr
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    "Python tracer context", /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    PyType_GenericNew, /* tp_new */
    nullptr /* tp_free */
};

struct PyCallInfo {
  PyCallInfo() = default;
  explicit PyCallInfo(PyFrameObject* frame) : line_no_(PyFrame_GetLineNumber(frame)) {
    auto f_code = PyFrame_GetCode_NPU(frame);
    file_name_ = THPUtils_unpackStringView(f_code->co_filename).data();
    func_name_ = THPUtils_unpackStringView(f_code->co_name).data();
  }

  size_t get_hash_id() {
    return c10::get_hash(line_no_, file_name_, func_name_);
  }

  std::string get_name() {
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
  explicit ModuleInfo(PyObject* module_class) : moudle_id_(reinterpret_cast<uintptr_t>(module_class)) {
    auto py_class_name = py::handle(module_class).attr("__class__").attr("__name__");
    module_name_ = at::StringView(py::str(py_class_name));
  }

  std::string get_name() {
    std::stringstream name_stream;
    name_stream << module_name_ << MODULE_NAME_DELIMITER << moudle_id_;
    return name_stream.str();
  }

  uintptr_t moudle_id_{0};
  at::StringView module_name_;
};

struct ThreadLocalResult {
  explicit ThreadLocalResult(const TraceContext* context) : context_(context) {}

  struct StartPyCall {
    explicit StartPyCall(size_t key)
        : hash_id_(static_cast<uint64_t>(key)), ts_(torch_npu::toolkit::profiler::Utils::GetClockTime()) {}

    uint64_t hash_id_{0};
    uint64_t ts_{0};
  };

  // Non-owning session identity; shared_ctx_ owns the Python reference.
  const TraceContext* context_;
  uint64_t tid_{0};
  bool enabled_{true};
  // Also lets stop() wait for callbacks which temporarily release the GIL.
  std::mutex callback_mutex_;
  std::unordered_map<size_t, PyCallInfo> py_call_cache_;
  std::unordered_map<size_t, at::StringView> pyc_call_cache_;
  std::unordered_map<size_t, ModuleInfo> module_info_cache_;
  std::vector<std::pair<size_t, std::vector<ModuleParam>>> module_param_cache_;
  std::vector<std::pair<size_t, std::vector<OptimizerParam>>> optimizer_param_cache_;
  AppendOnlyList<TraceEvent> events_;
  std::vector<StartPyCall> start_py_call_info_;
};

class PythonTracer final {
 public:
  ~PythonTracer();
  static void call(Command c);
  static int pyProfileFn(PyObject* obj, PyFrameObject* frame, int what, PyObject* arg);

 private:
  PythonTracer();
  static PythonTracer& singleton();
  static PythonTracer* get_singleton_in_child_thread();

  void start();
  void startOne();
  void stop();
  void stopOne();
  void clear();
  void setProfileAllThreads(Py_tracefunc func, PyObject* arg);
  void disableCallbacks();
  void captureStartFrames(ThreadLocalResult& tls, PyThreadState* thread_state);
  std::shared_ptr<ThreadLocalResult> findThreadLocalResult(PyThreadState* thread_state);
  void releaseContext();
  std::unique_lock<std::mutex> lockControl();
  size_t genPyCallHashId(ThreadLocalResult& tls, PyFrameObject* frame);
  void recordPyCall(ThreadLocalResult& tls, PyFrameObject* frame);
  void recordCCall(ThreadLocalResult& tls, PyFrameObject* frame, PyObject* arg);
  void recordReturn(ThreadLocalResult& tls, TraceTag tag);
  void recordEvent(ThreadLocalResult& tls, TraceTag tag, size_t hash_key);
  void reportTraceData(ThreadLocalResult& tls);
  void reportHashData();
  void reportParamData();
  std::string trimPrefix(std::string s);

 private:
  std::atomic<bool> active_{false};
  bool record_params_{false};
  PyInterpreterState* interpreter_{nullptr};
  TraceContext* shared_ctx_{nullptr};
  std::mutex control_mutex_;
#if defined(Py_GIL_DISABLED)
  std::mutex results_mutex_;
#endif
  std::mutex report_mutex_;
  std::vector<std::shared_ptr<ThreadLocalResult>> thread_local_results_;
  std::unordered_map<PyThreadState*, std::shared_ptr<ThreadLocalResult>> thread_local_results_map_;
  PyObject* module_call_code_{nullptr};
  PyObject* optimizer_call_code_{nullptr};
  std::vector<std::string> func_name_prefixes_;
  static std::atomic<bool> instance_created_;
};

std::atomic<bool> torch_npu::profiler::python_tracer::PythonTracer::instance_created_ = false;

PythonTracer& PythonTracer::singleton() {
  static PythonTracer singleton_;
  instance_created_ = true;
  return singleton_;
}

PythonTracer* PythonTracer::get_singleton_in_child_thread() {
  if (instance_created_) {
    return &singleton();
  } else {
    return nullptr;
  }
}

PythonTracer::PythonTracer() : active_(false) {
  pybind11::gil_scoped_acquire gil;
  module_call_code_ = py::module::import("torch.nn").attr("Module").attr("__call__").attr("__code__").ptr();
  optimizer_call_code_ =
      py::module::import("torch.optim").attr("Optimizer").attr("_optimizer_step_code").attr("__code__").ptr();
  func_name_prefixes_ =
      py::module::import("torch.profiler.python_tracer").attr("_prefix_regex")().cast<std::vector<std::string>>();
}

void PythonTracer::start() {
  pybind11::gil_scoped_acquire gil;
  auto control = lockControl();
  TORCH_CHECK(
      thread_local_results_.empty(), "PythonTracer should not have active contexts", PROF_ERROR(ErrCode::INTERNAL));
  bool expected{false};
  bool active = active_.compare_exchange_strong(expected, true);
  if (!active) {
    ASCEND_LOGW("There is already an active PythonTracer. Refusing to register profile functions.");
    return;
  }

#if defined(Py_GIL_DISABLED) && PY_VERSION_HEX < 0x030E0000
  TORCH_WARN("The Python tracer is not thread-safe on Python 3.13t. Please use Python 3.14t or later.");
#endif
  interpreter_ = PyInterpreterState_Get();
  auto config = torch::autograd::profiler::getProfilerConfig();
  record_params_ = config.report_input_shapes && (config.with_stack || config.with_modules) && config.profile_memory;

  shared_ctx_ = reinterpret_cast<TraceContext*>(TraceContextType.tp_alloc(&TraceContextType, 0));
  if (shared_ctx_ == nullptr) {
    active_ = false;
    throw py::error_already_set();
  }
  shared_ctx_->tracer_ = this;

  try {
    // Callbacks ignore threads until their startup frames and caches are ready.
    setProfileAllThreads(PythonTracer::pyProfileFn, reinterpret_cast<PyObject*>(shared_ctx_));
    StopTheWorldGuard stw(interpreter_);
    auto thread_states = getInterpreterThreads(interpreter_);
    for (auto* thread_state : thread_states) {
      auto tls = std::make_shared<ThreadLocalResult>(shared_ctx_);
      captureStartFrames(*tls, thread_state);
#if defined(Py_GIL_DISABLED)
      std::lock_guard<std::mutex> lock(results_mutex_);
#endif
      thread_local_results_.push_back(tls);
      thread_local_results_map_[thread_state] = std::move(tls);
    }
  } catch (...) {
    disableCallbacks();
    {
#if defined(Py_GIL_DISABLED)
      std::lock_guard<std::mutex> lock(results_mutex_);
#endif
      thread_local_results_map_.clear();
      thread_local_results_.clear();
    }
    releaseContext();
    interpreter_ = nullptr;
    record_params_ = false;
    throw;
  }
}

void PythonTracer::startOne() {
  pybind11::gil_scoped_acquire gil;
  auto control = lockControl();
  if (!active_.load()) {
    return;
  }
  auto* thread_state = PyThreadState_Get();
  if (thread_state->c_profilefunc == PythonTracer::pyProfileFn && findThreadLocalResult(thread_state)) {
    return;
  }
  if (thread_state->c_profilefunc != nullptr && thread_state->c_profilefunc != PythonTracer::pyProfileFn) {
    return;
  }
  auto tls = std::make_shared<ThreadLocalResult>(shared_ctx_);
  captureStartFrames(*tls, thread_state);
  {
#if defined(Py_GIL_DISABLED)
    std::lock_guard<std::mutex> lock(results_mutex_);
#endif
    thread_local_results_.push_back(tls);
    thread_local_results_map_[thread_state] = std::move(tls);
  }
  // This API operates on the calling thread; no thread-state swap is needed.
  PyEval_SetProfile(PythonTracer::pyProfileFn, reinterpret_cast<PyObject*>(shared_ctx_));
}

void PythonTracer::stop() {
  pybind11::gil_scoped_acquire gil;
  auto control = lockControl();
  TORCH_INTERNAL_ASSERT(active_.load(), "PythonTracer is not running.", PROF_ERROR(ErrCode::INTERNAL));
  disableCallbacks();
  for (auto& tls : thread_local_results_) {
    if (tls->tid_ != 0) {
      for (const auto& py_call : tls->start_py_call_info_) {
        tls->events_.emplace_back(tls->tid_, py_call.ts_, py_call.hash_id_, TraceTag::kPy_Call);
      }
    }
    reportTraceData(*tls);
  }
  reportHashData();
  reportParamData();
}

void PythonTracer::stopOne() {
  pybind11::gil_scoped_acquire gil;
  auto control = lockControl();
  if (!active_.load()) {
    return;
  }
  auto* thread_state = PyThreadState_Get();
  if (thread_state && thread_state->c_profilefunc == &PythonTracer::pyProfileFn) {
    PyEval_SetProfile(nullptr, nullptr);
  }
}

void PythonTracer::clear() {
  pybind11::gil_scoped_acquire gil;
  auto control = lockControl();
  TORCH_CHECK(!active_.load(), "Cannot clear state while PythonTracer is active.", PROF_ERROR(ErrCode::INTERNAL));
  {
#if defined(Py_GIL_DISABLED)
    std::lock_guard<std::mutex> lock(results_mutex_);
#endif
    thread_local_results_map_.clear();
    thread_local_results_.clear();
  }
  releaseContext();
  record_params_ = false;
  interpreter_ = nullptr;
}

std::unique_lock<std::mutex> PythonTracer::lockControl() {
  std::unique_lock<std::mutex> lock(control_mutex_, std::defer_lock);
  // Control operations may wait for Python callbacks, so never wait for this
  // mutex while preventing another control operation from reacquiring the GIL.
  pybind11::gil_scoped_release release;
  lock.lock();
  return lock;
}

PythonTracer::~PythonTracer() {
  if (Py_IsInitialized() && !Py_IsFinalizing()) {
    if (active_.load()) {
      stop();
    }
    releaseContext();
  }
}

void PythonTracer::releaseContext() {
  if (Py_IsInitialized() && !Py_IsFinalizing()) {
    pybind11::gil_scoped_acquire gil;
    Py_XDECREF(reinterpret_cast<PyObject*>(shared_ctx_));
  }
  shared_ctx_ = nullptr;
}

void PythonTracer::setProfileAllThreads(Py_tracefunc func, PyObject* arg) {
#if PY_VERSION_HEX >= 0x030D0000
  PyEval_SetProfileAllThreads(func, arg);
#else
  // Available on CPython 3.9-3.12. Keep the calling thread's state attached
  // while setting the callback on the target thread.
  for (auto* thread_state : getInterpreterThreads(interpreter_)) {
    if (_PyEval_SetProfile(thread_state, func, arg) < 0) {
      PyErr_WriteUnraisable(nullptr);
    }
  }
#endif
}

void PythonTracer::disableCallbacks() {
  active_ = false;
  setProfileAllThreads(nullptr, nullptr);
  // A callback can release the GIL while expanding metadata. Let it finish
  // before reading its buffers, without preventing it from reacquiring the GIL.
  pybind11::gil_scoped_release release;
  for (auto& tls : thread_local_results_) {
    std::lock_guard<std::mutex> lock(tls->callback_mutex_);
    tls->enabled_ = false;
  }
}

std::shared_ptr<ThreadLocalResult> PythonTracer::findThreadLocalResult(PyThreadState* thread_state) {
#if defined(Py_GIL_DISABLED)
  std::lock_guard<std::mutex> lock(results_mutex_);
#endif
  auto it = thread_local_results_map_.find(thread_state);
  return it == thread_local_results_map_.end() ? nullptr : it->second;
}

void PythonTracer::captureStartFrames(ThreadLocalResult& tls, PyThreadState* thread_state) {
  std::vector<THPFrameObjectPtr> current_stack;
  // PyThreadState_GetFrame and PyFrame_GetBack both return strong references.
  auto frame = PyThreadState_GetFrame(thread_state);
  while (frame != nullptr) {
    current_stack.emplace_back(frame);
    if (current_stack.size() >= STACK_MAX_DEPTH) {
      break;
    }
    frame = PyFrame_GetBack(frame);
  }
  for (auto it = current_stack.rbegin(); it != current_stack.rend(); ++it) {
    tls.start_py_call_info_.emplace_back(genPyCallHashId(tls, it->get()));
  }
}

std::string PythonTracer::trimPrefix(std::string s) {
  for (const auto& p : func_name_prefixes_) {
    if (s.compare(0, p.size(), p) == 0) {
      s.erase(0, p.size());
      return s;
    }
  }
  return s;
}

void PythonTracer::reportTraceData(ThreadLocalResult& tls) {
  if (tls.events_.size() > 0) {
    // Serialize uploads from the per-thread event buffers.
    std::lock_guard<std::mutex> lock(report_mutex_);
    ProfilerMgr::GetInstance()->UploadTraceEventData(
        std::make_unique<torch_npu::toolkit::profiler::PythonTracerFuncData>(
            torch_npu::toolkit::profiler::Utils::GetPid(), std::move(tls.events_)));
    tls.events_.clear();
  }
}

void PythonTracer::reportHashData() {
  std::vector<std::pair<uint64_t, std::string>> hash_data;
  std::unordered_set<uint64_t> reported;
  for (auto& tls : thread_local_results_) {
    for (auto& item : tls->py_call_cache_) {
      if (reported.insert(item.first).second) {
        hash_data.emplace_back(item.first, trimPrefix(item.second.get_name()));
      }
    }
    for (auto& item : tls->pyc_call_cache_) {
      if (reported.insert(item.first).second) {
        hash_data.emplace_back(item.first, std::string(item.second.str()));
      }
    }
    for (auto& item : tls->module_info_cache_) {
      if (reported.insert(item.first).second) {
        hash_data.emplace_back(item.first, item.second.get_name());
      }
    }
  }
  hash_data.emplace_back(EXIT_EVENT_HASH_ID, EXIT_EVENT_DESC);

  ProfilerMgr::GetInstance()->UploadTraceHashData(
      std::make_unique<torch_npu::toolkit::profiler::PythonTracerHashData>(hash_data));
}

void PythonTracer::reportParamData() {
  std::vector<std::pair<size_t, std::vector<ModuleParam>>> module_params;
  std::vector<std::pair<size_t, std::vector<OptimizerParam>>> optimizer_params;
  std::unordered_set<size_t> modules;
  std::unordered_set<size_t> optimizers;
  for (auto& tls : thread_local_results_) {
    for (auto& item : tls->module_param_cache_) {
      if (modules.insert(item.first).second) {
        module_params.push_back(std::move(item));
      }
    }
    for (auto& item : tls->optimizer_param_cache_) {
      if (optimizers.insert(item.first).second) {
        optimizer_params.push_back(std::move(item));
      }
    }
    tls->module_param_cache_.clear();
    tls->optimizer_param_cache_.clear();
  }
  if (!module_params.empty() || !optimizer_params.empty()) {
    ProfilerMgr::GetInstance()->UploadParamData(std::make_unique<torch_npu::toolkit::profiler::ParamTensorData>(
        std::move(module_params), std::move(optimizer_params)));
  }
}

void PythonTracer::recordEvent(ThreadLocalResult& tls, TraceTag tag, size_t hash_key) {
  tls.tid_ = torch_npu::toolkit::profiler::Utils::GetTid();
  tls.events_.emplace_back(tls.tid_, torch_npu::toolkit::profiler::Utils::GetClockTime(), hash_key, tag);
  if (tls.events_.size() >= TRACE_DUMP_THRESHOLD) {
    reportTraceData(tls);
  }
}

static TensorMetadata toTensorMetadata(PyObject* self) {
  if (!THPVariable_CheckExact(self)) {
    TensorMetadata m;
    return m;
  }
  const auto& t = THPVariable_Unpack(self);
  TensorMetadata m{t};
  return m;
}

static c10::optional<TensorMetadata> recordIfTensor(py::handle p) {
  return THPVariable_CheckExact(p.ptr()) ? c10::optional<TensorMetadata>(toTensorMetadata(p.ptr())) : c10::nullopt;
}

static std::vector<std::pair<std::string, TensorMetadata>> unpackTensorMap(const py::dict& tensor_map) {
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
    std::vector<std::pair<size_t, std::vector<ModuleParam>>>& module_param_cache,
    PyObject* cls,
    size_t hash_id) {
  std::vector<ModuleParam> module_params;
  py::dict params = py::handle(cls).attr("_parameters");
  for (auto& it : params) {
    auto* p = it.second.ptr();
    if (py::isinstance<py::str>(it.first) && THPVariable_CheckExact(p)) {
      module_params.emplace_back(ModuleParam{
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
    std::vector<std::pair<size_t, std::vector<OptimizerParam>>>& optimizer_param_cache,
    PyObject* cls,
    size_t hash_id) {
  std::vector<OptimizerParam> optimizer_params;
  const py::handle self{cls};
  for (const auto& it : (py::list)self.attr("param_groups")) {
    for (auto& param : py::cast<py::dict>(it).attr("get")("params")) {
      if (THPVariable_CheckExact(param.ptr())) {
        optimizer_params.emplace_back(OptimizerParam{
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

size_t PythonTracer::genPyCallHashId(ThreadLocalResult& tls, PyFrameObject* frame) {
  TORCH_INTERNAL_ASSERT(frame != nullptr, "frame can not be nullptr.", PTA_ERROR(ErrCode::PARAM));

  auto call_info = PyCallInfo(frame);
  auto hash_id = call_info.get_hash_id();
  auto f_code = PyFrame_GetCode_NPU(frame);
  if (tls.py_call_cache_.find(hash_id) == tls.py_call_cache_.end()) {
    tls.py_call_cache_.insert({hash_id, call_info});

    // check optim.Optimizer call
    if (record_params_ && ((PyObject*)f_code.get() == optimizer_call_code_)) {
      auto f_locals = PyFrame_GetLocals_NPU(frame);
      if (!f_locals) {
        PyErr_Clear();
        return hash_id;
      }
      PyObject* optimizer_class = nullptr;
#if PY_MAJOR_VERSION < 3 || (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 13)
      optimizer_class = PyDict_GetItemString(f_locals, "self");
#else
      auto optimizer_class_holder = THPObjectPtr(PyMapping_GetItemString(f_locals, "self"));
      optimizer_class = optimizer_class_holder.get();
#endif
      if (optimizer_class == nullptr) {
        if (!PyErr_Occurred()) {
          PyErr_SetString(PyExc_KeyError, "Optimizer frame has no self");
        }
        throw py::error_already_set();
      }
      parseOptimizerParams(tls.optimizer_param_cache_, optimizer_class, hash_id);
    }
  }

  // check nn.Module call
  if ((PyObject*)f_code.get() == module_call_code_) {
    auto f_locals = PyFrame_GetLocals_NPU(frame);
    if (!f_locals) {
      PyErr_Clear();
      return hash_id;
    }
    PyObject* module_class = nullptr;
#if PY_MAJOR_VERSION < 3 || (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 13)
    module_class = PyDict_GetItemString(f_locals, "self");
#else
    auto module_class_holder = THPObjectPtr(PyMapping_GetItemString(f_locals, "self"));
    module_class = module_class_holder.get();
#endif
    if (module_class == nullptr) {
      if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_KeyError, "Module frame has no self");
      }
      throw py::error_already_set();
    }
    hash_id = c10::get_hash(module_class);
    if (tls.module_info_cache_.find(hash_id) == tls.module_info_cache_.end()) {
      tls.module_info_cache_.insert({hash_id, ModuleInfo(module_class)});

      if (record_params_) {
        parseModuleParams(tls.module_param_cache_, module_class, hash_id);
      }
    }
  }
  return hash_id;
}

void PythonTracer::recordPyCall(ThreadLocalResult& tls, PyFrameObject* frame) {
  auto hash_id = genPyCallHashId(tls, frame);
  recordEvent(tls, TraceTag::kPy_Call, hash_id);
}

void PythonTracer::recordCCall(ThreadLocalResult& tls, PyFrameObject* frame, PyObject* arg) {
  std::string call_info = py::repr(arg);
  auto hash_id = c10::get_hash(call_info);
  if (tls.pyc_call_cache_.find(hash_id) == tls.pyc_call_cache_.end()) {
    tls.pyc_call_cache_.insert({hash_id, at::StringView(std::move(call_info))});
  }
  recordEvent(tls, TraceTag::kC_Call, hash_id);
}

void PythonTracer::recordReturn(ThreadLocalResult& tls, TraceTag tag) {
  recordEvent(tls, tag, EXIT_EVENT_HASH_ID);
}

int PythonTracer::pyProfileFn(PyObject* obj, PyFrameObject* frame, int what, PyObject* arg) {
  try {
    auto ctx = reinterpret_cast<TraceContext*>(obj);
    if (ctx == nullptr) {
      return 0;
    }
    auto* tracer = ctx->tracer_;
    if (tracer == nullptr || !tracer->active_.load()) {
      return 0;
    }
    auto tls = tracer->findThreadLocalResult(PyThreadState_Get());
    if (!tls || tls->context_ != ctx) {
      return 0;
    }
    std::unique_lock<std::mutex> callback(tls->callback_mutex_, std::try_to_lock);
    if (!callback.owns_lock() || !tls->enabled_ || !tracer->active_.load()) {
      return 0;
    }
    switch (what) {
      case PyTrace_CALL:
        tracer->recordPyCall(*tls, frame);
        break;

      case PyTrace_C_CALL:
        tracer->recordCCall(*tls, frame, arg);
        break;

      case PyTrace_EXCEPTION:
      case PyTrace_RETURN:
        tracer->recordReturn(*tls, TraceTag::kPy_Return);
        break;

      case PyTrace_C_EXCEPTION:
      case PyTrace_C_RETURN:
        tracer->recordReturn(*tls, TraceTag::kC_Return);
        break;

      default:
        break;
    }
    return 0;
  } catch (...) {
    PyErr_Clear();
    return 0;
  }
}

void PythonTracer::call(Command c) {
  switch (c) {
    case Command::kStartOne:
      if (auto tracer = get_singleton_in_child_thread()) {
        tracer->startOne();
      }
      break;

    case Command::kStartAll:
      PythonTracer::singleton().start();
      break;

    case Command::kStop:
      PythonTracer::singleton().stop();
      break;

    case Command::kStopOne:
      if (auto tracer = get_singleton_in_child_thread()) {
        tracer->stopOne();
      }
      break;

    case Command::kClear:
      PythonTracer::singleton().clear();
      break;

    default:
      break;
  }
};

void init() {
  pybind11::gil_scoped_acquire gil;
  TORCH_CHECK(PyType_Ready(&TraceContextType) == 0, PROF_ERROR(ErrCode::INTERNAL));
  registerFunctions(&PythonTracer::call);
}
} // namespace python_tracer
} // namespace profiler
} // namespace torch_npu
