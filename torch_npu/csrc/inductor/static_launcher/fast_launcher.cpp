#ifndef BUILD_LIBTORCH

#include "torch_npu/csrc/inductor/static_launcher/fast_launcher.h"

#include <cstddef>
#include <limits>
#include <memory>
#include <torch/csrc/Exceptions.h>

#include "torch_npu/csrc/inductor/static_launcher/runtime.h"

namespace {

using torch_npu::inductor::StaticNpuKernel;

struct FastLauncherState {
  std::shared_ptr<StaticNpuKernel> kernel;
  size_t nargs;
};

struct FastLauncher {
  PyObject_HEAD
  vectorcallfunc vectorcall;
  // tp_alloc does not construct nontrivial C++ members.
  FastLauncherState* state;
};

uint64_t UnsignedArgument(PyObject* value) {
  const auto number = PyLong_AsUnsignedLongLong(value);
  if (PyErr_Occurred()) {
    throw python_error();
  }
  return number;
}

PyObject* FastCall(PyObject* callable, PyObject* const* args, size_t nargsf, PyObject* kwnames) {
  HANDLE_TH_ERRORS
  auto* self = reinterpret_cast<FastLauncher*>(callable);
  if (kwnames != nullptr && PyTuple_GET_SIZE(kwnames) != 0) {
    PyErr_SetString(PyExc_TypeError, "NPU fast runner accepts positional arguments only");
    return nullptr;
  }
  const size_t nargs = PyVectorcall_NARGS(nargsf);
  TORCH_CHECK(
      nargs == self->state->nargs + 4,
      "NPU fast launcher argument count mismatch: ",
      nargs,
      " vs ",
      self->state->nargs + 4);
  uint32_t grid[3];
  for (size_t index = 0; index < 3; ++index) {
    const uint64_t value = UnsignedArgument(args[index]);
    TORCH_CHECK(value <= std::numeric_limits<uint32_t>::max(), "NPU fast launcher grid exceeds uint32 max");
    grid[index] = static_cast<uint32_t>(value);
  }
  const uint64_t stream = UnsignedArgument(args[3]);
  self->state->kernel->LaunchFast(grid[0], grid[1], grid[2], stream, args + 4, self->state->nargs);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* FastNew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  if (kwargs != nullptr && PyDict_Size(kwargs) != 0) {
    PyErr_SetString(PyExc_TypeError, "_FastNpuLauncher expects one positional loaded kernel");
    return nullptr;
  }
  PyObject* loaded = nullptr;
  if (!PyArg_UnpackTuple(args, "_FastNpuLauncher", 1, 1, &loaded)) {
    return nullptr;
  }
  auto owner = pybind11::cast<std::shared_ptr<StaticNpuKernel>>(pybind11::handle(loaded));
  TORCH_CHECK(owner != nullptr, "NPU fast launcher kernel is null");
  owner->CheckOpen();
  auto state = std::make_unique<FastLauncherState>();
  state->nargs = owner->ArgumentCount();
  state->kernel = std::move(owner);
  auto* self = reinterpret_cast<FastLauncher*>(type->tp_alloc(type, 0));
  if (self == nullptr) {
    return nullptr;
  }
  self->state = state.release();
  self->vectorcall = FastCall;
  return reinterpret_cast<PyObject*>(self);
  END_HANDLE_TH_ERRORS
}

void FastDealloc(PyObject* object) {
  auto* self = reinterpret_cast<FastLauncher*>(object);
  delete self->state;
  Py_TYPE(object)->tp_free(object);
}

PyTypeObject FastLauncherType = {PyVarObject_HEAD_INIT(nullptr, 0)
};

} // namespace

void RegisterNPUFastLauncherBindings(PyObject* module) {
  FastLauncherType.tp_name = "torch_npu._C._FastNpuLauncher";
  FastLauncherType.tp_basicsize = sizeof(FastLauncher);
  FastLauncherType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL;
  FastLauncherType.tp_vectorcall_offset = offsetof(FastLauncher, vectorcall);
  FastLauncherType.tp_call = PyVectorcall_Call;
  FastLauncherType.tp_new = FastNew;
  FastLauncherType.tp_dealloc = FastDealloc;
  FastLauncherType.tp_doc = "Bound NPU static kernel with a vectorcall argument entry";
  if (PyType_Ready(&FastLauncherType) < 0) {
    throw python_error();
  }
  Py_INCREF(&FastLauncherType);
  if (PyModule_AddObject(module, "_FastNpuLauncher", reinterpret_cast<PyObject*>(&FastLauncherType)) < 0) {
    Py_DECREF(&FastLauncherType);
    throw python_error();
  }
}

#endif
