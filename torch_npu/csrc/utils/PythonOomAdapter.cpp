#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/Exceptions.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <unistd.h>

#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/utils/PythonOomAdapter.h"

namespace torch_npu::utils {
namespace {
void observe(int64_t device, int64_t allocated, int64_t device_total, int64_t device_free) {
  auto dumppath = c10_npu::option::OptionsManager::GetOomSnapshotDumpPath();
  std::stringstream filename;
  auto now = std::chrono::system_clock::now();
  std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
  std::tm* timeInfo = std::localtime(&currentTime);
  filename << "oom_snapshot_" << getpid() << "_" << std::put_time(timeInfo, "%Y%m%d%H%M%S") << ".pickle";
  auto savefilepath = c10::str(dumppath, "/", filename.str());

  pybind11::gil_scoped_acquire g;
  auto module = THPObjectPtr(PyImport_ImportModule("torch_npu.npu.memory"));
  if (!module) {
    throw python_error();
  }
  PyObject* p_func = PyObject_GetAttrString(module, "_dump_snapshot");
  if (!module) {
    throw python_error();
  }
  PyObject* p_args = PyTuple_New(1);
  PyTuple_SetItem(p_args, 0, PyUnicode_FromString(savefilepath.c_str()));
  PyObject* p_res = PyObject_CallObject(p_func, p_args);
}

const c10_npu::option::PythonOomHooks hooks{observe};
} // namespace

void registerPythonOomAdapter() {
  c10_npu::option::registerPythonOomHooks(&hooks);
}
} // namespace torch_npu::utils
