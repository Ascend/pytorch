#ifndef BUILD_LIBTORCH
#include <torch/csrc/python_headers.h>
#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <pybind11/embed.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <c10/util/Logging.h>

#include "torch_npu/csrc/distributed/PythonDebugBridge.h"
#include "torch_npu/csrc/distributed/PythonDebugAdapter.h"

namespace py = pybind11;
using namespace py::literals;

namespace c10d_npu {
namespace {
bool dumpPythonTraceback(int rank, const std::string& prefix) {
  std::string filePath = c10d::getCvarString({"TORCH_HCCL_DEBUG_INFO_TEMP_FILE"}, "/tmp/hccl_trace_rank_");
  PyGILState_STATE gil = PyGILState_Ensure();
  try {
    py::dict locals = py::dict("path"_a = filePath.c_str(), "rank"_a = rank);
    py::exec(
        R"(
            import sys
            import os
            import traceback
            import threading
            from torch_npu.utils._path_manager import PathManager
            try:
                py_stacks = 'pid: {}\n'.format(os.getpid())
                threadInfos = {}
                for thread in threading.enumerate():
                    threadInfos[thread.ident] = thread
                for thread_id, stack in sys._current_frames().items():
                    stack_list = traceback.format_list(traceback.extract_stack(stack))
                    py_stacks += 'thread {}:\n'.format(threadInfos[thread_id] if thread_id in threadInfos.keys() else thread_id)
                    py_stacks += ''.join(stack_list)
                dump_file = '{path}{rank}_py_traceback'.format(**locals())
                PathManager.check_input_file_path(dump_file)
                with open(dump_file, 'w') as f:
                    f.write(py_stacks)
            except Exception as e:
                print(e);
            )",
        py::globals(),
        locals);
  } catch (const std::exception& e) {
    LOG(ERROR) << prefix << "dumpPythonTraceback error: " << e.what();
  } catch (...) {
    LOG(ERROR) << prefix << "dumpPythonTraceback Unknown exception type.";
  }
  PyGILState_Release(gil);
  return true;
}
} // namespace

void registerPythonDebugAdapter() {
  registerPythonTracebackDump(&dumpPythonTraceback);
}
} // namespace c10d_npu
#endif
