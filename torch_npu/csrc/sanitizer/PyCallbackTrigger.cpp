#include <torch/csrc/utils/python_arg_parser.h>
#include "torch_npu/csrc/sanitizer/NPUTrace.h"

#define CONCRETE_TRACE_NPU(func_name, ...)                               \
  if (Py_IsInitialized()) {                                              \
    pybind11::gil_scoped_acquire gil;                                    \
    try {                                                                \
      py::module mod = py::module::import("torch_npu.utils._npu_trace"); \
      py::object hook = mod.attr(func_name).attr("fire_callbacks");      \
      hook(__VA_ARGS__);                                                 \
    } catch (const std::exception& e) {                                  \
      LOG(ERROR) << "NPU trace hook execution failed: " << e.what();     \
    }                                                                    \
  }

namespace c10_npu {
namespace impl {
namespace {

void traceNpuAclStartExecution(const std::string& acl_name) {
  CONCRETE_TRACE_NPU("NPUACLStartExecuteCallbacks", acl_name);
}

void traceNpuAclFinishExecution(const std::string& acl_name) {
  CONCRETE_TRACE_NPU("NPUACLFinishExecuteCallbacks", acl_name);
}

void traceNpuEventCreation(uintptr_t event) {
  CONCRETE_TRACE_NPU("NPUEventCreationCallbacks", event);
}

void traceNpuEventDeletion(uintptr_t event) {
  CONCRETE_TRACE_NPU("NPUEventDeletionCallbacks", event);
}

void traceNpuEventRecord(uintptr_t event, uintptr_t stream) {
  CONCRETE_TRACE_NPU("NPUEventRecordCallbacks", event, stream);
}

void traceNpuEventWait(uintptr_t event, uintptr_t stream) {
  CONCRETE_TRACE_NPU("NPUEventWaitCallbacks", event, stream);
}

void traceNpuEventGetHandle(uintptr_t event, uintptr_t handle) {
  CONCRETE_TRACE_NPU("NPUEventGetHandleCallbacks", event, handle);
}

void traceNpuEventOpenHandle(uintptr_t handle, uintptr_t event) {
  CONCRETE_TRACE_NPU("NPUEventOpenHandleCallbacks", handle, event);
}

void traceNpuMemoryAllocation(uintptr_t ptr) {
  CONCRETE_TRACE_NPU("NPUMemoryAllocationCallbacks", ptr);
}

void traceNpuMemoryDeallocation(uintptr_t ptr) {
  CONCRETE_TRACE_NPU("NPUMemoryDeallocationCallbacks", ptr);
}

void traceNpuStreamCreation(uintptr_t stream) {
  CONCRETE_TRACE_NPU("NPUStreamCreationCallbacks", stream);
}

void traceNpuDeviceSynchronization() {
  CONCRETE_TRACE_NPU("NPUDeviceSynchronizationCallbacks");
}

void traceNpuStreamSynchronization(uintptr_t stream) {
  CONCRETE_TRACE_NPU("NPUStreamSynchronizationCallbacks", stream);
}

void traceNpuEventSynchronization(uintptr_t event) {
  CONCRETE_TRACE_NPU("NPUEventSynchronizationCallbacks", event);
}

void traceNpuExternalEventWrite(uintptr_t event, uintptr_t stream) {
  CONCRETE_TRACE_NPU("traceNpuExternalEventWriteCallbacks", event, stream);
}

void traceNpuExternalEventWait(uintptr_t event, uintptr_t stream) {
  CONCRETE_TRACE_NPU("traceNpuExternalEventWaitCallbacks", event, stream);
}

void traceNpuRecordStream(uintptr_t data_ptr, uintptr_t stream) {
  CONCRETE_TRACE_NPU("NPURecordStreamCallbacks", data_ptr, stream);
}

void traceNpuEraseStream(uintptr_t data_ptr, uintptr_t stream) {
  CONCRETE_TRACE_NPU("NPUEraseStreamCallbacks", data_ptr, stream);
}

const NPUTraceCallbacks python_callbacks{
    traceNpuAclStartExecution,
    traceNpuAclFinishExecution,
    traceNpuEventCreation,
    traceNpuEventDeletion,
    traceNpuEventRecord,
    traceNpuEventWait,
    traceNpuEventGetHandle,
    traceNpuEventOpenHandle,
    traceNpuMemoryAllocation,
    traceNpuMemoryDeallocation,
    traceNpuStreamCreation,
    traceNpuDeviceSynchronization,
    traceNpuStreamSynchronization,
    traceNpuEventSynchronization,
    traceNpuExternalEventWrite,
    traceNpuExternalEventWait,
    traceNpuRecordStream,
    traceNpuEraseStream,
};

} // namespace

PyCallbackTrigger* PyCallbackTrigger::instance(const int mode) {
  static PyCallbackTrigger trigger(mode, python_callbacks);
  return &trigger;
}

PyCallbackTrigger* getPyCallbackTrigger(const int mode) {
  return PyCallbackTrigger::instance(mode);
}

void activateNPUTrace(const int mode) {
  NPUTrace::setTrace(getPyCallbackTrigger(mode));
}

} // namespace impl
} // namespace c10_npu
