#pragma once

#include <string>
#include <torch/csrc/utils/python_arg_parser.h>

#define CONCRETE_TRACE_NPU(func_name, ...)                                     \
    if (Py_IsInitialized()) {                                                  \
        pybind11::gil_scoped_acquire gil;                                      \
        try {                                                                  \
            py::module mod = py::module::import("torch_npu.utils._npu_trace"); \
            py::object hook = mod.attr(func_name).attr("fire_callbacks");      \
            hook(__VA_ARGS__);                                                 \
        } catch (const std::exception& e) {                                    \
            LOG(ERROR) << "NPU trace hook execution failed: " << e.what();     \
        }                                                                      \
    }

namespace c10_npu {
namespace impl {

enum class SanitizerMode { STREAM = 0, KERNEL };

struct PyCallbackTrigger {
    const SanitizerMode sanitizer_mode;

    PyCallbackTrigger(const int mode) : sanitizer_mode(static_cast<SanitizerMode>(mode)){};
    void traceNpuAclStartExecution(std::string acl_name) const
    {
        if (sanitizer_mode == SanitizerMode::KERNEL) {
            CONCRETE_TRACE_NPU("NPUACLStartExecuteCallbacks", acl_name);
        }
    }
    void traceNpuAclFinishExecution(std::string acl_name) const
    {
        if (sanitizer_mode == SanitizerMode::KERNEL) {
            CONCRETE_TRACE_NPU("NPUACLFinishExecuteCallbacks", acl_name);
        }
    }
    void traceNpuEventCreation(uintptr_t event) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
            CONCRETE_TRACE_NPU("NPUEventCreationCallbacks", event);
        }
    }
    void traceNpuEventDeletion(uintptr_t event) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
            CONCRETE_TRACE_NPU("NPUEventDeletionCallbacks", event);
        }
    }
    void traceNpuEventRecord(uintptr_t event, uintptr_t stream) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
            CONCRETE_TRACE_NPU("NPUEventRecordCallbacks", event, stream);
        }
    }
    void traceNpuEventWait(uintptr_t event, uintptr_t stream) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
            CONCRETE_TRACE_NPU("NPUEventWaitCallbacks", event, stream);
        }
    }
    void traceNpuMemoryAllocation(uintptr_t ptr) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
            CONCRETE_TRACE_NPU("NPUMemoryAllocationCallbacks", ptr);
        }
    }
    void traceNpuMemoryDeallocation(uintptr_t ptr) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
            CONCRETE_TRACE_NPU("NPUMemoryDeallocationCallbacks", ptr);
        }
    }
    void traceNpuStreamCreation(uintptr_t stream) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
            CONCRETE_TRACE_NPU("NPUStreamCreationCallbacks", stream);
        }
    }
    void traceNpuDeviceSynchronization() const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
            CONCRETE_TRACE_NPU("NPUDeviceSynchronizationCallbacks");
        }
    }
    void traceNpuStreamSynchronization(uintptr_t stream) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
            CONCRETE_TRACE_NPU("NPUStreamSynchronizationCallbacks", stream);
        }
    }
    void traceNpuEventSynchronization(uintptr_t event) const
    {
        if (sanitizer_mode == SanitizerMode::STREAM) {
            CONCRETE_TRACE_NPU("NPUEventSynchronizationCallbacks", event);
        }
    }
    static PyCallbackTrigger* instance(const int mode)
    {
        static PyCallbackTrigger trigger = PyCallbackTrigger(mode);
        return &trigger;
    }
};

PyCallbackTrigger* getPyCallbackTrigger(const int mode);

} // namespace impl
} // namespace c10_npu
