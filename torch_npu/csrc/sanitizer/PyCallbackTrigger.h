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

enum class SanitizerMode { KERNEL = 1 };

struct PyCallbackTrigger {
    const SanitizerMode sanitizer_mode;

    PyCallbackTrigger(const int mode) : sanitizer_mode(static_cast<SanitizerMode>(mode)){};
    void traceNpuAclStartExecution(std::string acl_name) const
    {
        if (sanitizer_mode == SanitizerMode::KERNEL)
            CONCRETE_TRACE_NPU("NPUACLStartExecuteCallbacks", acl_name);
    }
    void traceNpuAclFinishExecution(std::string acl_name) const
    {
        if (sanitizer_mode == SanitizerMode::KERNEL)
            CONCRETE_TRACE_NPU("NPUACLFinishExecuteCallbacks", acl_name);
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
