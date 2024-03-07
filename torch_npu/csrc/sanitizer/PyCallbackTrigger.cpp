#include <torch/csrc/utils/python_arg_parser.h>
#include "torch_npu/csrc/sanitizer/PyCallbackTrigger.h"

using namespace at;
using namespace c10_npu;

namespace {

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

struct SanitizerPyCallbackTriggerVTable final
    : public c10_npu::impl::PyCallbackTriggerVTable {
    std::string name() const override;

    void traceNpuAclExecution(std::string acl_name) const override
    {
        CONCRETE_TRACE_NPU("NPUACLExecuteCallbacks", acl_name);
    }

    static SanitizerPyCallbackTriggerVTable* instance()
    {
        static SanitizerPyCallbackTriggerVTable s;
        return &s;
    }
};
} // anonymous namespace

std::string SanitizerPyCallbackTriggerVTable::name() const
{
    return "<sanitizer callback trigger>";
}

namespace c10_npu {
namespace impl {

static std::unique_ptr<PyCallbackTrigger> pyCallbackTriggerInstance;

PyCallbackTrigger* getPyCallbackTrigger()
{
    if (!pyCallbackTriggerInstance) {
        pyCallbackTriggerInstance = std::make_unique<PyCallbackTrigger>(
            SanitizerPyCallbackTriggerVTable::instance()
        );
    }
    return pyCallbackTriggerInstance.get();
}

} // namespace impl
} // namespace c10_npu
