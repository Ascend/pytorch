#pragma once

#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/sanitizer/PyCallbackTrigger.h"

namespace c10_npu {
namespace impl {

struct NPUTrace {
    static std::atomic<const PyCallbackTrigger*> npu_trace_state;
    static bool have_state;

    // This function will only register the first interpreter that tries to invoke
    // it. For all of the next ones it will be a no-op.
    static void setTrace(const PyCallbackTrigger*);
    static const PyCallbackTrigger* getTrace()
    {
        if (!have_state)
            return nullptr;
        return npu_trace_state.load(std::memory_order_acquire);
    }
};

TORCH_NPU_API void activateNPUTrace();

} // namespace impl
} // namespace c10
