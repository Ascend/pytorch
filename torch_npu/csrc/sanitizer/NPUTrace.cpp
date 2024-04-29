#include <cstdlib>
#include "torch_npu/csrc/sanitizer/NPUTrace.h"

namespace c10_npu {
namespace impl {

std::atomic<const PyCallbackTrigger*> NPUTrace::npu_trace_state{nullptr};
bool NPUTrace::have_state{false};

void NPUTrace::setTrace(const PyCallbackTrigger* trace)
{
    static std::once_flag flag;
    std::call_once(flag, [&]() {
        npu_trace_state.store(trace, std::memory_order_release);
    });
    have_state = true;
}

void activateNPUTrace(const int mode)
{
    c10_npu::impl::NPUTrace::setTrace(getPyCallbackTrigger(mode));
}

} // namespace impl
} // namespace c10_npu
