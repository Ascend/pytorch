#include <ATen/Context.h>

#include "torch_npu/csrc/profiler/combined_traceback.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/npu/memory_snapshot.h"
#include "torch_npu/csrc/utils/LazyInit.h"

namespace torch_npu {

std::shared_ptr<torch_npu::GatheredContext> gather()
{
    return torch_npu::CapturedTraceback::gather(true, true, false);
}

std::shared_ptr<torch_npu::GatheredContext> gather_with_cpp()
{
    return torch_npu::CapturedTraceback::gather(true, true, true);
}

static void checkOptionIn(const std::string& option,
                          std::initializer_list<std::string> valid,
                          const char* error)
{
    TORCH_CHECK(valid.end() != std::find(valid.begin(), valid.end(), option),
                error);
}

void _record_memory_history(c10::optional<std::string> enabled,
                            c10::optional<std::string> context,
                            std::string stacks, size_t max_entries)
{
    if (enabled) {
        checkOptionIn(*enabled, {"state", "all"},
                      "expected state to be 'state', 'all', or None");
    }
    if (context) {
        checkOptionIn(
            *context, {"state", "alloc", "all"},
            "expected context to be 'state', 'alloc', 'all', or None");
    }
    checkOptionIn(stacks, {"python", "all"},
                  "expected stacks to be 'python', or 'all'");

    c10_npu::NPUCachingAllocator::CreateContextFn recorder = gather;
    if (enabled && stacks == "all") {
        recorder = gather_with_cpp;
        // warm up C++ stack unwinding
        torch_npu::unwind::unwind();
    }
    max_entries = (enabled && *enabled == "all") ? max_entries : 1;
    auto when = c10_npu::NPUCachingAllocator::RecordContext::NEVER;
    if (context) {
        if (context == "all") {
            when = c10_npu::NPUCachingAllocator::RecordContext::ALL;
        } else if (context == "alloc") {
            when = c10_npu::NPUCachingAllocator::RecordContext::ALLOC;
        } else if (context == "state") {
            when = c10_npu::NPUCachingAllocator::RecordContext::STATE;
        }
    }
    torch_npu::utils::npu_lazy_init();
    c10_npu::NPUCachingAllocator::recordHistory(enabled.has_value(), recorder,
                                                max_entries, when);
}

} // namespace torch_npu
