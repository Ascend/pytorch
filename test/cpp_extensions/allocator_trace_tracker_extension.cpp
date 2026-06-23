#include <atomic>
#include <mutex>

#include <torch/extension.h>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace py = pybind11;

namespace {

using TraceEntry = c10_npu::NPUCachingAllocator::TraceEntry;

std::once_flag tracker_registration_once;
std::atomic<int64_t> segment_alloc_count{0};
std::atomic<int64_t> segment_free_count{0};

void update_trace_tracker_state(const TraceEntry& te)
{
    if (te.action_ == TraceEntry::SNAPSHOT) {
        return;
    }

    if (te.action_ == TraceEntry::SEGMENT_ALLOC) {
        segment_alloc_count.fetch_add(1, std::memory_order_relaxed);
    } else if (te.action_ == TraceEntry::SEGMENT_FREE) {
        segment_free_count.fetch_add(1, std::memory_order_relaxed);
    }
}

void attach_trace_tracker()
{
    std::call_once(tracker_registration_once, []() {
        c10_npu::NPUCachingAllocator::attachAllocatorTraceTracker(
            &update_trace_tracker_state);
    });
}

void reset_trace_tracker_state()
{
    segment_alloc_count.store(0, std::memory_order_relaxed);
    segment_free_count.store(0, std::memory_order_relaxed);
}

py::dict get_trace_tracker_state()
{
    py::dict result;
    result["segment_alloc_count"] =
        segment_alloc_count.load(std::memory_order_relaxed);
    result["segment_free_count"] =
        segment_free_count.load(std::memory_order_relaxed);
    return result;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("attach_trace_tracker", &attach_trace_tracker);
    m.def("reset_trace_tracker_state", &reset_trace_tracker_state);
    m.def("get_trace_tracker_state", &get_trace_tracker_state);
}
