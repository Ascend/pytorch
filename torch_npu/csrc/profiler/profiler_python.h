#pragma once

#include <cstdint>

namespace torch_npu {
namespace profiler {
namespace python_tracer {
void init();

enum class TraceTag {
    kPy_Call = 0,
    kPy_Return,
    kC_Call,
    kC_Return
};

struct TraceEvent {
    TraceEvent() = default;
    TraceEvent(uint64_t tid, uint64_t timestamp, size_t key, TraceTag tag)
        : tid_(tid),
          ts_(timestamp),
          key_(static_cast<uint64_t>(key)),
          tag_(static_cast<uint8_t>(tag)) {}

    TraceEvent(const TraceEvent&) = default;
    TraceEvent& operator=(const TraceEvent&) = default;

    uint64_t tid_{0};
    uint64_t ts_{0};
    uint64_t key_{0};
    uint8_t tag_{0};
};
} // python_tracer
} // namespace profiler
} // namespace torch_npu
