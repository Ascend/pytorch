#pragma once

#include <c10/core/Allocator.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace c10_npu {
namespace NPUWorkspaceAllocator {

using c10_npu::NPUCachingAllocator::CreateContextFn;
using c10_npu::NPUCachingAllocator::BlockInfo;
using c10_npu::NPUCachingAllocator::SegmentInfo;
using c10_npu::NPUCachingAllocator::TraceEntry;
using c10_npu::NPUCachingAllocator::SnapshotInfo;
using c10_npu::NPUCachingAllocator::RecordContext;
using c10_npu::NPUCachingAllocator::Stat;

struct DeviceStats {
    // SUM: bytes requested by client code
    Stat allocated_bytes;
    // SUM: bytes reserved by this memory allocator (both free and used)
    Stat reserved_bytes;
};

c10::Allocator* get();
void init();
c10::DataPtr malloc_with_stream(size_t size, aclrtStream stream);
void emptyCache(int device, bool need_empty_queue, bool check_error = true);
void recordHistory(bool enabled, CreateContextFn context_recorder, RecordContext when);
SnapshotInfo snapshot();
DeviceStats getDeviceStats(int device);

} // namespace NPUWorkspaceAllocator
} // namespace c10_npu

namespace at_npu {
namespace native {

TORCH_NPU_API at::Tensor allocate_workspace(uint64_t workspace_size, aclrtStream stream);

} // namespace native
} // namespace at_npu
