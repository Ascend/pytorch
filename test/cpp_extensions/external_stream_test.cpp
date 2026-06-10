#include <torch/extension.h>
#include <c10/util/irange.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUGraph.h"
#include "third_party/acl/inc/acl/acl.h"

using namespace at;

// Get NPU device - assumes torch_npu is already initialized via import
static c10::Device get_npu_device() {
    return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

// Test 1: External stream creation
bool test_external_stream_creation()
{
    // Create ACL stream directly
    aclrtStream acl_stream = nullptr;
    auto ret = aclrtCreateStream(&acl_stream);
    if (ret != ACL_ERROR_NONE || acl_stream == nullptr) {
        return false;
    }

    // Wrap as NPUStream
    c10_npu::NPUStream npu_stream = c10_npu::getStreamFromExternal(acl_stream, 0);

    // Verify stream_id (bit 30 marker for external)
    c10::StreamId stream_id = npu_stream.id();
    if (!((static_cast<uint64_t>(stream_id) & (1ULL << 30)) != 0)) {
        aclrtDestroyStream(acl_stream);
        return false;
    }

    // Verify aclrtStream is preserved
    if (npu_stream.stream() != acl_stream) {
        aclrtDestroyStream(acl_stream);
        return false;
    }

    // Default stream should have different encoding (bit 30 = 0)
    c10_npu::NPUStream default_stream = c10_npu::getDefaultNPUStream(0);
    c10::StreamId default_id = default_stream.id();
    if ((static_cast<uint64_t>(default_id) & (1ULL << 30)) != 0) {
        aclrtDestroyStream(acl_stream);
        return false;
    }

    aclrtDestroyStream(acl_stream);
    return true;
}

// Test 2: External stream as current stream
bool test_external_stream_as_current()
{
    aclrtStream acl_stream = nullptr;
    auto ret = aclrtCreateStream(&acl_stream);
    if (ret != ACL_ERROR_NONE) {
        return false;
    }

    c10_npu::NPUStream ext_stream = c10_npu::getStreamFromExternal(acl_stream, 0);

    // Set current stream to external
    c10_npu::setCurrentNPUStream(ext_stream);

    // Get current stream - should be the external stream
    c10_npu::NPUStream current = c10_npu::getCurrentNPUStream(0);
    if (current.id() != ext_stream.id()) {
        c10_npu::setCurrentNPUStream(c10_npu::getDefaultNPUStream(0));
        aclrtDestroyStream(acl_stream);
        return false;
    }

    // Reset to default stream
    c10_npu::setCurrentNPUStream(c10_npu::getDefaultNPUStream(0));

    aclrtDestroyStream(acl_stream);
    return true;
}

// Test 3: Tensor operations on external stream
bool test_operations_on_external_stream()
{
    auto npu_device = get_npu_device();

    aclrtStream acl_stream = nullptr;
    auto ret = aclrtCreateStream(&acl_stream);
    if (ret != ACL_ERROR_NONE) {
        return false;
    }

    c10_npu::NPUStream ext_stream = c10_npu::getStreamFromExternal(acl_stream, 0);

    // Set current stream to external
    c10_npu::setCurrentNPUStream(ext_stream);

    // Execute operations - should work synchronously on external stream
    auto a = torch::randn({2, 3}).to(npu_device);
    auto b = torch::randn({2, 3}).to(npu_device);
    auto c = a + b;

    bool success = (c.size(0) == 2 && c.size(1) == 3);

    // Reset to default stream BEFORE tensor destruction
    c10_npu::setCurrentNPUStream(c10_npu::getDefaultNPUStream(0));

    aclrtDestroyStream(acl_stream);
    return success;
}

// Test 4: Event record restriction
bool test_event_record_restriction()
{
    aclrtStream acl_stream = nullptr;
    auto ret = aclrtCreateStream(&acl_stream);
    if (ret != ACL_ERROR_NONE) {
        return false;
    }

    c10_npu::NPUStream ext_stream = c10_npu::getStreamFromExternal(acl_stream, 0);

    c10_npu::NPUEvent event;

    // record() should throw for external stream
    bool threw = false;
    try {
        event.record(ext_stream);
    } catch (const c10::Error&) {
        threw = true;
    }

    aclrtDestroyStream(acl_stream);
    return threw;
}

// Test 5: Event block restriction
bool test_event_block_restriction()
{
    aclrtStream acl_stream = nullptr;
    auto ret = aclrtCreateStream(&acl_stream);
    if (ret != ACL_ERROR_NONE) {
        return false;
    }

    c10_npu::NPUStream ext_stream = c10_npu::getStreamFromExternal(acl_stream, 0);

    c10_npu::NPUEvent event;

    // block() should throw for external stream
    bool threw = false;
    try {
        event.block(ext_stream);
    } catch (const c10::Error&) {
        threw = true;
    }

    aclrtDestroyStream(acl_stream);
    return threw;
}

// Test 6: Graph capture restriction
bool test_graph_capture_restriction()
{
    aclrtStream acl_stream = nullptr;
    auto ret = aclrtCreateStream(&acl_stream);
    if (ret != ACL_ERROR_NONE) {
        return false;
    }

    c10_npu::NPUStream ext_stream = c10_npu::getStreamFromExternal(acl_stream, 0);

    // Set current stream to external
    c10_npu::setCurrentNPUStream(ext_stream);

    c10_npu::NPUGraph graph;

    // capture_begin should throw
    bool threw = false;
    try {
        graph.capture_begin();
    } catch (const c10::Error&) {
        threw = true;
    }

    // Reset to default stream
    c10_npu::setCurrentNPUStream(c10_npu::getDefaultNPUStream(0));

    aclrtDestroyStream(acl_stream);
    return threw;
}

// Test 7: Query restriction
bool test_query_restriction()
{
    aclrtStream acl_stream = nullptr;
    auto ret = aclrtCreateStream(&acl_stream);
    if (ret != ACL_ERROR_NONE) {
        return false;
    }

    c10_npu::NPUStream ext_stream = c10_npu::getStreamFromExternal(acl_stream, 0);

    // query() should throw
    bool threw = false;
    try {
        ext_stream.query();
    } catch (const c10::Error&) {
        threw = true;
    }

    aclrtDestroyStream(acl_stream);
    return threw;
}

// Test 8: Synchronize restriction
bool test_synchronize_restriction()
{
    aclrtStream acl_stream = nullptr;
    auto ret = aclrtCreateStream(&acl_stream);
    if (ret != ACL_ERROR_NONE) {
        return false;
    }

    c10_npu::NPUStream ext_stream = c10_npu::getStreamFromExternal(acl_stream, 0);

    // synchronize() should throw
    bool threw = false;
    try {
        ext_stream.synchronize();
    } catch (const c10::Error&) {
        threw = true;
    }

    aclrtDestroyStream(acl_stream);
    return threw;
}

// Test 9: isSyncLaunchStream
bool test_is_sync_launch_stream()
{
    aclrtStream acl_stream = nullptr;
    auto ret = aclrtCreateStream(&acl_stream);
    if (ret != ACL_ERROR_NONE) {
        return false;
    }

    c10_npu::NPUStream ext_stream = c10_npu::getStreamFromExternal(acl_stream, 0);

    // External stream should NOT be sync launch stream
    bool is_sync = ext_stream.isSyncLaunchStream();

    aclrtDestroyStream(acl_stream);
    return !is_sync;
}

// Test 10: Multiple external streams
bool test_multiple_external_streams()
{
    aclrtStream acl_stream1 = nullptr;
    aclrtStream acl_stream2 = nullptr;
    auto ret1 = aclrtCreateStream(&acl_stream1);
    auto ret2 = aclrtCreateStream(&acl_stream2);

    if (ret1 != ACL_ERROR_NONE || ret2 != ACL_ERROR_NONE) {
        if (acl_stream1) aclrtDestroyStream(acl_stream1);
        if (acl_stream2) aclrtDestroyStream(acl_stream2);
        return false;
    }

    c10_npu::NPUStream npu_stream1 = c10_npu::getStreamFromExternal(acl_stream1, 0);
    c10_npu::NPUStream npu_stream2 = c10_npu::getStreamFromExternal(acl_stream2, 0);

    // Stream IDs should be different
    bool ids_different = (npu_stream1.id() != npu_stream2.id());

    // Both should have bit 30 marker
    bool both_external = ((static_cast<uint64_t>(npu_stream1.id()) & (1ULL << 30)) != 0) &&
                         ((static_cast<uint64_t>(npu_stream2.id()) & (1ULL << 30)) != 0);

    // aclrtStream values should be preserved
    bool streams_preserved = (npu_stream1.stream() == acl_stream1) &&
                             (npu_stream2.stream() == acl_stream2);

    aclrtDestroyStream(acl_stream1);
    aclrtDestroyStream(acl_stream2);

    return ids_different && both_external && streams_preserved;
}

// Test 11: Same acl_stream returns same NPUStream
bool test_same_acl_stream_same_npu_stream()
{
    aclrtStream acl_stream = nullptr;
    auto ret = aclrtCreateStream(&acl_stream);
    if (ret != ACL_ERROR_NONE) {
        return false;
    }

    // Call getStreamFromExternal twice with same acl_stream
    c10_npu::NPUStream npu_stream1 = c10_npu::getStreamFromExternal(acl_stream, 0);
    c10_npu::NPUStream npu_stream2 = c10_npu::getStreamFromExternal(acl_stream, 0);

    // Should return same NPUStream (same stream_id)
    bool same = (npu_stream1.id() == npu_stream2.id());

    aclrtDestroyStream(acl_stream);
    return same;
}

// Test 12: Pool stream vs external stream
bool test_pool_vs_external_stream()
{
    aclrtStream acl_stream = nullptr;
    auto ret = aclrtCreateStream(&acl_stream);
    if (ret != ACL_ERROR_NONE) {
        return false;
    }

    c10_npu::NPUStream ext_stream = c10_npu::getStreamFromExternal(acl_stream, 0);
    c10_npu::NPUStream pool_stream = c10_npu::getNPUStreamFromPool(0);

    // External stream has bit 30 = 1
    bool ext_has_marker = (static_cast<uint64_t>(ext_stream.id()) & (1ULL << 30)) != 0;

    // Pool stream has bit 30 = 0
    bool pool_no_marker = (static_cast<uint64_t>(pool_stream.id()) & (1ULL << 30)) == 0;

    aclrtDestroyStream(acl_stream);
    return ext_has_marker && pool_no_marker;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("test_external_stream_creation", &test_external_stream_creation);
    m.def("test_external_stream_as_current", &test_external_stream_as_current);
    m.def("test_operations_on_external_stream", &test_operations_on_external_stream);
    m.def("test_event_record_restriction", &test_event_record_restriction);
    m.def("test_event_block_restriction", &test_event_block_restriction);
    m.def("test_graph_capture_restriction", &test_graph_capture_restriction);
    m.def("test_query_restriction", &test_query_restriction);
    m.def("test_synchronize_restriction", &test_synchronize_restriction);
    m.def("test_is_sync_launch_stream", &test_is_sync_launch_stream);
    m.def("test_multiple_external_streams", &test_multiple_external_streams);
    m.def("test_same_acl_stream_same_npu_stream", &test_same_acl_stream_same_npu_stream);
    m.def("test_pool_vs_external_stream", &test_pool_vs_external_stream);
}