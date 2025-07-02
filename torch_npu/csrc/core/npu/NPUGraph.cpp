#include "torch_npu/csrc/core/npu/NPUGraph.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include <ATen/Functions.h>

namespace c10_npu {

static bool _npu_graphs_debug = false;
constexpr int kSynchronizeBusyWaitMillis = 10;

MempoolId_t graph_pool_handle()
{
    // Sets just the second value, to distinguish it from MempoolId_ts created from
    // aclmdlRICaptureGetInfo id_s in capture_begin.
    auto new_pool = c10_npu::MemPool();
    return new_pool.id();
}

void graph_task_group_begin(c10_npu::NPUStream stream)
{
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureTaskGrpBegin(stream));
}

NPUTaskGroupHandle graph_task_group_end(c10_npu::NPUStream stream)
{
    aclrtTaskGrp group;
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureTaskGrpEnd(stream, &group));
    NPUTaskGroupHandle handle;
    handle.task_group = group;
    return handle;
}

void graph_task_update_begin(c10_npu::NPUStream stream, NPUTaskGroupHandle handle)
{
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureTaskUpdateBegin(stream, handle.task_group));
}

void graph_task_update_end(c10_npu::NPUStream stream)
{
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureTaskUpdateEnd(stream));
}

/**
 * Note [CUDA Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Q: Why do we need graph capture and launch bindings in Pytorch?
 *    Why can't they live in a user extension, for example?
 *
 * A1: Convenience.
 * A2: To ensure valid numerics on replay, some native CUDA ops (like RNG ops with
 *     CPU statefulness) need cooperation from the capture and replay bindings
 *     (see Note [CUDA Graph-safe RNG states] in CUDAGeneratorImpl.h).
 *
 *     We can't expect users to know about this cooperation.  If users write capture
 *     bindings naively in an extension, they likely won't interact with the native
 *     ops properly.  Their graphs would yield invalid numerics on replay.
 */

/**
 * Note [Interaction with CUDA graph capture] in CUDACachingAllocator.cpp
 * describes memory management for captures.
 */

std::atomic<int> NPUGraph::pending_event_queries = 0;

// Track any outstanding event queries that could happen e.g., in a NCCL watchdog so that they
// can be resolved before the capture begins. Note that event queries are not allowed during a
// graph capture in the default capture mode.
void NPUGraph::inc_pending_event_queries()
{
    pending_event_queries++;
}

void NPUGraph::dec_pending_event_queries()
{
    TORCH_INTERNAL_ASSERT(pending_event_queries > 0,
                          "Attempted to decrement the number of outstanding events to be queried, but it was <= 0.");
    pending_event_queries--;
}

int NPUGraph::num_pending_event_queries()
{
    return pending_event_queries;
}

NPUGraph::NPUGraph()
    // NPUStreams may not be default-constructed.
    : capture_stream_(c10_npu::getCurrentNPUStream()) {
}

void NPUGraph::capture_begin(MempoolId_t pool, aclmdlRICaptureMode capture_mode)
{
    static const auto _task_queue_enable = c10_npu::option::OptionsManager::GetTaskQueueEnable();
    TORCH_CHECK(_task_queue_enable != 2,
        "Do not support TASK_QUEUE_ENABLE = 2 during NPU graph capture, please "
        "export TASK_QUEUE_ENABLE=1/0.",
        PTA_ERROR(ErrCode::NOT_SUPPORT));

    TORCH_CHECK(!has_graph_exec_,
                "This NPUGraph instance already owns a captured graph. "
                "To capture a new graph, create a new instance.");

    auto stream = c10_npu::getCurrentNPUStream();

    TORCH_CHECK(stream != c10_npu::getDefaultNPUStream(),
                "NPU graphs must be captured on a non-default stream. "
                "(However, after capture, it's ok to replay them on the "
                "default stream.)");

    capture_stream_ = stream;
    capture_dev_ = c10_npu::current_device();

    if (pool.first != 0 || pool.second != 0) {
        // Either value being nonzero means the user supplied a pool to share.
        // But only one should be nonzero.
        // If pool was created by another graph's capture_begin, first should be nonzero.
        // If pool was created by graph_pool_handle, second should be nonzero.
        TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
        mempool_id_ = pool;
    } else {
        // User did not ask us to share a mempool. Create graph pool handle using is_user_created=false.
        // Sets just the first value, to distinguish it from MempoolId_ts created by graph_pool_handle().
        auto mempool = c10_npu::MemPool({}, false);
        mempool_id_ = mempool.id();
        TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
    }

    // Addendum: beginAllocateStreamToPool is now called before cudaStreamBeginCapture to prevent an
    // autograd thread's free() call triggering an invalid cudaEventRecord in the caching allocator
    // due to the capture status being updated _after_ a capture had already started.
    c10_npu::NPUCachingAllocator::beginAllocateToPool(capture_dev_, mempool_id_, [this](aclrtStream stream) {
        aclmdlRICaptureStatus status;
        aclmdlRI model_ri;
        NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureGetInfo(stream, &status, &model_ri));
        return status == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE && model_ri == model_ri_;
    });

    // At this point, any NCCL watchdogs should be aware that we are in capture mode
    // and therefore should not enqueue any additional work that could be event-queried.
    // We still must wait on any existing work that has not been cleaned up.
    while (num_pending_event_queries()) {
        TORCH_WARN_ONCE("Waiting for pending NCCL work to finish before starting graph capture.");
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }

    // cudaStreamCaptureModeGlobal is the most conservative option to
    // prevent potentially unsafe CUDA API calls during capture.
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureBegin(capture_stream_, capture_mode));

    aclmdlRICaptureStatus status;
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureGetInfo(stream, &status, &model_ri_));
    TORCH_INTERNAL_ASSERT(status == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE);
}

void NPUGraph::capture_end()
{
    auto stream = c10_npu::getCurrentNPUStream();

    TORCH_CHECK(stream == capture_stream_,
                "Capture must end on the same stream it began on.");

    aclmdlRI model_ri;
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureEnd(capture_stream_, &model_ri));

    c10_npu::NPUCachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

    TORCH_CHECK(model_ri == model_ri_, "Invalid end capture model id: ", model_ri);

    // In typical graph usage some tensors (e.g. the tensors used for graph IO) are not freed
    // between replays.
    // If Pytorch compiles and runs with a CUDA 11.4+ toolkit, there's a chance the allocator backend
    // is cudaMallocAsync.
    // cudaMallocAsync is generally graph-safe, but if some tensors are not freed between replays,
    // the graph's internal bookkeeping requires that we instantiate with
    // cudaGraphInstantiateFlagAutoFreeOnLaunch. See
    // cudaGraphLaunch
    // cudaGraphInstantiateWithFlags
    has_graph_exec_ = true;

    uint32_t num_graph_nodes = 0;
}

void NPUGraph::replay()
{
    TORCH_CHECK(has_graph_exec_,
                "Called NPUGraph::replay without a preceding successful capture.");

    c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

    // model_ri_ may be replayed in any stream.
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRIExecuteAsync(model_ri_, c10_npu::getCurrentNPUStream()));
}

void NPUGraph::enable_debug_mode()
{
    _npu_graphs_debug = true;
}

void NPUGraph::debug_dump()
{
    if (_npu_graphs_debug) {
        if (has_graph_exec_) {
            TORCH_WARN("DEBUG: calling NPUGraph::debug_dump() for model id ", model_ri_);
            NPU_CHECK_ERROR(c10_npu::acl::AclmdlRIDebugPrint(model_ri_));
        }
    } else {
        TORCH_WARN("NPU Graphs debug not enabled, set with NPUGraph::enable_debug_mode().");
    }
}

void NPUGraph::reset()
{
    // I'd prefer these checks throw exceptions, not print warnings,
    // but the destructor calls reset(), and at least one CI build
    // refuses to compile with a throwing destructor.
    //
    // Instead of calling reset() in the destructor to clean up, I could
    // call reset() in the __del__ method of a thin Python wrapper,
    // in which case reset would be allowed to throw exceptions.
    // But Stackoverflow does not like user-defined __del__.
    // __del__ prevents Graph instances from EVER being garbage collected
    // if they participate in a reference cycle.
    // And exceptions thrown in __del__ only print a warning anyway.
    //
    // Calling reset() in the C++ destructor, with warnings instead of exceptions
    // if calls fail, is the compromise we chose.
    //
    // If capture_begin, the capture, or capture_end failed at some point, this NPUGraph, the generator,
    // and the allocator could end up in all kinds of weird states depending where failure occurred.
    // If the user catches the failure exception in a script, or is running in REPL or (god forbid)
    // a Jupyter notebook, I don't see an easy way for reset() to gracefully fix all such possible error states.
    if (has_graph_exec_) {
        // notifyCaptureDestroy may throw. How should we handle this?
        c10_npu::NPUCachingAllocator::releasePool(capture_dev_, mempool_id_);
        NPU_CHECK_ERROR(c10_npu::acl::AclmdlRIDestroy(model_ri_));
        has_graph_exec_ = false;
    }
}

// Returns an id another graph's capture_begin can use to share the same memory pool as this graph.
MempoolId_t NPUGraph::pool()
{
    TORCH_CHECK(has_graph_exec_,
                "Called NPUGraph::pool() without a preceding successful capture.");
    return mempool_id_;
}

NPUGraph::~NPUGraph()
{
    reset();
}

} // namespace c10_npu
