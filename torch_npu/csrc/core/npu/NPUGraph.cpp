#include "torch_npu/csrc/core/npu/NPUGraph.h"
#include "torch_npu/csrc/core/npu/NPUAllocatorConfig.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStreamUtils.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "third_party/acl/inc/acl/error_codes/rt_error_codes.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/GetCANNInfo.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

#include <ATen/Functions.h>
#include <ATen/core/CachingHostAllocator.h>

namespace c10_npu {

static bool _npu_graphs_debug = false;
static std::mutex _currently_capturing_graphs_mutex;
static ska::flat_hash_map<CaptureId_t, NPUGraph*> _currently_capturing_graphs;

namespace {
void apply_cache_op_info(aclrtStream stream, bool enabled)
{
    if (!IsGteCANNVersion("8.5.0", "CANN")) {
        return;
    }
    aclrtStreamAttrValue val;
    val.cacheOpInfoSwitch = static_cast<uint32_t>(enabled ? 1u : 0u);
    int32_t ret = c10_npu::acl::AclrtSetStreamAttribute(stream, aclrtStreamAttr::ACL_STREAM_ATTR_CACHE_OP_INFO,
        &val);
    if (ret == ACL_ERROR_RT_PARAM_INVALID) {
        ASCEND_LOGW("Report shape function is disabled due to incompatible CANN version.");
    } else {
        TORCH_CHECK(ret == ACL_RT_SUCCESS, "AclrtSetStreamAttribute failed with error code: ", ret);
    }
}

void begin_allocate_to_pool(
    int capture_dev,
    MempoolId_t mempool_id,
    std::function<bool(aclrtStream)> filter)
{
    c10_npu::NPUCachingAllocator::beginAllocateToPool(capture_dev, mempool_id, filter);
    at::getHostAllocator(at::kPrivateUse1)->begin_allocate_to_pool(
        mempool_id,
        [filter](c10::Stream stream) {
            return filter(c10_npu::NPUStream(c10_npu::NPUStream::UNCHECKED, stream).stream(false));
        });
}
}

MempoolId_t graph_pool_handle()
{
    // Sets just the second value, to distinguish it from MempoolId_ts created from
    // aclmdlRICaptureGetInfo id_s in capture_begin.
    return c10_npu::MemPool::graph_pool_handle();
}

void graph_task_group_begin(c10_npu::NPUStream stream)
{
    c10_npu::detail::checkNotExternalStream(stream, "graph_task_group_begin");
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureTaskGrpBegin(stream));
}

NPUTaskGroupHandle graph_task_group_end(c10_npu::NPUStream stream)
{
    c10_npu::detail::checkNotExternalStream(stream, "graph_task_group_end");
    aclrtTaskGrp group;
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureTaskGrpEnd(stream, &group));
    NPUTaskGroupHandle handle;
    handle.task_group = group;
    return handle;
}

void graph_task_update_begin(c10_npu::NPUStream stream, NPUTaskGroupHandle handle)
{
    c10_npu::detail::checkNotExternalStream(stream, "graph_task_update_begin");
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureTaskUpdateBegin(stream, handle.task_group));
}

void graph_task_update_end(c10_npu::NPUStream stream)
{
    c10_npu::detail::checkNotExternalStream(stream, "graph_task_update_end");
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureTaskUpdateEnd(stream));
}

void super_kernel_scope_begin(const char* scope_name)
{
    auto stream = c10_npu::getCurrentNPUStream();
    c10_npu::detail::checkNotExternalStream(stream, "super_kernel_scope_begin");
    NPU_CHECK_ERROR(c10_npu::skapi::AclskScopeBegin(scope_name, stream));
}

void super_kernel_scope_end(const char* scope_name)
{
    auto stream = c10_npu::getCurrentNPUStream();
    c10_npu::detail::checkNotExternalStream(stream, "super_kernel_scope_end");
    NPU_CHECK_ERROR(c10_npu::skapi::AclskScopeEnd(scope_name, stream));
}

void launch_callback(c10_npu::NPUStream stream, NPUCallbackFunc func, void *fnData)
{
    c10_npu::detail::checkNotExternalStream(stream, "launch_callback");
    aclrtCallbackBlockType type = aclrtCallbackBlockType::ACL_CALLBACK_BLOCK;
    NPU_CHECK_ERROR(c10_npu::acl::AclrtLaunchCallback(func, fnData, type, stream));
}

void launch_host_func(c10_npu::NPUStream stream, NPUCallbackFunc func, void *fnData)
{
    NPU_CHECK_ERROR(c10_npu::acl::AclrtLaunchHostFunc(stream, func, fnData));
}

void subscribe_report(uint64_t threadId, c10_npu::NPUStream stream)
{
    c10_npu::detail::checkNotExternalStream(stream, "subscribe_report");
    NPU_CHECK_ERROR(c10_npu::acl::AclrtSubscribeReport(threadId, stream));
}

void unsubscribe_report(uint64_t threadId, c10_npu::NPUStream stream)
{
    c10_npu::detail::checkNotExternalStream(stream, "unsubscribe_report");
    NPU_CHECK_ERROR(c10_npu::acl::AclrtUnSubscribeReport(threadId, stream));
}

/**
 * Note [NPU Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Q: Why do we need graph capture and launch bindings in Pytorch?
 *    Why can't they live in a user extension, for example?
 *
 * A1: Convenience.
 * A2: To ensure valid numerics on replay, some native NPU ops (like RNG ops with
 *     CPU statefulness) need cooperation from the capture and replay bindings
 *     (see Note [NPU Graph-safe RNG states] in NPUGeneratorImpl.h).
 *
 *     We can't expect users to know about this cooperation.  If users write capture
 *     bindings naively in an extension, they likely won't interact with the native
 *     ops properly.  Their graphs would yield invalid numerics on replay.
 */

/**
 * Note [Interaction with NPU graph capture] in NPUCachingAllocator.cpp
 * describes memory management for captures.
 */

NPUGraph::NPUGraph()
    // NPUStreams may not be default-constructed.
    : capture_stream_(c10_npu::getCurrentNPUStream()) {
}

void NPUGraph::register_generator_state(c10::intrusive_ptr<at_npu::NPUGeneratorState> state)
{
    captured_generator_states_[std::move(state)] = 0;
}

void NPUGraph::register_generator_state(const at::Generator& generator)
{
    c10::intrusive_ptr<at_npu::NPUGeneratorImpl> npu_gen =
        c10::dynamic_intrusive_pointer_cast<at_npu::NPUGeneratorImpl>(generator.getIntrusivePtr());
    npu_gen->register_graph(this);
}

void NPUGraph::capture_begin(MempoolId_t pool, aclmdlRICaptureMode capture_mode, bool report_shape)
{
    NPUGRAPH_LOGD("NPUGRAPH Capture begin");
    static const auto _task_queue_enable = c10_npu::option::OptionsManager::GetTaskQueueEnable();
    TORCH_CHECK(_task_queue_enable != 2,
        "Do not support TASK_QUEUE_ENABLE = 2 during NPU graph capture, please "
        "export TASK_QUEUE_ENABLE=1/0.",
        PTA_ERROR(ErrCode::NOT_SUPPORT));

    TORCH_CHECK(
        !c10_npu::NPUCachingAllocator::NPUAllocatorConfig::pin_memory_expandable_segments(),
        "ACLGraph capture is not supported when pin_memory_expandable_segments=True. "
        "NPUExpandableHostAllocatorImpl overrides allocate/free/empty_cache/record_event "
        "and does not integrate with the base CachingHostAllocator's private pool mechanism, "
        "which would cause capture-time pinned blocks to be incorrectly recycled and lead to "
        "data corruption on graph replay. "
        "Please unset pin_memory_expandable_segments in PYTORCH_NPU_ALLOC_CONF to use ACLGraph "
        "with pin_memory, or avoid pin_memory calls inside capture regions.",
        PTA_ERROR(ErrCode::NOT_SUPPORT));

    TORCH_CHECK(!has_graph_exec_,
                "This NPUGraph instance already owns a captured graph. "
                "To capture a new graph, create a new instance.");

    capture_mode_ = capture_mode;

    auto stream = c10_npu::getCurrentNPUStream();
    c10_npu::detail::checkNotExternalStream(stream, "NPUGraph::capture_begin");

    TORCH_CHECK(stream != c10_npu::getDefaultNPUStream(),
                "NPU graphs must be captured on a non-default stream. "
                "(However, after capture, it's ok to replay them on the "
                "default stream.)");

    apply_cache_op_info(stream, report_shape);

    // default generator is always registered
    auto* gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(c10::nullopt, at_npu::detail::getDefaultNPUGenerator());
    gen->register_graph(this);
    gen->set_secondary_stream_capture_state(false);

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
        mempool_id_ = c10_npu::MemPool::graph_pool_handle(false);
        TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
    }

    auto filter = create_allocate_filter();

    // Addendum: beginAllocateToPool is now called before AclmdlRICaptureBegin to prevent an
    // autograd thread's free() call from interacting with the caching allocator
    // due to the capture status being updated _after_ a capture had already started.
    c10_npu::NPUCachingAllocator::beginAllocateToPool(capture_dev_, mempool_id_, filter);

    // Register host allocator to the same private pool for pin_memory support during capture.
    // Use stream(false) to obtain the raw aclrtStream without flushing the PTA task queue,
    // since AclmdlRICaptureGetInfo only queries stream state and does not require queue drain.
    at::getHostAllocator(at::kPrivateUse1)->begin_allocate_to_pool(
        mempool_id_,
        [filter](c10::Stream stream) {
            return filter(c10_npu::NPUStream(c10_npu::NPUStream::UNCHECKED, stream).stream(false));
        });

    // ACL_MODEL_RI_CAPTURE_MODE_GLOBAL is the most conservative option to
    // prevent potentially unsafe NPU API calls during capture.
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureBegin(capture_stream_, capture_mode));

    aclmdlRICaptureStatus status;
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureGetInfo(stream, &status, &model_ri_));
    TORCH_INTERNAL_ASSERT(status == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE);
    capture_id_ = c10_npu::captureIdFromModelRI(model_ri_);

    {
        std::lock_guard<std::mutex> lock(_currently_capturing_graphs_mutex);
        _currently_capturing_graphs.emplace(capture_id_, this);
    }

    for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
        generator_state->init_capture_state(capture_id_);
    }

    NPUGRAPH_LOGD("NPUGRAPH Capture begin done: model_ri=%p, capture_dev=%d, "
                  "mempool_id=(%lu,%lu), stream=%p, capture_mode=%d",
                  static_cast<void*>(model_ri_), capture_dev_,
                  mempool_id_.first, mempool_id_.second,
                  static_cast<void*>(capture_stream_.stream(false)), static_cast<int>(capture_mode));
}

void NPUGraph::capture_end()
{
    NPUGRAPH_LOGD("NPUGRAPH Capture end");
    auto stream = c10_npu::getCurrentNPUStream();

    c10_npu::detail::checkNotExternalStream(stream, "NPUGraph::capture_end");

    TORCH_CHECK(stream == capture_stream_,
                "Capture must end on the same stream it began on.");

    apply_cache_op_info(stream, false);

    aclmdlRI model_ri;
    auto ret = c10_npu::acl::AclmdlRICaptureEnd(capture_stream_, &model_ri);
    {
        std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
        TORCH_CHECK(
          _currently_capturing_graphs.count(capture_id_),
          "capture_end() called before capture_begin().");
        _currently_capturing_graphs.erase(capture_id_);
    }

    c10_npu::NPUCachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);
    at::getHostAllocator(at::kPrivateUse1)->end_allocate_to_pool(mempool_id_);

    NPU_CHECK_ERROR(ret);
    TORCH_CHECK(model_ri == model_ri_, "Invalid end capture model id: ", model_ri);

    // In typical graph usage some tensors (e.g. the tensors used for graph IO)
    // are not freed between replays. Keep graph-owned allocator bookkeeping
    // alive until reset releases the private pool.
    has_graph_exec_ = true;

    for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
        wholegraph_increments = generator_state->capture_epilogue(capture_id_);
    }

    uint32_t num_graph_nodes = 0;

    NPUGRAPH_LOGD("NPUGRAPH Capture end done: model_ri=%p, has_graph_exec=%d",
                  static_cast<void*>(model_ri_), static_cast<int>(has_graph_exec_));
}

void NPUGraph::replay()
{
    NPUGRAPH_LOGD("NPUGRAPH Replay model_ri=%p, device=%d", static_cast<void*>(model_ri_), capture_dev_);
    TORCH_CHECK(has_graph_exec_,
                "Called NPUGraph::replay without a preceding successful capture.");

    c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

    for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
        generator_state->replay_prologue(capture_id_, wholegraph_increments);
    }

    // model_ri_ may be replayed in any stream.
    auto stream = c10_npu::getCurrentNPUStream();
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRIExecuteAsync(model_ri_, stream));
    if (c10_npu::option::OptionsManager::CheckBlockingEnable() &&
        !c10_npu::detail::isExternalStream(stream)) {
        NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
    }
}

void NPUGraph::enable_debug_mode()
{
    _npu_graphs_debug = true;
}

void NPUGraph::debug_dump(const std::string& debug_path)
{
    NPUGRAPH_LOGD("NPUGRAPH Debug dump to %s", debug_path.c_str());
    if (has_graph_exec_) {
        TORCH_WARN("calling NPUGraph::debug_dump() for model id ", model_ri_);
        NPU_CHECK_ERROR(c10_npu::acl::AclmdlRIDebugJsonPrint(model_ri_, debug_path.c_str(), 1));
    } else {
        TORCH_WARN("Called NPUGraph::debug_dump without a preceding successful capture.");
    }
}

void NPUGraph::super_kernel_optimize(const aclskOptions *options)
{
    TORCH_CHECK(has_graph_exec_,
                "Called NPUGraph::super_kernel_optimize without a preceding successful capture.");
    NPU_CHECK_ERROR(c10_npu::skapi::AclskOptimize(model_ri_, options));
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
    if (capture_id_ != std::numeric_limits<CaptureId_t>::max()) {
        {
            std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
            _currently_capturing_graphs.erase(capture_id_);
        }
        for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
            generator_state->remove_capture_state(capture_id_);
        }
    }

    if (has_graph_exec_) {
        NPUGRAPH_LOGD("NPUGRAPH Reset releasing graph: model_ri=%p, capture_dev=%d, "
                      "mempool_id=(%lu,%lu)",
                      static_cast<void*>(model_ri_), capture_dev_,
                      mempool_id_.first, mempool_id_.second);
        // notifyCaptureDestroy may throw. How should we handle this?
        c10_npu::NPUCachingAllocator::releasePool(capture_dev_, mempool_id_);
        at::getHostAllocator(at::kPrivateUse1)->release_pool(mempool_id_);
        if (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
            NPU_CHECK_WARN(c10_npu::acl::AclmdlRIDestroy(model_ri_));
        }
        has_graph_exec_ = false;
    }
    capture_id_ = std::numeric_limits<CaptureId_t>::max();
}

// Returns an id another graph's capture_begin can use to share the same memory pool as this graph.
MempoolId_t NPUGraph::pool()
{
    TORCH_CHECK(has_graph_exec_,
                "Called NPUGraph::pool() without a preceding successful capture.");
    return mempool_id_;
}

NPUGraph* NPUGraph::get_currently_capturing_graph()
{
    std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
    auto capture_id = c10_npu::currentStreamCaptureIdMayInitCtx();
    TORCH_CHECK(capture_id.has_value(),
                "The current NPU stream is not currently capturing.",
                PTA_ERROR(ErrCode::PARAM));
    TORCH_CHECK(_currently_capturing_graphs.count(capture_id.value()),
                "get_currently_capturing_graph() can be used only between capture_begin() and capture_end(). "
                "Did you use a stream without making it depend upon the original stream used for capture?",
                PTA_ERROR(ErrCode::PARAM));
    return _currently_capturing_graphs.at(capture_id.value());
}

std::function<bool(aclrtStream)> NPUGraph::create_allocate_filter() const
{
    return [this](aclrtStream stream) -> bool {
        auto capturing_id = c10_npu::captureIdMayInitCtx(stream);
        NPUGRAPH_LOGD("NPUGRAPH AllocateToPool filter: stream=%p, capture_id=%llu, expected_capture_id=%llu, "
                      "mempool_id=(%lu,%lu)",
                      static_cast<void*>(stream), capturing_id.value_or(0),
                      capture_id_, mempool_id_.first, mempool_id_.second);
        return capturing_id.has_value() && capturing_id.value() == capture_id_;
    };
}

std::function<bool(aclrtStream)> NPUGraph::create_child_allocate_filter(aclmdlRI child_model_ri) const
{
    TORCH_INTERNAL_ASSERT(child_model_ri != nullptr, "Child modelRI allocate filter requires a non-null modelRI.");
    auto child_capture_id = c10_npu::captureIdFromModelRI(child_model_ri);
    return [this, child_capture_id](aclrtStream stream) -> bool {
        auto capturing_id = c10_npu::captureIdMayInitCtx(stream);
        NPUGRAPH_LOGD("NPUGRAPH Child AllocateToPool filter: stream=%p, capture_id=%llu, "
                      "expected_child_capture_id=%llu, mempool_id=(%lu,%lu)",
                      static_cast<void*>(stream), capturing_id.value_or(0),
                      child_capture_id, mempool_id_.first, mempool_id_.second);
        return capturing_id.has_value() && capturing_id.value() == child_capture_id;
    };
}

void NPUGraph::set_conditional_handle(aclmdlRICondHandle handle, const at::Tensor& scalar_npu_pred_tensor)
{
    TORCH_CHECK(scalar_npu_pred_tensor.device().type() == c10::DeviceType::PrivateUse1,
                "Conditions must be on an npu device to use conditional nodes in npu graphs.",
                PTA_ERROR(ErrCode::PARAM));
    TORCH_CHECK(scalar_npu_pred_tensor.numel() == 1,
                "Conditions must be scalar tensors to use conditional nodes in npu graphs.",
                PTA_ERROR(ErrCode::PARAM));
    TORCH_CHECK(scalar_npu_pred_tensor.scalar_type() == at::kBool,
                "Conditions must be bool tensors to use conditional nodes in npu graphs.",
                PTA_ERROR(ErrCode::TYPE));
    TORCH_CHECK(scalar_npu_pred_tensor.device().index() == capture_dev_,
                "Conditions must be on the same npu device as the graph capture.",
                PTA_ERROR(ErrCode::PARAM));

    uint64_t* cond_ptr = nullptr;
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICondHandleGetCondPtr(handle, &cond_ptr));
    TORCH_CHECK(cond_ptr != nullptr,
                "aclmdlRICondHandleGetCondPtr returned a null condition pointer.",
                PTA_ERROR(ErrCode::PTR));

    auto stream = c10_npu::getCurrentNPUStream();
    // PTA does not support <<<>>> kernel launches here, so initialize the condition handle via memset + memcpy.
    NPU_CHECK_ERROR(c10_npu::acl::AclrtMemSetAsync(
        cond_ptr,
        sizeof(uint64_t),
        0,
        sizeof(uint64_t),
        stream.stream(false)));
    NPU_CHECK_ERROR(aclrtMemcpyAsync(
        cond_ptr,
        sizeof(uint64_t),
        scalar_npu_pred_tensor.const_data_ptr<bool>(),
        sizeof(bool),
        ACL_MEMCPY_DEVICE_TO_DEVICE,
        stream.stream(false)));
}

void NPUGraph::begin_capture_to_if_node(const at::Tensor& scalar_npu_pred_tensor)
{
    TORCH_CHECK(!has_graph_exec_, "This NPUGraph instance already owns a captured graph.",
                PTA_ERROR(ErrCode::PARAM));

    auto parent_stream = c10_npu::getCurrentNPUStream();
    aclmdlRICaptureStatus status;
    aclmdlRI parent_model_ri;
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureGetInfo(parent_stream, &status, &parent_model_ri));
    TORCH_CHECK(status == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE,
                "capture_begin() must be called before begin_capture_to_if_node().",
                PTA_ERROR(ErrCode::PARAM));
    TORCH_CHECK(parent_model_ri == (conditional_model_ri_stack_.empty() ? model_ri_ : conditional_model_ri_stack_.top()),
                "Conditional capture must be started from the current NPUGraph capture stream.",
                PTA_ERROR(ErrCode::PARAM));

    aclmdlRICondHandle handle = nullptr;
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICondHandleCreate(
        parent_model_ri, 0, ACL_MODEL_RI_COND_HANDLE_ASSIGN_DEFAULT, &handle));
    set_conditional_handle(handle, scalar_npu_pred_tensor);

    aclmdlRI child_model_ri = nullptr;
    aclmdlRICondTaskParams params;
    params.handle = handle;
    params.type = ACL_MODEL_RI_COND_TYPE_IF;
    params.size = 1;
    params.modelRIArray = &child_model_ri;
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRIAddCondTask(params, parent_stream, 0));
    TORCH_CHECK(child_model_ri != nullptr, "aclmdlRIAddCondTask did not return a child modelRI.",
                PTA_ERROR(ErrCode::PTR));
    NPUGRAPH_LOGD("NPUGRAPH Conditional begin: parent_model_ri=%p, child_model_ri=%p, "
                  "parent_stream=%p, mempool_id=(%lu,%lu)",
                  static_cast<void*>(parent_model_ri), static_cast<void*>(child_model_ri),
                  static_cast<void*>(parent_stream.stream(false)), mempool_id_.first, mempool_id_.second);

    // AclmdlRIAddCondTask returns the child modelRI before child capture begins,
    // so the allocator filter can target the child graph pool before any branch
    // allocations are made.
    c10_npu::NPUCachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);
    at::getHostAllocator(at::kPrivateUse1)->end_allocate_to_pool(mempool_id_);
    begin_allocate_to_pool(capture_dev_, mempool_id_, create_child_allocate_filter(child_model_ri));

    auto child_stream = c10_npu::getStreamFromPool(false, capture_dev_);
    NPUGRAPH_LOGD("NPUGRAPH Conditional capture to child modelRI: child_model_ri=%p, child_stream=%p, "
                  "mempool_id=(%lu,%lu)",
                  static_cast<void*>(child_model_ri), static_cast<void*>(child_stream.stream(false)),
                  mempool_id_.first, mempool_id_.second);
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureToModelRIBegin(child_stream, child_model_ri, capture_mode_));
    aclmdlRICaptureStatus child_status;
    aclmdlRI capturing_child_model_ri;
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureGetInfo(child_stream, &child_status, &capturing_child_model_ri));
    TORCH_INTERNAL_ASSERT(child_status == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE,
                          "Child stream should be actively capturing after AclmdlRICaptureToModelRIBegin.");
    TORCH_INTERNAL_ASSERT(capturing_child_model_ri == child_model_ri,
                          "Child stream should capture to the conditional node child modelRI.");
    conditional_model_ri_stack_.push(child_model_ri);
    conditional_node_streams_.emplace(child_stream);

    {
        std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
        auto child_capture_id = c10_npu::captureIdFromModelRI(child_model_ri);
        _currently_capturing_graphs.emplace(child_capture_id, this);
    }
}

void NPUGraph::end_capture_to_conditional_node()
{
    TORCH_INTERNAL_ASSERT(!conditional_model_ri_stack_.empty(),
                          "Missing modelRI for conditional node.");
    TORCH_INTERNAL_ASSERT(!conditional_node_streams_.empty(),
                          "Missing stream for conditional node.");

    aclmdlRI child_model_ri = conditional_model_ri_stack_.top();
    auto child_capture_id = c10_npu::captureIdFromModelRI(child_model_ri);
    {
        std::unique_lock<std::mutex> lock(_currently_capturing_graphs_mutex);
        TORCH_CHECK(_currently_capturing_graphs.count(child_capture_id),
                    "capture_end() called before capture_begin().",
                    PTA_ERROR(ErrCode::PARAM));
        _currently_capturing_graphs.erase(child_capture_id);
    }

    auto stream = conditional_node_streams_.top().current_stream();
    aclmdlRI ended_model_ri = nullptr;
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureEnd(stream, &ended_model_ri));
    TORCH_CHECK(ended_model_ri == child_model_ri,
                "Invalid conditional capture end modelRI.",
                PTA_ERROR(ErrCode::PARAM));
    NPUGRAPH_LOGD("NPUGRAPH Conditional end: child_model_ri=%p, stream=%p, mempool_id=(%lu,%lu)",
                  static_cast<void*>(child_model_ri), static_cast<void*>(stream.stream(false)),
                  mempool_id_.first, mempool_id_.second);

    conditional_node_streams_.pop();
    conditional_model_ri_stack_.pop();

    c10_npu::NPUCachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);
    at::getHostAllocator(at::kPrivateUse1)->end_allocate_to_pool(mempool_id_);
    if (conditional_model_ri_stack_.empty()) {
        NPUGRAPH_LOGD("NPUGRAPH Conditional restore root allocate filter: model_ri=%p, mempool_id=(%lu,%lu)",
                      static_cast<void*>(model_ri_), mempool_id_.first, mempool_id_.second);
        begin_allocate_to_pool(capture_dev_, mempool_id_, create_allocate_filter());
    } else {
        NPUGRAPH_LOGD("NPUGRAPH Conditional restore parent allocate filter: parent_model_ri=%p, mempool_id=(%lu,%lu)",
                      static_cast<void*>(conditional_model_ri_stack_.top()), mempool_id_.first, mempool_id_.second);
        begin_allocate_to_pool(capture_dev_, mempool_id_, create_child_allocate_filter(conditional_model_ri_stack_.top()));
    }

    bool rng_or_generators_changed = false;
    for (const auto& [generator_state, wholegraph_increment] : captured_generator_states_) {
        if (generator_state->get_capture_state(child_capture_id) != nullptr) {
            rng_or_generators_changed = true;
            break;
        }
    }
    TORCH_CHECK(!rng_or_generators_changed,
                "RNG within data-dependent conditional nodes is not supported yet.",
                PTA_ERROR(ErrCode::NOT_SUPPORT));
}

NPUGraph::~NPUGraph()
{
    reset();
}

} // namespace c10_npu
