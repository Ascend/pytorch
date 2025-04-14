#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/util/flat_hash_map.h>

#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace c10_npu {

// Standalone way to get a unique mempool id usable as a pool=... argument
// to CUDAGraph::capture_begin
TORCH_NPU_API MempoolId_t graph_pool_handle();

struct TORCH_NPU_API NPUTaskGroupHandle {
    aclrtTaskGrp task_group;
};

TORCH_NPU_API void graph_task_group_begin(c10_npu::NPUStream stream);
TORCH_NPU_API NPUTaskGroupHandle graph_task_group_end(c10_npu::NPUStream stream);
TORCH_NPU_API void graph_task_update_begin(c10_npu::NPUStream stream, NPUTaskGroupHandle handle);
TORCH_NPU_API void graph_task_update_end(c10_npu::NPUStream stream);

struct TORCH_NPU_API NPUGraph {
    NPUGraph();
    ~NPUGraph();

    static void inc_pending_event_queries();
    static void dec_pending_event_queries();
    static int num_pending_event_queries();

    void capture_begin(
        MempoolId_t pool = {0, 0},
        aclmdlRICaptureMode capture_mode = aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_GLOBAL);
    void capture_end();
    void replay();
    void reset();
    MempoolId_t pool();
    void enable_debug_mode();
    void debug_dump();

protected:
    aclmdlRI model_ri_ = nullptr;

    static std::atomic<int> pending_event_queries;

    // Set to true in capture_end if NPU graph is captured succeeded
    bool has_graph_exec_ = false;

    // the ID assigned by cuda during graph capture,
    // used to identify when a stream is participating in capture
    CaptureId_t capture_id_ = -1;

    // uuid used to request a particular private mempool from CUDACachingAllocator.
    // By default, this will be set to {id_, 0}.
    //
    // If capture_begin is called with "pool=other_graph.pool()", this graph's mempool_id_
    // will be set to the other graph's mempool_id_, and therefore share a mempool with the
    // other graph.
    //
    // If capture_begin is called with "pool=handle" where "handle" came from graph_pool_handle(),
    // it will share a mempool with any other captures that used "pool=handle".
    //
    // Sharing a mempool across graphs saves memory, and it's safe if you
    // know you'll replay those graphs in the same order you captured them.
    MempoolId_t mempool_id_;

    // Stream on which capture began
    NPUStream capture_stream_;

    // Device where capture occurred. Right now, for simplicity, we require all ops
    // in a capture to run on the same device, but this is a limitation of CUDAGraph,
    // not CUDA itself.  We can straightforwardly modify CUDAGraph to support multi-device
    // captures if needed.
    int capture_dev_;
};

} // namespace c10_npu
