#pragma once

#include <mutex>
#include <thread>
#include <unordered_map>
#include <variant>
#include <future>
#include <atomic>
#include <string>
#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Utils.hpp>
#include <c10d/Work.hpp>

#include "torch_npu/csrc/distributed/LCCLUtils.hpp"
#include "torch_npu/csrc/npu/Event.h"


namespace c10d_npu {

const std::string LCCL_BACKEND_NAME = "lccl";

class ProcessGroupLCCL : public c10d::Backend {
public:
    class WorkLCCL : public c10d::Work, public std::enable_shared_from_this<WorkLCCL> {
    public:
        // Constructor takes a list of NPU devices to adapt framework, But LCCL support one device only!!!
        explicit WorkLCCL(const std::vector<at::Device>& devices);

        ~WorkLCCL() override;
        // Checks if request has completed. In this specific case of LCCL, it checks
        // if the LCCL operation has completed on the NPU in its own LCCL stream.
        // Non-blocking operation.
        bool isCompleted() override;

        bool isSuccess() const override;

        bool wait(std::chrono::milliseconds timeout) override;

        // Let current stream wait on the completing of the LCCL work
        // Throws on exceptions. Blocking operation, which will wait for work completion.
        void synchronize() override;

        // Helper function that checks if the LCCL have finished execution on the NPUs
        bool finishedNPUExecution();
        std::vector<at::Tensor> result() override;

    protected:
        // The cached list of NPU devices to operate on. LCCL support one device per rank only
        std::vector<at::Device> devices_;

        // The LCCL communicators used for this work item.
        std::vector<at_npu::lccl::LcclComm> lcclComms_;

        // multiple runtime devices. These start npu events are needed by desync debugging if enabled.
        std::shared_ptr<std::vector<c10_npu::NPUEvent>> lcclStartEvents_;

        // The end npu events of LCCL operator tracking this work item on multiple npu devices.
        std::shared_ptr<std::vector<c10_npu::NPUEvent>> lcclEndEvents_;

        // Clone of blockingWait_ from ProcessGroupLCCL.
        bool blockingWait_ = false;

        // Clone of opTimeout_ from ProcessGroupLCCL.
        std::chrono::milliseconds opTimeout_;

        // Time point representing when the work started.
        std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

    private:
        // Helper function for synchronize
        void synchronizeInternal(std::chrono::milliseconds timeout);

        // Checks for LCCL errors and sets an appropriate exception_ptr.
        void checkAndSetException() const;

        // Checks for LCCL errors and throws an appropriate exception.
        void checkAndThrowException() const;

        // Just checks whether NPU execution has completed, without modifying
        // exception_ptr.
        bool finishedNPUExecutionInternal() const;

        // Get a Future object that will be marked as completed internally.
        c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

        // Store a reference to LCCL collective's outputs, used by result and to
        // give a more descriptive message when representing the Work as a string.
        std::shared_ptr<std::vector<at::Tensor>> outputs_;

        // Reference to the store so that we can write aborted communicators to the store.
        c10::intrusive_ptr<c10d::Store> store_;

        // The future returned by getFuture.
        c10::intrusive_ptr<at::ivalue::Future> future_;

        std::vector<at::Tensor> lazy_destroy_tensors_;
        friend class ProcessGroupLCCL;
    };

    ProcessGroupLCCL(
        const c10::intrusive_ptr<c10d::Store>& store,
        int rank,
        int size);

    ~ProcessGroupLCCL() override;

    const std::string getBackendName() const override
    {
        return LCCL_BACKEND_NAME;
    }

    c10::intrusive_ptr<c10d::Work> allreduce(
        std::vector<at::Tensor>& tensors,
        const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;

    c10::intrusive_ptr<c10d::Work> allgather(
        std::vector<std::vector<at::Tensor>>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

    c10::intrusive_ptr<c10d::Work> broadcast(
        std::vector<at::Tensor>& tensors,
        const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;

    c10::intrusive_ptr<c10d::Work> reduce_scatter(
        std::vector<at::Tensor>& outputTensors,
        std::vector<std::vector<at::Tensor>>& inputTensors,
        const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

    static const int64_t kProcessGroupLCCLOpTimeoutMillis;

protected:
    // Helper that either looks up the cached LCCL communicators or creates
    // a new set of LCCL communicators as a cache entry
    std::vector<at_npu::lccl::LcclComm>& getLCCLComm(
        const std::string& devicesKey,
        const std::vector<at::Device>& devices);

    c10::intrusive_ptr<c10d::Store> store_;

    // Whether or not wait() and synchronize() are blocking operations that wait
    // for the operation to complete.
    bool blockingWait_ = false;

    // Timeout for operations. This is only used when blockingWait_ is enabled.
    std::chrono::milliseconds opTimeout_;

    // The NPU streams used by LCCL kernels
    std::unordered_map<std::string, std::vector<c10_npu::NPUStream>> lcclStreams_;
    std::unordered_map<std::string, std::vector<at_npu::lccl::LcclComm>> devLCCLCommMap_;
    // The NPU events used to sync LCCL streams
    std::unordered_map<std::string, std::vector<c10_npu::NPUEvent>> lcclEvents_;
    // Mutex to guard maps like devLCCLCommMap_.
    std::mutex mutex_;
private:
    template <typename Fn>
    c10::intrusive_ptr<c10d::Work> collective(
        std::vector<at::Tensor>& input,
        std::vector<at::Tensor>& output,
        Fn fn,
        c10d::OpType opType);

    template <typename Fn, typename PreProcess, typename PostProcess>
    c10::intrusive_ptr<c10d::Work> collective(
        std::vector<at::Tensor>& input,
        std::vector<at::Tensor>& output,
        Fn fn,
        PreProcess pre,
        PostProcess post,
        c10d::OpType opType);
};

} // namespace c10d_npu
