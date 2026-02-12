#pragma once

#include <mutex>
#include <thread>
#include <unordered_map>
#include <variant>
#include <future>
#include <atomic>
#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Utils.hpp>
#include <c10d/Work.hpp>

#include "third_party/hccl/inc/hccl/hccl.h"
#include "torch_npu/csrc/core/npu/interface/HcclInterface.h"
#include "torch_npu/csrc/distributed/HCCLUtils.hpp"
#include "torch_npu/csrc/npu/Event.h"


namespace c10d_npu {
// Environment variable which controls whether or not wait() is blocking or
// non-blocking.
constexpr const char* HCCL_BLOCKING_WAIT = "HCCL_BLOCKING_WAIT";
constexpr const char* HCCL_BACKEND_NAME = "hccl";

constexpr const char* EXCEPTION_DUMP = "exception_dump";

// Environment variable which controls whether or not we perform Async Error
// Handling with HCCL.
constexpr const char* HCCL_ASYNC_ERROR_HANDLING = "HCCL_ASYNC_ERROR_HANDLING";

// Environment Variable to control whether Desync Debug is enabled.
// This variable must be set together with HCCL_ASYNC_ERROR_HANDLING.
constexpr const char* HCCL_DESYNC_DEBUG = "HCCL_DESYNC_DEBUG";

constexpr const int DEFAULT_TIMEOUT = 30 * 60 * 1000;

// Control whether dumping debug info on watchdog
// timeout is enabled. This variable must be set together with
// TORCH_HCCL_ENABLE_MONITORING=1 and TORCH_HCCL_TRACE_BUFFER_SIZE > 0.
static std::vector<std::string> TORCH_HCCL_DUMP_ON_TIMEOUT = {
    "TORCH_HCCL_DUMP_ON_TIMEOUT"};

// Enable monitoring thread which aborts the process when the ProcessGroupHCCL
// Watchdog thread gets stuck and no heartbeat is detected after
// TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC. This can happen due to calling CANN/HCCL
// APIs that may hang. It is Useful to prevent jobs being stuck for a prolonged
// time than necessary tying up cluster resources.
static std::vector<std::string> TORCH_HCCL_ENABLE_MONITORING = {
    "TORCH_HCCL_ENABLE_MONITORING"};

// The maximum number of events we store in the flight recorder's ring buffer.
// (One event could be the start or end of a collective, for example).
static std::vector<std::string> TORCH_HCCL_TRACE_BUFFER_SIZE = {
    "TORCH_HCCL_TRACE_BUFFER_SIZE"};

// Control how much extra time we will wait for dumping the debugging info
// before we exit and throws timeout exception.
static std::vector<std::string> TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC = {
    "TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC"};

// Control the watchdog heartbeat timeout period after which the monitoring
// thread will abort the process.
static std::vector<std::string> TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC = {
    "TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC"};

// Control the interval inside the watchdog thread to check the coordinated
// signal from other ranks, e.g. to dump the debugging information.
static std::vector<std::string> TORCH_HCCL_COORD_CHECK_MILSEC = {
    "TORCH_HCCL_COORD_CHECK_MILSEC"};

// Control whether to always use high priority streams
static std::vector<std::string> TORCH_HCCL_HIGH_PRIORITY = {
    "TORCH_HCCL_HIGH_PRIORITY"};

// A struct to hold the latest status of the process group.
struct ProcessGroupStatus {
    // the sequential number of the last collective enqueued into workMetaList_
    // This is useful for indentifying a rank that has not join a collective
    // initialized to be -1 to indicate no collective has been enqueued
    int64_t lastEnqueuedSeq{-1};
    // the sequential number of the last collective started as the kernel
    int64_t lastStartedSeq{-1};
    // the sequential number of the last colletive completed marked by
    // the watchdog thread
    // initialized to be -1 to indicate no collective has been completed
    int64_t lastCompletedSeq{-1};

    // the name of the last collective enqueued into workMetaList_
    std::string lastEnqueuedWorkName;
    // the name of the last collective started as the kernel
    std::string lastStartedWorkName;
    // the name of the last collective completed
    std::string lastCompletedWorkName;

    // the sizes of the last work enqueued
    size_t lastEnqueuedNumelIn;
    size_t lastEnqueuedNumelOut;
    // the sizes of the last work completed
    size_t lastCompletedNumelIn;
    size_t lastCompletedNumelOut;
    // the sizes of the last work started
    size_t lastStartedNumelIn;
    size_t lastStartedNumelOut;
};

struct DumpPipe {
    DumpPipe(int rank)
    {
        std::string fileStem = c10d::getCvarString({"TORCH_HCCL_DEBUG_INFO_PIPE_FILE"}, "");
        if (fileStem.empty() || c10d::getCvarInt({"TORCH_HCCL_TRACE_BUFFER_SIZE"}, 0) <= 0) {
            return;
        }
        TORCH_CHECK(!fileStem.empty(), "TORCH_HCCL_DEBUG_INFO_TEMP_FILE is empty");
        std::string filename = c10::str(fileStem, rank, ".pipe");
        TORCH_CHECK(
            unlink(filename.c_str()) != -1 || errno == ENOENT,
            "Error removing existing named pipe ",
            filename);
        TORCH_CHECK(
            mkfifo(filename.c_str(), 0666) != -1,
            "Error creating named pipe ",
            filename);
        fd_ = open(filename.c_str(), O_RDONLY | O_NONBLOCK);
        LOG(INFO) << "Pipe file " << filename
                  << " has been opened, write to it to trigger HCCL Debug Dump.";
        TORCH_CHECK(fd_ != -1, "Error opening named pipe ", filename);
    }
    bool shouldDump()
    {
        if (fd_ == -1) {
            return false;
        }
        char buf[128];
        // non-blocking from O_NONBLOCK above.
        // Ignore EINTR because we already will poll this
        // again later.
        ssize_t bytesRead = read(fd_, &buf, 128);
        return bytesRead > 0;
    }
    ~DumpPipe()
    {
        if (fd_ != -1) {
            close(fd_);
        }
    }

private:
    int fd_ = -1;
};

// A shelf for stashing tensors between op call and `work.wait()`.
// Used in case of async ops.
class TensorShelf {
public:
    // Stash tensors so that CachingAllocator cannot recycle them prematurely.
    void stash(std::vector<at::Tensor>& tensors);
    // Stash tensors from another shelf.
    void stash(TensorShelf& other);
    // Stash a single tensor.
    void stash(const at::Tensor& tensor);
    // Unstage the stashed tensors so that CachingAllocator can recycle them.
    // Same as `clear()`.
    void unstash();
    // Whether shelf is empty.
    bool empty();
    // Clear the shelf.
    void clear();

protected:
    // Get the inner tensor vector. Use with caution as it is not protected by
    // mutex.
    std::vector<at::Tensor>& get();

private:
    std::vector<at::Tensor> tVector_;
    // Need a mutex to protect `tVector_` because it can be potentially accessed
    // from both main thread and watchdog thread.
    std::mutex mutex_;
};

// NoHandling: do not handle asynchronous HCCL errors
// TearDown: tear down process upon error, see `WorkHCCL::handleException`
// CleanUpOnly: just clean up collectives and abort communicators without
// tearing down process SkipCleanUp: (this is a temporary option and can be
// removed in future) tear down process without cleaning up HCCL communicators.
// This should be used as a last resort in case `hcclCommAbort` itself is
// hanging
enum ErrorHandlingMode {
    NoHandling = 0,
    TearDown = 1,
    CleanUpOnly = 2,
    SkipCleanUp = 3
};

enum class HcclCommType {
    DEFAULT = 0,
    P2P = 1
};

enum class WatchdogStatus {
    INIT = 0,
    RUN = 1,
    STOP = 2
};

#define SHOULD_CLEAN_UP(a) ((a) != NoHandling && (a) != SkipCleanUp)

#define SHOULD_TEAR_DOWN(a) ((a) != NoHandling && (a) != CleanUpOnly)

// ProcessGroupHCCL implements HCCL bindings for c10d.
//
// All functions of the class are expected to be called in the same order
// across all processes in the process group.  This is the only way that we
// can guarantee to match up the same calls among all processes.
//
// All HCCL functions provided by this class are asynchronous functions. More
// specifically, each HCCL call is scheduled on a separate runtime stream that
// is different from the current runtime stream. This is for the purpose of
// achieving potentially concurrency and better performance. As a result,
// it is the callers' responsibilty to make sure that the runtime stream their
// code works on needs to wait for the HCCL operation from
// this class.
//
// This can be done by calling:
//
// either WorkHCCL::wait() or WorkHCCL::synchronize(), both achieves the same
// functionality and are synonyms.
//
// Also note that WorkHCCL::finishedGPUExecution() is a helper function only
// provided by ProcessGroupHCCL to check if the HCCL operation of WorkHCCL has
// finished execution on the NPU (not just scheduled).
//
// Example on using the HCCL process group, use ProcessGroupHCCL pg(store, rank, size) to create,
// and use std::shared_ptr<WorkHCCL> work = pg.allreduce(tensors) to start a work.
//
//   // At this point, HCCL kernel has already by queued successfully
//   // Now, let current stream wait for the HCCL to finish, this function is
//   // async operation as well
//
//   work->wait()
//
//   // Now continue on other work in the current stream.

class C10_NPU_API ProcessGroupHCCL : public c10d::Backend {
public:
    class WorkHCCL : public c10d::Work, public std::enable_shared_from_this<WorkHCCL> {
    public:
        // Constructor takes a list of NPU devices to adapt framework
        // But HCCL support one device only!!!
        explicit WorkHCCL(
            const std::vector<at::Device>& devices,
            int rank,
            c10d::OpType opType,
            uint64_t seq,
            bool desyncDebug);

        WorkHCCL(const WorkHCCL& w);

        WorkHCCL& operator=(const WorkHCCL& w) = default;

        ~WorkHCCL() override;

        // Checks if the HCCL kernel has started to execute.
        bool isStarted(ErrorHandlingMode errorHandling);

        std::shared_ptr<bool> is_dispatched = std::make_shared<bool>(false);
        bool is_reported = false;

        bool is_dumped = false;

        // Checks if request has completed. In this specific case of HCCL, it checks
        // if the HCCL operation has completed on the NPU in its own HCCL stream.
        // Non-blocking operation.
        bool isCompleted() override;

        bool isSuccess() const override;

        // Same as calling synchronize() for HCCL work.
        bool wait(std::chrono::milliseconds timeout) override;

        void abort() override;

        // Let current stream wait on the completing of the HCCL work
        // Throws on exceptions. Blocking operation, which will wait for work
        // completion.
        void synchronize() override;

        // Helper function to handle exception (throw if needed).
        void handleException(ErrorHandlingMode asyncErrorHandling);

        // Helper function that checks if the HCCL have finished
        // execution on the NPUs
        bool finishedNPUExecution();
        std::vector<at::Tensor> result() override;

        // Extend tensors lifecycle to work.synchronize, the tensors is local
        // variable and recordStream.
        void lazyDestroy(std::vector<at::Tensor> tensors);

        // Helper function that sets an exception_ptr on the WorkHCCL object.
        void setException(std::exception_ptr exception_ptr);

        // Helper function that returns True if the WorkHCCL object has timed out
        // and False otherwise.
        // In case of timeout, set exception on the WorkHCCL object.
        bool checkTimeout(c10::optional<std::chrono::milliseconds> timeout = c10::nullopt);

        void checkDispatch();

        bool checkExec();

    protected:
        // The cached list of NPU devices to operate on.
        // HCCL support one device per rank only
        std::vector<at::Device> devices_;

        // The HCCL communicators used for this work item.
        std::vector<std::shared_ptr<HCCLComm>> hcclComms_;

        // // The HCCL communicators used for this work item.
        // std::vector<std::shared_ptr<HCCLComm>> hcclComms_;
        // The HCCL communicators used for this work item. on
        // multiple runtime devices. These start npu events are needed by desync
        // debugging if enabled.
        std::shared_ptr<std::vector<c10_npu::NPUEvent>> hcclStartEvents_;

        // The end npu events of HCCL operator tracking this work item on
        // multiple npu devices.
        std::shared_ptr<std::vector<c10_npu::NPUEvent>> hcclEndEvents_;

        // Tensors used for barrier op
        std::vector<at::Tensor> barrierTensors_;

        // Clone of blockingWait_ from ProcessGroupHCCL.
        bool blockingWait_ = false;

        // Clone of opTimeout_ from ProcessGroupHCCL.
        std::chrono::milliseconds opTimeout_;

        // Time point representing when the work started.
        std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

        // Record the collective sequential number.
        uint64_t seq_{0};

        // Indicates if the hccl start event has been updated to the store trace.
        // This will be used by desync debug.
        bool startTraceUpdated_{false};

        // Record collective sizes for debug. We only record the size on the first
        // device as multi-device per process is deprecated
        size_t numelIn_ = -1;
        size_t numelOut_ = -1;

        // Wrapper method for the static checkForHCCLErrors which can be overridden
        // for tests.
        virtual std::exception_ptr checkForHCCLErrors(
            const std::vector<std::shared_ptr<HCCLComm>>& hcclComms) const;
        
        friend std::ostream& operator<<(
        std::ostream& output,
        const WorkHCCL& workHCCL);

        // AVOID_RECORD_STREAMS implementation helper.
        // Stores references to participating non-output tensors (ie inputs,
        // flattened intermediates).
        // We'll clear this list in synchronizeInternal, just after user-facing
        // stream(s) are synced with the hccl work stream(s).
        // For in-place collectives, some refs stashed here may alias outputs_,
        // but that doesn't do any harm.
        std::shared_ptr<TensorShelf> stashed_for_allocator_safety_;

    private:
        // Helper function for synchronize
        void synchronizeInternal(std::chrono::milliseconds timeout);

        // Checks for HCCL errors and sets an appropriate exception_ptr.
        void checkAndSetException();

        // Checks for HCCL errors and throws an appropriate exception.
        void checkAndThrowException();

        // Just checks whether NPU execution has started, without modifying
        // exception_ptr.
        bool startedNPUExecutionInternal(ErrorHandlingMode errorHandling) const;

        // Just checks whether NPU execution has completed, without modifying
        // exception_ptr.
        bool finishedNPUExecutionInternal() const;

        // Get a Future object that will be marked as completed internally.
        c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

        // Store a reference to HCCL collective's outputs, used by result and to
        // give a more descriptive message when representing the Work as a string.
        std::shared_ptr<std::vector<at::Tensor>> outputs_;

        // Reference to the store so that we can write aborted communicators
        // to the store.
        c10::intrusive_ptr<c10d::Store> store_;

        // The future returned by getFuture.
        c10::intrusive_ptr<at::ivalue::Future> future_;

        // save inputs for tensor free when WorkHCCL::wait
        std::vector<std::pair<c10::weak_intrusive_ptr<c10::StorageImpl>, c10_npu::NPUStream>> recorded_inputs_;
        std::vector<void*> recorded_block_ptr_for_inputs_;
        std::vector<std::pair<c10::weak_intrusive_ptr<c10::StorageImpl>, c10_npu::NPUStream>> recorded_outputs_;

        std::vector<at::Tensor> lazy_destroy_tensors_;
		
        // unique id used to tell the trace buffer that this
        // work has completed
        c10::optional<uint64_t> trace_id_;
        
        mutable std::once_flag print_flag;

        friend class ProcessGroupHCCL;
    };
    struct Options : c10d::Backend::Options {
        explicit Options(bool is_high_priority_stream = false);

        // return intrusive_ptr of the object
        static c10::intrusive_ptr<Options> create(
            bool _is_high_priority_stream = false,
            std::chrono::milliseconds timeout = kNoTimeout)
        {
            return c10::make_intrusive<Options>(_is_high_priority_stream);
        }

        std::unordered_map<std::string, std::variant<uint32_t, uint64_t, int32_t, std::string>> hccl_config;

        std::chrono::milliseconds opTimeout;
        // Schedule HCCL operations on high priority CUDA streams
        bool is_high_priority_stream;

        std::vector<uint32_t> global_ranks_in_group;

        std::string group_id;
    };

    // Class that runs as a side thread to check whether the HCCL collective
    // is timed out or errors on the cached HCCL communicators.
    class Watchdog {
    public:
        Watchdog(ProcessGroupHCCL* pg);
        virtual ~Watchdog() = default;

        // Start the watchdog thread.
        void start();

        // Join the watchdog thread.
        void join();

        // Function that runs as part of a separate thread and checks for errors on
        // HCCL communicators. We need a separate thread to check for HCCL errors
        // since we can't rely on the user calling certain methods like wait(),
        // isCompleted() etc. to detect and remediate errors. In addition to this,
        // we need a mechanism to safely abort and remove HCCL communicators from
        // our cache. This can be done cleanly by having a thread for the
        // ProcessGroupHCCL class. Attempting to modify the communicator cache from
        // the WorkHCCL class might run into issues with object lifetime since the
        // ProcessGroupHCCL object might get destroyed before the WorkHCCL object.
        void run();

        // Watchdog's inside loop.
        // Takes care of cleaning up completed work, and aborting upon failure or
        // timeout.
        void runLoop();

        // Notify the loop inside watchdog.
        void notify();

        void checkAndSetRemoteError();

        // A helper function to get the src rank of a signal from the Store. This is
        // nonblocking function returning -1 if the signal is not available yet.
        int getSignalSrcRank(
            c10::intrusive_ptr<c10d::Store>& store,
            const std::string& signal);

        uint64_t getHeartbt() const;

        void setDesyncDebug(bool desyncDebug);

    private:
        std::thread hcclCommWatchdogThread_;

        // We need to keep a reference to the PG instance so that we can access
        // the member functions of the PG instance. We store a raw pointer on
        // purpose because the watchdog thread now still lives within the
        // lifetime of the PG instance.
        ProcessGroupHCCL* pg_;

        // cached rank for logging inside Watchdog
        int rank_ = -1;

        std::exception_ptr watchDogException_ = nullptr;

        // Condition Variable for watchdog thread sleep
        std::condition_variable workMetaListCV_;

        // Heartbeat of watchdog thread.
        std::atomic_uint64_t heartbeat_{};

        // Whether or not to propagate detected errors to all ranks in the same PG
        // through TCPStore. Check for the variable propagatePgError_

        // Whether or not to enable timeout root cause analysis.
        bool desyncDebug_;
    };

    // If you wish to create multiple process groups, each with a potentially
    // different rank and size, you can do so by passing a new store instance
    // to each one. If you have only a single store object, you can
    // use the `c10d::PrefixStore` to derive scoped instances.
    // This is also what the Python API in torch.distributed does.

    // The process group instance keeps a reference to the store because
    // it may be used long after the constructor runs. In fact, the constructor
    // doesn't create any HCCL communicators. A single HCCL communicator can
    // only be used on a specific set of devices, and are therefore created
    // on-demand when a collective runs. If another collective is executed later,
    // against a different set of devices, the process group creates another HCCL
    // communicator. These HCCL communicators are cached and reused if possible.
    ProcessGroupHCCL(
        const c10::intrusive_ptr<c10d::Store>& store,
        int rank,
        int size,
        c10::intrusive_ptr<Options> options = Options::create());

    // This constructor includes the deprecated `groupName` argument.
    // If you have existing code that uses the `groupName`, you can replace
    // it by specifying a `c10d::PrefixStore(groupName, store)` for store.
    C10_DEPRECATED ProcessGroupHCCL(
        const c10::intrusive_ptr<c10d::Store>& store,
        int rank,
        int size,
        const std::string& groupName,
        c10::intrusive_ptr<Options> options = Options::create())
        : ProcessGroupHCCL(store, rank, size, options) {}

    ~ProcessGroupHCCL() override;

    c10::intrusive_ptr<Options> getOptions()
    {
        return options_;
    }

    const std::string getBackendName() const override
    {
        return std::string(HCCL_BACKEND_NAME);
    }
    
    bool supportsCoalescing() const override
    {
        return true;
    }

    void startCoalescing() override;

    c10::intrusive_ptr<c10d::Work> endCoalescing() override;

    // For specifying a composite optype, such as ALLGATHER and REDUCE_SCATTER
    c10::intrusive_ptr<c10d::Work> endCoalescing(c10d::OpType optype);

    c10::intrusive_ptr<c10d::Work> broadcast(
        std::vector<at::Tensor>& tensors,
        const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;

    c10::intrusive_ptr<c10d::Work> allreduce(
        std::vector<at::Tensor>& tensors,
        const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;

    c10::intrusive_ptr<c10d::Work> allreduce_coalesced(
        std::vector<at::Tensor>& tensors,
        const c10d::AllreduceCoalescedOptions& opts =
            c10d::AllreduceCoalescedOptions()) override;

    c10::intrusive_ptr<c10d::Work> reduce(
        std::vector<at::Tensor>& tensors,
        const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override;

    c10::intrusive_ptr<c10d::Work> _reduce_oop(
        at::Tensor& outputTensors,
        at::Tensor& inputTensors,
        const c10d::ReduceOptions& opts = c10d::ReduceOptions());

    c10::intrusive_ptr<c10d::Work> batch_isend_irecv(
	    std::vector<std::string>& op_type,
	    std::vector<at::Tensor>& tensors,
	    std::vector<uint32_t> remote_rank_list);

    at::Tensor byte_alignment(at::Tensor& tensors) const;

    c10::intrusive_ptr<c10d::Work> _reduce_scatter_base_uneven(
        at::Tensor& outputTensor,
        at::Tensor& inputTensor,
        std::vector<int64_t>& inputSplitSizes,
        const c10d::ReduceScatterOptions& opts);

    c10::intrusive_ptr<c10d::Work> _allgather_base_uneven(
        at::Tensor& outputTensor,
        at::Tensor& inputTensor,
        std::vector<int64_t>& outputSplitSizes,
        const c10d::AllgatherOptions& opts);

    c10::intrusive_ptr<c10d::Work> allgather(
        std::vector<std::vector<at::Tensor>>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

    c10::intrusive_ptr<c10d::Work> allgather_togather(
        std::vector<at::Tensor>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const c10d::AllgatherOptions& opts = c10d::AllgatherOptions());

    c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced(
        std::vector<at::Tensor>& outputs,
        std::vector<at::Tensor>& inputs,
        const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

    c10::intrusive_ptr<c10d::Work> _allgather_base(
        at::Tensor& outputbuffer,
        at::Tensor& inputbuffer,
        const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

    c10::intrusive_ptr<c10d::Work> reduce_scatter(
        std::vector<at::Tensor>& outputTensors,
        std::vector<std::vector<at::Tensor>>& inputTensors,
        const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

    c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(
        at::Tensor& outputTensor,
        at::Tensor& inputTensor,
        const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

    c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced(
        std::vector<at::Tensor>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

    c10::intrusive_ptr<c10d::Work> barrier(
        const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;

    // Unsupported Ops
    c10::intrusive_ptr<c10d::Work> gather(
        std::vector<std::vector<at::Tensor>>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const c10d::GatherOptions& opts = c10d::GatherOptions()) override;

    c10::intrusive_ptr<c10d::Work> scatter(
        std::vector<at::Tensor>& outputTensors,
        std::vector<std::vector<at::Tensor>>& inputTensors,
        const c10d::ScatterOptions& opts = c10d::ScatterOptions()) override;

    c10::intrusive_ptr<c10d::Work> send(
        std::vector<at::Tensor>& tensors,
        int dstRank,
        int tag) override;

    c10::intrusive_ptr<c10d::Work> recv(
        std::vector<at::Tensor>& tensors,
        int srcRank,
        int tag) override;

    void groupStart();

    void groupEnd();

    c10::intrusive_ptr<c10d::Work> recvAnysource(
        std::vector<at::Tensor>& tensors,
        int tag) override;

    c10::intrusive_ptr<c10d::Work> alltoall_base(
        at::Tensor& outputTensor,
        at::Tensor& inputTensor,
        std::vector<int64_t>& outputSplitSizes,
        std::vector<int64_t>& inputSplitSizes,
        const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

    c10::intrusive_ptr<c10d::Work> alltoall(
        std::vector<at::Tensor>& output_tensors,
        std::vector<at::Tensor>& input_tensors,
        const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

    static const int64_t kProcessGroupHCCLOpTimeoutMillis;

    // Agrees on an initial sequence number for the whole group by having rank 0
    // create it and broadcast it to other ranks using the store.
    void setSequenceNumberForGroup() override;

    // Retrieves the current sequence number for the whole group, which should be
    // in sync. If the returned number is not consistent across the group, it
    // may indicate that there is some sort of collective desynchronization.
    uint64_t getSequenceNumberForGroup() override;

    std::shared_ptr<HCCLComm> getHcclCommByDevices(const std::vector<at::Device>& devices);

    int64_t getHcclComm(int rankid);

    void setHcclCommName(const std::string& hccl_comm_name);

    void resumeHcclComm(int device_id);

    void setNSLBCommConfig(HcclCommConfig** commConfig);

    bool setCommWorkingDevNic(
        const HcclComm& comm,
        int nranks,
        std::vector<uint32_t>& ranks,
        std::vector<bool>& useBackup,
        int rankid,
        int hcclCommType,
        int p2pPeer);

    bool setSwitchNicComm(int rankid, int nranks, std::vector<uint32_t>& ranks, std::vector<bool>& useBackup);

    void setWatchdogStatus(int status);

    void clearWorkMetaList();

    std::string getHcclCommName(int rankid, bool init_comm = true);

    // Provides an API to abort the ProcessGroup (similar to hcclCommAbort)
    // instead of relying on ProcessGroupHCCL destructor.
    bool abort(c10::optional<std::string> abortReason = c10::nullopt);

    void shutdown(c10::optional<std::string> reason = c10::nullopt);

    void deleteTCPStoreKey();

    void abortAndClearHcclComm(c10::optional<std::string> abortReason);

    std::string getHcclCommNameWithoutInit(std::vector<std::shared_ptr<HCCLComm>>& hcclComms) const;

    // Return the global ranks of a PG
    const std::vector<uint32_t>& groupRanks() const;

    int64_t getStreamId(bool p2p, int peer);

    void windowRegisterAndExchange(int64_t windowSize, std::vector<uint32_t>& peerRanks);

    const at::Tensor& getWindowMem();

protected:
    // Helper that broadcasts HCCL Master ID to all ranks through the store
    void broadcastMasterID(
        HcclRootInfo* hcclID,
        bool isSingleP2POp,
        const std::string& devicesKey,
        int p2pRank);

    // Helper that either looks up the cached HCCL communicators or creates
    // a new set of HCCL communicators as a cache entry
    std::vector<std::shared_ptr<HCCLComm>>& getHCCLComm(
        const std::string& devicesKey,
        const std::vector<at::Device>& devices,
        HcclCommType commType = HcclCommType::DEFAULT,
        HcclCommConfig* commConfig = nullptr,
        int p2pRank = 0);

    // Get the data vol for HCCL operators.
    void recordDataVol(std::string opName, const std::string dataVol, const int currRank,
        std::vector<std::shared_ptr<HCCLComm>>& hcclComms);

    // Get the comm for HCCL operators.
    void recordComm(std::string filename, std::string opName, const int currRank,
        std::vector<std::shared_ptr<HCCLComm>>& hcclComms);

    // Wrapper method which can be overridden for tests.
    virtual std::exception_ptr checkForHCCLErrors(
        const std::vector<std::shared_ptr<HCCLComm>>& hcclComms);

    virtual c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL> initWork(
        std::vector<at::Device> devices,
        int rank,
        c10d::OpType opType,
        const char* profilingTitle = nullptr,
        const std::vector<at::Tensor>& inputs = {},
        const std::vector<at::Tensor>& outputs = {},
        bool record = false);

    void setGroupDesc(const std::string& desc)
    {
        pg_desc_ = desc;
    }

    const std::string& getGroupDesc() const
    {
        return pg_desc_;
    }
    
    void setP2pPeer(int newPeer)
    {
        peer_ = newPeer;
    }

    const int getP2pPeer() const
    {
        return peer_;
    }
    
    // In the timeout case and we will dump debug info such as the NCCL flight
    // recorder to storage. Down the road, if we have more complicated or blocking
    // operations, we might need to use a side thread to do it.
    bool dumpDebuggingInfo();
    void dumpTraceAndResetStatus();
    bool dumpPythonTraceback();
    std::future<bool> launchAsyncPythonTracebackDump();

    // Function that runs as part of a separate thread aside from watchdog
    // thread because we need to check the heartbeat from watchdog thread
    // so that when we get stuck in some HCCL/CANN calls,
    // we can dump the debugging information and abort the process.
    virtual void heartbeatMonitor();
    
    // Instance of the watchdog thread.
    std::unique_ptr<Watchdog> watchdog_;
    // Function that directly trigger std::abort so that the whole process
    // gets terminated.
    virtual void terminateProcess(std::string errMsg);

    // A helper function to wait for a future to complete or timeout.
    void waitForFutureOrTimeout(
        std::future<bool>& fut,
        const std::chrono::milliseconds& timeOutMilSec,
        const std::string& futDescription,
        bool throwException = false);

    // Do not call this directly, use ProcessGroup::setGroupName instead.
    void setGroupName(const std::string& name)
    {
        pg_name_ = name;
    }

    const std::string& getGroupName() const
    {
        return pg_name_;
    }

    static const int64_t kWatchdogThreadSleepMillis;

    // The store is used to broadcast the HCCL unique ID of rank 0. This store
    // comes with prefix and it is different across ProcessGroup HCCL instances
    // (aka, different ProcessGroups).
    c10::intrusive_ptr<c10d::Store> store_;

    // Reference to the store without prefix so that keys are same across all
    // ProcessGroup HCCL instances and (key, value) pairs written to the store are
    // global.
    c10::intrusive_ptr<c10d::Store> globalStore_;

    bool storeError_{false};

    const c10::intrusive_ptr<Options> options_;

    // The number of HCCL communicators that have been created during
    // the lifetime of this process group. This sequence number is
    // used to scope keys used in the store.
    uint64_t hcclCommCounter_{0};

    // The store keys to trace the last HCCL collective kernel Ascend events - start
    // event and end event respectively. These are used to do desync root cause
    // analysis.
    const std::string traceKeyStart_;
    const std::string traceKeyEnd_;

    // The HCCL communicator that the process group has cached.
    // The key is a list of NPU devices that an operation is operating on
    // The NPU devices are stored in a device sequence and the cache HCCL
    // communicator is associated with this NPU device sequence

    // e.g. If the process group op only uses device 0, then the value of
    // the used device string stored (value of the hashmap) would be "0".

    //      If the process group op uses device 0 - 7 and the each tensor of the
    //      input tensor list is on device, 0, 1, 2, 3, 4, 5, 6, 7 separately,
    //      then the value of the used device string (key) stored would be
    //      "0,1,2,3,4,5,6,7"

    //      If the process group op uses device 0 - 7 and the each tensor of the
    //      input tensor list is on device, 0, 4, 5, 6, 7, 1, 2, 3 separately,
    //      then the value of the used device string stored would be
    //      "0,4,5,6,7,1,2,3"
    //
    //      Note that the order of the device for the tensor list matters.
    std::unordered_map<std::string, std::vector<std::shared_ptr<HCCLComm>>> devHCCLCommMap_;

    std::unordered_set<std::string> reportedErrorComms_;
    
    std::unordered_map<int, std::vector<std::string>> p2pSendRecvKeys_;

    std::unordered_map<std::string, std::string> devHCCLCommNameMap_;

    std::unordered_set<std::string> TCPStoreKeyList_;

    // Mutex to guard maps like devHCCLCommMap_.
    std::mutex mutex_;

    // Heartbeat of watchdog thread.
    std::atomic_uint64_t heartbeat_;

    // The time interval used for deciding whether there is no watchdog heartbeat.
    int heartbeatTimeoutInSec_;

    // timeout for the dump to finish.
    int waitTimeoutDumpInMilSec_;

    // Interval of check coordinated signals in ProcessGroupHCCL from other ranks
    // e.g., trigger the dump of the debugging info for timeout when notified.
    int coordCheckIntervalMilSec_;

    // Size of ring buffer where we store HCCL Traces for debugging.
    int hcclTraceBufferSize_;

    // We gate the heartbeat monitor thread so that we can roll it out gradually.
    static std::atomic<bool> monitorThreadEnabled_;

    // Monitor thread which checks the heartbeat of Watchdog thread.
    // If the monitor thread finds there is no heartbeat, it will dump debug info
    // and then kill the watchdog thread to avoid hang.
    std::thread hcclHeartbeatMonitorThread_;

    // Whether or not we should terminate the watchdog and workCleanup threads.
    std::atomic<bool> terminateProcessGroup_;

    // Whether or not we should terminate the heartbeat monitoring threads.
    std::atomic<bool> terminateHeartbeatMonitorThread_;

    // Whether we are in the shutdown mode when we are trying to get debug info,
    // such as desync report.
    std::atomic<bool> collectiveDebugInfoMode_;

    // This is the signal from watchdog threads to indicate whether the monitor
    // thread should dump. Making it static so that it is accessiable from all the
    // PGs. With this flag, monitor thread would dump debug info under any one of
    // the 3 conditions: 1: this flag is set to true by the watchdog thread when
    // it detects a timeout. 2: timeout signal is received from
    // other ranks through tcpstore 3: no heartbeat of watchdog Note that only the
    // monitor thread from PG0 should dump the debug info and only once
    static std::atomic<bool> shouldDump_;

    // Vector to Store WorkHCCL pointers
    std::list<ProcessGroupHCCL::WorkHCCL> workMetaList_;

    // Mutex to Guard monitorWakeUpCV_
    std::mutex monitorMutex_;

    // Mutex to Guard workMetaList_
    std::mutex workMetaListMutex_;

    // Add Work Pointer to workVector
    void workEnqueue(c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>);

    // Condition Variable for watchdog thread sleep
    std::condition_variable workMetaListCV_;

    // Condition Variable for monitor thread to wake up early
    std::condition_variable monitorWakeUpCV_;

    std::chrono::time_point<std::chrono::steady_clock> lastWorkListUpdateTime_;

    // Condition variable to control how long the  watchdog thread waits.
    std::condition_variable watchdogCV_;

    // Mutex for watchdog.
    std::mutex watchdogCVMutex_;

    // The NPU steams used by HCCL kernels
    std::unordered_map<std::string, std::vector<c10_npu::NPUStream>>
        hcclStreams_;

    // The NPU events used to sync HCCL streams
    std::unordered_map<std::string, std::vector<c10_npu::NPUEvent>> hcclEvents_;

    // The NPU events used to control task rate to protect streams
    std::unordered_map<std::string, std::vector<c10_npu::NPUEvent>>
        rateCtrlEvents_;
    
    std::unordered_map<std::string, std::vector<uint64_t>> collectiveCnts_;

    // Device Indexes used for all collectives in this group
    std::set<int> usedDeviceIdxs_;

    int coalescing_state_ = 0;

    at::Device coalescedDevice_ = at::Device("npu");

    std::shared_ptr<HCCLComm> coalescedComm_ = nullptr;

    TensorShelf coalescedTensors_;

    // map from the key: "group name + pg counter (ID)" to the
    // HCCL Master ID count. This needs to be group and pg specific

    // For each process group, we need a uniform unique HCCL Master ID counter to
    // ensure that HCCL operation in this process group can be completed
    // successfully. Since each process group ID belongs to a group name, the key
    // to this map is a combination of group name and ProcessGroupHCCL ID.
    static std::unordered_map<std::string, ssize_t> pgUniqueHCCLIDCnt_;

    // map from group name to the pg counter (ID) within that group

    // For each group with the "group name" (which is the key), we need to
    // keep track of a unique process group ID when creating a new
    // ProcessGroupHCCL for this "group name". Therefore, the value of this
    // map keeps the unique ProcessGroupHCCL's ID for a specific group with
    // the "group name". The reason we need a per-group process group ID counter
    // is that different group can have different ranks and we need ensure that
    // each group has its own uniform process group ID for all its ranks.
    static std::unordered_map<std::string, ssize_t> processGroupCounterMap_;

    // Some ops may have completed, but user still hasn't called `work.wait()`.
    // When watchdog detects this, it transfers the TensorShelf from `work` to
    // this `shelves` structure. Next time we execute ProcessGroupNCCL's methods
    // on main thread, we clear the `shelves` in one shot. This is mainly because
    // watchdog (a side thread) unstashing the shelf directly seems to cause some
    // problem.
    std::vector<std::shared_ptr<TensorShelf>> shelvesToUnstash_;
    std::mutex shelvesMutex_;

    // Whether or not wait() and synchronize() are blocking operations that wait
    // for the operation to complete.
    bool blockingWait_ = false;

    // Whether or not the workCleanupThread is used to perform async error
    // handling.
    ErrorHandlingMode asyncErrorHandling_ = NoHandling;

    // Whether or not to enable timeout root cause analysis.
    bool desyncDebug_;

    // Whether or not to dump debug info on exception including both watchdog
    // timeout and hccl errors.
    bool dumpOnException_;

    bool hasGlobalDumped = false;

    // the perfdump path
    static std::string perfdumppath;

    struct CommStruct {
        std::string comm_name;
        std::string op_name;

        bool operator<(const CommStruct& other) const
        {
            return std::tie(comm_name, op_name) < std::tie(other.comm_name, other.op_name);
        }
    };

    std::set<CommStruct> commset;

    // Temporarily not implemented: std::unordered_set<std::string> abortedComms_;

    // The number of active hcclGroupStart() calls. This counter will be increased
    // by 1 when hcclGroupStart() is called and decreased by 1 when hcclGroupEnd()
    // is called.
    static thread_local uint64_t hcclActiveGroupCounter_;

    // Counting for the sequential number of NCCL collective call.
    // (specifically, how many actual kernels we launched, which differs from
    // op_id_ when coalescing is enabled)
    uint64_t seqCollective_{0};

    // Counting for the sequential number of NCCL P2P calls.
    uint64_t seqP2P_{0};

    // Counting for the sequential number of HCCL collective call.
    // (specfically, how many actual kernels we launched, which differs from)
    // op_id_ when coalescing is enabled)
    uint64_t seq_{0};

    // Incrementing counter for logical operations (collective or p2p) issued on
    // the ProcessGroup
    uint64_t op_id_{0};

    std::string pg_name_;

    int peer_;

    std::vector<uint32_t> global_ranks_in_group;

    std::exception_ptr watchDogException_ = nullptr;

    std::shared_ptr<ProcessGroupStatus> pgStatus_ = std::make_shared<ProcessGroupStatus>();

    struct StatusStruct {
        uint64_t seq = 0;
        std::string pgId;
        std::string opType;
        std::string commIds;
        std::string status;
    };

    StatusStruct StatusInfo;

    bool refreshStatusInfo(ProcessGroupHCCL::WorkHCCL work, std::string status);

    bool is_refreshed = false;

    static std::unordered_map<std::string, StatusStruct> StatusOutput_;

    std::mutex StatusMapmutex_;

    void updateStatusOutput();

    bool recordHcclStatus(const std::string path, bool end = false, bool error = false);

    static int deviceId_;

    static int numRanks_;

    static std::string exceptionMessage_;

    size_t uid_;

    std::string logPrefix_;

    std::string pg_desc_;

    std::string tcpMasterAddr;

    uint32_t tcpMasterPort;

private:
    // Helper that encapsulates work shared across all collective communication
    // primitives.
    template <typename Fn>
    c10::intrusive_ptr<c10d::Work> collective(
        std::vector<at::Tensor>& input,
        std::vector<at::Tensor>& output,
        Fn fn,
        c10d::OpType opType,
        bool asyncOp = false);
    
    template <typename Fn, typename PreProcess, typename PostProcess>
    c10::intrusive_ptr<c10d::Work> collective(
        std::vector<at::Tensor>& input,
        std::vector<at::Tensor>& output,
        Fn fn,
        PreProcess pre,
        PostProcess post,
        c10d::OpType opType,
        bool asyncOp = false);

    template <typename Fn, typename PreProcess, typename PostProcess>
    c10::intrusive_ptr<c10d::Work> collectiveCoalesced(
        std::vector<at::Tensor>& input,
        std::vector<at::Tensor>& output,
        Fn fn,
        PreProcess pre,
        PostProcess post,
        c10d::OpType opType,
        bool asyncOp = false);

    std::vector<std::shared_ptr<HCCLComm>>& createHCCLComm(
        const std::string& devicesKey,
        const std::vector<at::Device>& devices,
        HcclCommType commType = HcclCommType::DEFAULT,
        HcclCommConfig* commConfig = nullptr,
        int p2pRank = 0);

    void createHCCLCommOrigin(
        const std::string& devicesKey,
        const std::vector<at::Device>& devices,
        HcclCommType commType,
        HcclCommConfig* commConfig,
        std::vector<std::shared_ptr<HCCLComm>> &hcclComms,
        std::vector<c10_npu::NPUStream> &streamVal,
        int p2pRank);

    bool createHCCLCommEx(
        const std::string& devicesKey,
        const std::vector<at::Device>& devices,
        HcclCommType commType,
        HcclCommConfig* commConfig,
        std::vector<std::shared_ptr<HCCLComm>> &hcclComms,
        std::vector<c10_npu::NPUStream> &streamVal,
        int p2pRank);

    void createHCCLCommForZeroCopy(
        std::vector<std::shared_ptr<HCCLComm>> &hcclComms,
        std::unordered_map<std::string, std::string> &envMap);

    // Helper that encapsulates work shared across point-to-point communication
    // primitives. It is the same structure as the helper used for collective
    // communication primitives.
    template <typename Fn>
    c10::intrusive_ptr<c10d::Work> pointToPoint(
        std::vector<at::Tensor>& tensor,
        Fn fn,
        int peer,
        c10d::OpType opType);

    template <typename Fn, typename PreProcess, typename PostProcess>
    c10::intrusive_ptr<c10d::Work> pointToPoint(
        std::vector<at::Tensor>& tensor,
        Fn fn,
        int peer,
        c10d::OpType opType,
        PreProcess pre,
        PostProcess post);

    // Checks for HCCL errors on each of the communicators and returns an
    // appropriate exception_ptr (nullptr if no errors).
    static std::exception_ptr checkForHCCLErrorsInternal(const std::vector<std::shared_ptr<HCCLComm>>& hcclComms);

    // Checks for HCCLL errors via devHCCLCommMap_ instead of WorkHCCL
    void checkHcclComms();

    // Function that runs as part of a separate thread and checks for errors on
    // HCCL communicators. We need a separate thread to check for HCCL errors
    // since we can't rely on the user calling certain methods like wait(),
    // isCompleted() etc. to detect and remediate errors. In addition to this, we
    // need a mechanism to safely abort and remove HCCL communicators from our
    // cache. This can be done cleanly by having a thread for the ProcessGroupHCCL
    // class. Attempting to modify the communicator cache from the WorkHCCL class
    // might run into issues with object lifetime since the ProcessGroupHCCL
    // object might get destroyed before the WorkHCCL object.
    void hcclCommWatchdog();

    // Watchdog's inside loop.
    // Takes care of cleaning up completed work, and aborting upon failure or
    // timeout.
    void workCleanupLoop();

        // Desync debug helper
    void logWorkStart(WorkHCCL& work);

    // Desync debug helper
    void logWorkEnd(WorkHCCL& work);

    // Generates a prefix that is unique to this process group and rank, for
    // disambiguating logs
    std::string createLogPrefix() const;

    // Returns the unique prefix created in createLogPrefix
    const std::string &logPrefix() const;

    // Returns the global rank of the device. This function assumes that users
    // always create a default global process group(PG) which includes all
    // devices. It is called in the constructor of ProcessGroupHCCL, so it always
    // return the rank_ of the the very first PG created, aka, default global PG.
    const int &globalRank() const;

    void silenceCheck(at::Tensor &input, c10d::OpType opType);

    HcclCommConfig createHcclCommConfigWithOptions();

    c10_npu::NPUStream getHcclNPUStream(const at::Device &device);

    static std::string getMstxHcclMsg(const std::string &opName,
                                      uint64_t dataCnt,
                                      HcclDataType hcclType,
                                      HcclComm comm,
                                      int64_t streamId,
                                      int srcRank,
                                      int dstRank);

    std::unordered_map<c10d::OpType, std::pair<at::Tensor, at::Tensor>> silenceCheckCache_;

    WatchdogStatus watchdogStatus = WatchdogStatus::RUN;

    static ProcessGroupHCCL* global_;

    // window memory
    void* windowHandle_ = nullptr;

    c10::optional<at::Tensor> windowMem_;

    uint32_t cached_aic_num;
    
    uint32_t cached_aiv_num;

};

// Dumps the HCCL comm traces and additional information about the Process
// Group.
TORCH_API std::string dump_hccl_trace(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive);

// Dumps the HCCL comm traces and additional information about the Process
// Group in JSON formatted string.
// We don't include stack traces in JSON format as it is far too much data.
TORCH_API std::string dump_hccl_trace_json(
    bool includeCollectives,
    bool onlyActive);

// Gets a mutable reference to a global optional function.Heartbeat Monitor
// will use this function to dump traces, if available. Inside fbcode, we
// store a function here that uses an internal tool for process tracing
TORCH_API c10::optional<std::function<void(std::function<void(const std::string &)>)>> &get_cpp_trace_dumper();

// Similar to get_cpp_trace_dumper, this stores a function defined in
// torch-python layer that lets us check whether the GIL can be acquired,
// helpful for instrumenting in cases where a hang was observed.
typedef bool (*gil_checker_t)();

TORCH_API gil_checker_t &get_gil_checker();
} // namespace c10d_npu
