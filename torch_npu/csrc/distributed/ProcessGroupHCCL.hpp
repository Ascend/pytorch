// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <mutex>
#include <thread>
#include <unordered_map>
#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Utils.hpp>

#include <list>

#include "third_party/hccl/inc/hccl/hccl.h"
#include "torch_npu/csrc/distributed/HCCLUtils.hpp"
#include "torch_npu/csrc/npu/Event.h"


namespace c10d_npu {
// Environment variable which controls whether or not wait() is blocking or non-blocking.
constexpr const char* HCCL_BLOCKING_WAIT = "HCCL_BLOCKING_WAIT";
constexpr const char* HCCL_BACKEND_NAME = "hccl";

constexpr const char* HCCL_ASYNC_ERROR_HANDLING = "HCCL_ASYNC_ERROR_HANDLING";
constexpr const char* HCCL_DESYNC_DEBUG = "HCCL_DESYNC_DEBUG";

constexpr const int DEFAULT_TIMEOUT = 30 * 60 * 1000;

enum ErrorHandlingMode { NoHandling = 0, TearDown = 1, CleanUpOnly = 2};

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

class ProcessGroupHCCL : public c10d::ProcessGroup {
public:
  class WorkHCCL : public c10d::ProcessGroup::Work {
  public:
    // Constructor takes a list of NPU devices to adapt framework
    // But HCCL support one device only!!!
    explicit WorkHCCL(
        const std::vector<at::Device>& devices,
        int rank,
        c10d::OpType opType,
        uint64_t seq,
        bool desyncDebug);
    explicit WorkHCCL(const WorkHCCL& w);
    WorkHCCL& operator=(const WorkHCCL& w) = default;
    virtual ~WorkHCCL();

    // Checks if the HCCL kernel has started to execute.
    bool isStarted();

    std::shared_ptr<bool> is_dispatched = std::make_shared<bool>(false);
    bool is_reported = false;

    // Checks if request has completed. In this specific case of HCCL, it checks
    // if the HCCL operation has completed on the NPU in its own HCCL stream.
    // Non-blocking operation.
    bool isCompleted() override;

    bool isSuccess() const override;

    // Same as calling synchronize() for HCCL work.
    bool wait(std::chrono::milliseconds timeout) override;

    void handleExceptionGuard(ErrorHandlingMode asyncErrorHandling);

    // Let current stream wait on the completing of the HCCL work
    // Throws on exceptions. Blocking operation, which will wait for work
    // completion.
    void synchronize() override;

    // Helper function that checks if the HCCL have finished
    // execution on the NPUs
    bool finishedNPUExecution();

    void setException(std::exception_ptr exception_ptr);

    // Helper function that returns True if the WorkHCCL object has timed out
    // and False otherwise.
    bool timedOut();

    void checkDispatch();

    // Extend tensors lifecycle to work.synchronize, the tensors is local
    // variable and recordStream.  
    void lazyDestory(std::vector<at::Tensor> tensors);

  protected:

    // The NPU events tracking this work item on multiple NPU devices
    std::shared_ptr<std::vector<c10_npu::NPUEvent>> npuEvents_;

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

    bool startTraceUpdated_{false};

    virtual std::exception_ptr checkForRTSErrors(const
        std::vector<at::Device>& devicelist) const;
    virtual std::exception_ptr checkForHCCLErrors(const
        std::vector<std::shared_ptr<HCCLComm>>& hcclComms) const;

  private:
    // Checks for HCCL errors and sets an appropriate exception_ptr.
    void checkAndSetException();

    // Checks for HCCL errors and throws an appropriate exception.
    void checkAndThrowException();

    // Just checks whether GPU execution has started, without modifying
    // exception_ptr.
    bool startedNPUExecutionInternal() const;

    // Just checks whether NPU execution has completed, without modifying
    // exception_ptr.
    bool finishedNPUExecutionInternal() const;

    c10::intrusive_ptr<c10d::Store> store_;

    // save inputs for tensor free when WorkHCCL::wait
    std::vector<std::pair<c10::weak_intrusive_ptr<c10::StorageImpl>, c10_npu::NPUStream>> recorded_inputs_;
    std::vector<std::pair<c10::weak_intrusive_ptr<c10::StorageImpl>, c10_npu::NPUStream>> recorded_outputs_;

    std::vector<at::Tensor> lazy_destory_tensors_;

    std::vector<at::Tensor> stashed_for_allocator_safety_;

    friend class ProcessGroupHCCL;
  };

  struct Options : torch::CustomClassHolder {
    explicit Options(bool is_high_priority_stream = false);

    // return intrusive_ptr of the object
    static c10::intrusive_ptr<Options> create(
        bool is_high_priority_stream = false,
        std::chrono::milliseconds timeout = kNoTimeout) {
      return c10::make_intrusive<Options>(is_high_priority_stream);
    }

    std::chrono::milliseconds opTimeout;
    // Schedule HCCL operations on high priority CUDA streams
    bool is_high_priority_stream;
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

  virtual ~ProcessGroupHCCL();

  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }

  const std::string getBackendName() const {
      return "undefined";
  }
  c10::intrusive_ptr<c10d::ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> allreduce_out(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int64_t fusion_id,
    const c10d::AllreduceOptions& opts = c10d::AllreduceOptions());

  c10::intrusive_ptr<c10d::ProcessGroup::Work>allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceCoalescedOptions& opts =
          c10d::AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work>reduce(
      std::vector<at::Tensor>& tensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override;

    at::Tensor byte_alignment(at::Tensor& tensors);

  c10::intrusive_ptr<c10d::ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> allgather_togather(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions());

  c10::intrusive_ptr<c10d::ProcessGroup::Work> _allgather_base(
      at::Tensor& outputbuffer,
      at::Tensor& inputbuffer,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> _reduce_scatter_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> barrier(
      const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;

  // Unsupported Ops
  c10::intrusive_ptr<c10d::ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) override;

    c10::intrusive_ptr<c10d::ProcessGroup::Work> batch_isend_irecv(
        std::vector<std::string>& op_type,
        std::vector<at::Tensor>& tensors,
        std::vector<uint32_t> remote_rank_list);

  c10::intrusive_ptr<c10d::ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ScatterOptions& opts = c10d::ScatterOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> alltoall(
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

  int64_t getHcclComm(int rankid);

  std::string getHcclCommName(int rankid);

  std::string getHcclCommNameWithoutInit(int rankid, std::vector<std::shared_ptr<HCCLComm>>& hcclComms);

  int64_t getStreamId(bool p2p = false);
protected:

    // Get the data vol for HCCL operators.
    void recordDataVol(std::string opName, const std::string dataVol, const int currRank,
    std::vector<std::shared_ptr<HCCLComm>>& hcclComms);

    // Wrapper method which can be overridden for tests.
    virtual std::exception_ptr checkForRTSErrors(
        const std::vector<at::Device>& devicelist);
    virtual std::exception_ptr checkForHCCLErrors(
        const std::vector<std::shared_ptr<HCCLComm>>& hcclComms);

  // Helper that broadcasts HCCL Master ID to all ranks through the store
  void broadcastMasterID(HcclRootInfo* hcclID);

  // Helper that either looks up the cached HCCL communicators or creates
  // a new set of HCCL communicators as a cache entry
  std::vector<std::shared_ptr<HCCLComm>>& getHCCLComm(
      const std::string& devicesKey,
      const std::vector<at::Device>& devices);

  // Temporarily not implemented
  // virtual std::exception_ptr checkForHCCLErrors(const
  // std::vector<std::shared_ptr<HCCLComm>>& hcclComms);

  virtual  c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL> initWork(
    std::vector<at::Device> devices,
    int rank,
    c10d::OpType opType);

  static const int64_t kWatchdogThreadSleepMillis;
  static const int64_t kWorkCleanupThreadSleepMillis;

  // The store is used to broadcast the HCCL Master ID of rank 0.
  c10::intrusive_ptr<c10d::Store> store_;
  const c10::intrusive_ptr<Options> options_;

  // The number of HCCL communicators that have been created during
  // the lifetime of this process group. This sequence number is
  // used to scope keys used in the store.
  uint64_t hcclCommCounter_{0};

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
  std::unordered_map<std::string, std::vector<std::shared_ptr<HCCLComm>>>
      devHCCLCommMap_;

  bool storeError_{false};

  // Mutex to guard maps like devHCCLCommMap_.
  std::mutex mutex_;

  // Mutex to guard devHCCLCommMap_.
  std::mutex devHCCLCommMapLock_;

  // Watchdog thread which looks for errors on the cached HCCL communicators.
  std::thread hcclCommWatchdogThread_;

  // Whether or not we should terminate the watchdog thread.
  std::atomic<bool> terminateWatchdog_;

  // Whether or not we should terminate the watchdog and workCleanup threads.
  std::atomic<bool> terminateProcessGroup_;

  std::unordered_set<std::string> abortedComms_;

  // Map from HCCLUniqueId to appropriate communicator.
  std::unordered_map<std::string, std::vector<std::shared_ptr<HCCLComm>>>
      hcclIdToCommMap_;

    // Whether or not the workCleanupThread is used to perform async error
    // handling.
    ErrorHandlingMode asyncErrorHandling_ = NoHandling;

    // Whether or not to enable timeout root cause analysis.
    bool desyncDebug_ = false;

    // Condition variable to control how long the  watchdog thread waits.
    std::condition_variable watchdogCV_;

    // Mutex for watchdog.
    std::mutex watchdogCVMutex_;

    // Thread that removes HCCL Work upon timeout
    std::thread workCleanupThread_;

    // Vector to Store WorkHCCL pointers
    std::list<ProcessGroupHCCL::WorkHCCL> workMetaList_;

    std::mutex workMetaListMutex_;

    // Condition Variable for timeout thread sleep
    std::condition_variable workMetaListCV_;

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

  // Whether or not wait() and synchronize() are blocking operations that wait
  // for the operation to complete.
  bool blockingWait_ = false;

  // Timeout for operations. This is only used when blockingWait_ is enabled.
  std::chrono::milliseconds opTimeout_;

  // Temporarily not implemented: std::unordered_set<std::string> abortedComms_;

  // The number of active HCCLGroupStart() calls. This counter will be increased
  // by 1 when HCCLGroupStart() is called and decreased by 1 when HCCLGroupEnd()
  // is called.
  static thread_local uint64_t hcclActiveGroupCounter_;

  // Counting for the sequential number of HCCL collective call.
  uint64_t seq_{0};

  const std::string traceKeyStart_;
  const std::string traceKeyEnd_;

  // Add Work Pointer to workVector
  void workEnqueue(c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>);

private:
  // Helper that encapsulates work shared across all collective communication
  // primitives.

  void abortTimedOutCollectives(
    std::unordered_set<std::string>& abortedCommIds);

  void workCleanupLoop();

  template <typename Fn>
  c10::intrusive_ptr<c10d::ProcessGroup::Work> collective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      c10d::OpType opType);
  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<c10d::ProcessGroup::Work> collective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      c10d::OpType opType);

    static std::exception_ptr checkForRTSErrorsInternal(
        const std::vector<at::Device>& devicelist);
    static std::exception_ptr checkForHCCLErrorsInternal(
        const std::vector<std::shared_ptr<HCCLComm>>& hcclComms);
    static std::exception_ptr rts_device_error_query(int32_t devId);

    void hcclCommWatchdog();

    void hcclCommWatchdogInternal();

    void silenceCheck(at::Tensor &input, c10d::OpType opType);
    std::unordered_map<c10d::OpType, std::pair<at::Tensor, at::Tensor>> silenceCheckCache_;
};
} // namespace c10d_npu