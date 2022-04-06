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

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include <c10/npu/NPUGuard.h>
#include <c10/npu/NPUStream.h>
#include <ATen/record_function.h>
#include <algorithm>
#include <map>
#include <tuple>
#include <unordered_set>

#include "torch_npu/csrc/distributed/HCCLUtils.hpp"
#include "torch_npu/csrc/distributed/ProcessGroupHCCL.hpp"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/acl/acl_base.h"

namespace c10d_npu {
namespace {
using hcclUs = std::chrono::steady_clock::time_point;
#define DURATION_US(x) \
  (std::chrono::duration_cast<std::chrono::microseconds>(x))
#define TIME_NOW() ({ std::chrono::steady_clock::now(); })

// HCCL ReduceOp mapping
std::map<c10d::ReduceOp, HcclReduceOp> hcclOp = {
    {c10d::ReduceOp::MIN, HCCL_REDUCE_MIN},
    {c10d::ReduceOp::MAX, HCCL_REDUCE_MAX},
    {c10d::ReduceOp::SUM, HCCL_REDUCE_SUM},
    {c10d::ReduceOp::PRODUCT, HCCL_REDUCE_PROD},
};

// HCCL DataType mapping
std::map<at::ScalarType, HcclDataType> hcclDataType = {
    {at::kChar, HCCL_DATA_TYPE_INT8},
    {at::kFloat, HCCL_DATA_TYPE_FP32},
    {at::kInt, HCCL_DATA_TYPE_INT32},
    {at::kHalf, HCCL_DATA_TYPE_FP16},
    {at::kShort, HCCL_DATA_TYPE_INT16},
    {at::kLong, HCCL_DATA_TYPE_INT64},
};

int64_t physical_numel(at::Tensor self){
  auto sizes = self.storage().unsafeGetStorageImpl()->npu_desc_.storage_sizes_;
  int64_t n = 1;
  for (auto s : sizes) {
    n *= s;
  }
  return n;
}

// Helper function that gets the data type and issues error if not supported
HcclDataType getHcclDataType(at::ScalarType type) {
  try {
    return hcclDataType.at(type);
  } catch (std::out_of_range& e) {
    throw std::runtime_error("Unsupported data type for HCCL process group");
  }
}

// Get the deviceList String from the list of devices
std::string getKeyFromDevices(const std::vector<at::Device>& devices) {
  std::string deviceList;
  for (auto& device : devices) {
    if (deviceList.empty()) {
      deviceList = std::to_string(device.index());
    } else {
      deviceList += "," + std::to_string(device.index());
    }
  }
  return deviceList;
}

// Get the list of devices from list of tensors
std::vector<at::Device> getDeviceList(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Device> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    res.push_back(tensor.device());
  }
  return res;
}

// [Sync Streams] Helper that lets the input hcclStreams to wait for the current
// stream. HCCL communications run on hcclStreams, but input tensors are
// allocated on different streams (i.e., current streams). Communications on
// hcclStreams cannot start before pending input tensor ops on current streams
// finish. Otherwise, ops on two streams might read/write same tensors
// concurrently.

// The synchronization above alone is not enough. We also need to make sure
// input tensors are not freed before their usages on hcclStreams finish. This
// can be achieved by calling ::recordStream,
// which remembers the usage stream (hcclStream), creates an event on the usage
// stream when GC attempts to free the input tensor, and delays GC until that
// event is done.
void syncStreams(
    const std::vector<at::Device>& devices,
    std::vector<at::npu::NPUEvent>& hcclEvents,
    std::vector<c10::npu::NPUStream>& hcclStreams) {
  for (size_t i = 0; i < devices.size(); ++i) {
    c10::npu::NPUStream& hcclStream = hcclStreams[i];
    at::npu::NPUEvent& hcclEvent = hcclEvents[i];
    hcclEvent.record(c10::npu::getCurrentNPUStream(devices[i].index()));
    hcclEvent.block(hcclStream);
  }
}

// exit call back for allreduce error
void exceptionCallback(aclrtExceptionInfo* exceptionInfo) {
  std::string err = "AllReduce error in:" + std::string(__FILE__) + ": " +
      std::to_string(__LINE__);
  throw std::runtime_error(err);
}
} // namespace

constexpr int64_t kSynchronizeBusyWaitMillis = 10;
constexpr int64_t maxOpNumPerSyncPoint = 2;
const int64_t ProcessGroupHCCL::kProcessGroupHCCLOpTimeoutMillis = 10 * 1000;
ProcessGroupHCCL::WorkHCCL::WorkHCCL(const std::vector<at::Device>& devices)
    : devices_(devices), workStartTime_(std::chrono::steady_clock::now()) {
  // Creates the npu event wrappers
  // Note: The actual events are lazily created when first recorded to with
  // DEFAULT_FLAGS = npuEventDisableTiming.
  npuEvents_.resize(devices.size());
  hcclComms_.resize(devices.size());
}

ProcessGroupHCCL::WorkHCCL::~WorkHCCL() {}

bool ProcessGroupHCCL::WorkHCCL::isCompleted() {
  checkAndSetException();
  return exception() || finishedNPUExecutionInternal();
}

bool ProcessGroupHCCL::WorkHCCL::isSuccess() const {
  if (exception()) {
    // Already detected an exception.
    return false;
  }
  return finishedNPUExecutionInternal();
}

void ProcessGroupHCCL::WorkHCCL::checkAndSetException() {
  if (exception()) {
    // We already have an exception.
    return;
  }
}

// Helper that checks if the HCCL kernels are completed on the NPU
bool ProcessGroupHCCL::WorkHCCL::finishedNPUExecution() {
  checkAndSetException();
  return finishedNPUExecutionInternal();
}

// check if HCCL task is finished
bool ProcessGroupHCCL::WorkHCCL::finishedNPUExecutionInternal() const {
  for (size_t i = 0; i < devices_.size(); ++i) {
    // Checking Event completed by Eventquery
    aclrtEventStatus status;
    auto ret = aclrtQueryEvent(npuEvents_[i], &status);
    if (ret != ACL_ERROR_NONE || status == ACL_EVENT_STATUS_NOT_READY) {
      return false;
    }
  }
  return true;
}

void ProcessGroupHCCL::WorkHCCL::checkAndThrowException() {
  // Set the appropriate exception if found.
  checkAndSetException();

  // Throw an exception, only if we have a valid exception.
  if (exception()) {
    std::rethrow_exception(exception());
  }
}

// Waiting on the work's corresponding NPU events
void ProcessGroupHCCL::WorkHCCL::synchronize() {
  for (size_t i = 0; i < devices_.size(); ++i) {
    auto currentStream = at::npu::getCurrentNPUStream(devices_[i].index());
    // Block the current stream on the HCCL stream
    npuEvents_[i].block(currentStream);
    // If we use the work to do barrier, we should block here
    if (!barrierTensors_.empty()) {
      c10::npu::NPUGuard npuGuard(devices_[i]);
      c10::npu::npuSynchronizeDevice();
    }
  }

  // In case of blocking, wait for the operation to complete.
  if (blockingWait_) {
    // Wait for the operation to complete.
    while (!isCompleted()) {
      auto currentTimepoint = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::milliseconds>(
              currentTimepoint - workStartTime_) > opTimeout_) {
        throw std::runtime_error("Operation timed out!");
      }
      // Check for errors and throw appropriate exception.
      checkAndThrowException();
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
    checkAndThrowException();
  }
}

// Same as calling synchronize().
bool ProcessGroupHCCL::WorkHCCL::wait(std::chrono::milliseconds timeout) {
  synchronize();
  // Always return true, because abort API is not implemented.
  return true;
}

ProcessGroupHCCL::ProcessGroupHCCL(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : c10d::ProcessGroup(rank, size),
      store_(store),
      hcclCommCounter_(0),
      terminateWatchdog_(false),
      opTimeout_(options->opTimeout) {
  char* blockingWait = getenv(HCCL_BLOCKING_WAIT);
  try {
    if (blockingWait != nullptr) {
      auto val = std::stoi(blockingWait);
      if (val == 1) {
        // Make wait() and synchronize() a blocking call.
        blockingWait_ = true;
      } else if (val != 0) {
        throw std::runtime_error(
            "Invalid value for environment variable: " +
            std::string(HCCL_BLOCKING_WAIT));
      }
    }
  } catch (std::exception& e) {
    throw std::runtime_error(
        "Invalid value for environment variable: " +
        std::string(HCCL_BLOCKING_WAIT));
  }
}

ProcessGroupHCCL::~ProcessGroupHCCL() {}

void ProcessGroupHCCL::broadcastMasterID(HcclRootInfo* hcclID) {
  // For every HCCL communicator that we create we need to broadcast
  // a unique ID from rank 0 to all other ranks. This broadcast is
  // done by rank 0 setting a key in the store and all other ranks
  // retrieving the contents of that key. A single process group
  // may create multiple HCCL communicators, so we use a sequence
  // number to differentiate between them.
  std::string storeKey = std::to_string(hcclCommCounter_++);
  if (rank_ == 0) {
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(hcclID),
        reinterpret_cast<uint8_t*>(hcclID) + HCCL_ROOT_INFO_BYTES);
    store_->set(storeKey, vec);
  } else {
    auto vec = store_->get(storeKey);
    TORCH_CHECK(vec.size() == HCCL_ROOT_INFO_BYTES);
    std::memcpy(hcclID, vec.data(), vec.size());
  }
}

std::vector<std::shared_ptr<HCCLComm>>& ProcessGroupHCCL::getHCCLComm(
    const std::string& devicesKey,
    const std::vector<at::Device>& devices) {
  // Sanity check
  if (devicesKey.empty()) {
    throw std::runtime_error(
        "Not able to create/get the HCCL Communicator since "
        "the NPU devices are not known");
  }

  for (auto& device : devices) {
    usedDeviceIdxs_.insert(device.index());
  }

  {
    std::lock_guard<std::mutex> lock(devHCCLCommMapLock_);
    if (devHCCLCommMap_.find(devicesKey) != devHCCLCommMap_.end()) {
      // Reuse the cached communicator if there is one.
      return devHCCLCommMap_[devicesKey];
    }
  }

  // HCCL communicator not cached, create a new entry
  std::vector<std::shared_ptr<HCCLComm>> hcclComms;
  hcclComms.resize(devices.size());

  HcclRootInfo hcclID;
  if (rank_ == 0) {
    C10D_HCCL_CHECK(HcclGetRootInfo(&hcclID));
  }
  broadcastMasterID(&hcclID);

  c10::npu::OptionalNPUGuard npuGuard;
  std::vector<c10::npu::NPUStream> streamVal;
  streamVal.reserve(devices.size());

  for (size_t i = 0; i < devices.size(); ++i) {
    int numRanks = getSize();
    int rank = getRank() * devices.size() + i;

    npuGuard.set_index(devices[i].index());
    hcclComms[i] = HCCLComm::create(numRanks, rank, hcclID);

    // Creates the HCCL streams
    streamVal.push_back(c10::npu::getNPUStreamFromPool(devices[i].index()));
  }

  hcclStreams_.emplace(devicesKey, std::move(streamVal));

  // Note: these events are created with the (default) cudaEventDisableTiming
  // flag This flag provides the best performance when used with
  // StreamWaitEvent() and EventQuery(). Since we here don't measure the
  // performance using npuEvent, this should be set.
  hcclEvents_.emplace(
      std::piecewise_construct,
      std::make_tuple(devicesKey),
      std::make_tuple(devices.size()));

  // stream length is 1024,
  rateCtrlEvents_.emplace(
      std::piecewise_construct,
      std::make_tuple(devicesKey),
      std::make_tuple(devices.size()));

  // record collectiveCnts.
  collectiveCnts_.emplace(
      std::piecewise_construct,
      std::make_tuple(devicesKey),
      std::make_tuple(devices.size()));

  // Hold the lock before modifying the cache.
  std::lock_guard<std::mutex> lock(devHCCLCommMapLock_);

  // Move the NCCL resource to cache
  devHCCLCommMap_.emplace(devicesKey, std::move(hcclComms));
  return devHCCLCommMap_[devicesKey];
}

namespace {

// Check that all `tensors' have the same type and shape and are distributed
// across distinct NPUs.
void check_npu_tensors(const std::vector<at::Tensor>& tensors) {
  // HCCL support one NPU per process only
  if (tensors.size() != 1) {
    throw std::runtime_error(
        "Tensor list mustn't be larger than the number of available NPUs");
  }
  // HCCL support contiguous tensor only
  if (!tensors[0].is_contiguous()) {
    throw std::runtime_error("Tensors must be contiguous");
  }
}

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_for_scatter_gather(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other,
    size_t world_size) {
  if (tensor_lists.size() != other.size()) {
    throw std::runtime_error(
        "Tensor list operands to scatter/gather must have the same length");
  }
  const auto num_devices = tensor_lists.size();

  std::vector<at::Tensor> flattened;
  flattened.resize(num_devices);

  for (auto i = size_t{}; i < num_devices; ++i) {
    if (tensor_lists[i].size() != world_size * num_devices) {
      throw std::runtime_error(
          "Tensor list input to scatter/gather must match number of collective"
          " participants");
    }

    // Only check device match for the first tensor in the list; the call to
    // newLikeFlat() below will check the rest.
    if (tensor_lists[i].front().get_device() != other[i].get_device()) {
      throw std::runtime_error(
          "Corresponding input/output tensors to scatter/gather must all reside"
          " on the same device");
    }

    for (const auto& t : tensor_lists[i]) {
      if (t.numel() != other[i].numel()) {
        throw std::runtime_error(
            "All tensor operands to scatter/gather must have the same size");
      }
    }
    // Flatten the tensors (from all ranks) into a single big tensor.
    flattened[i] = c10d::newLikeFlat(tensor_lists, i);
  }
  return flattened;
}

} // namespace

c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL> ProcessGroupHCCL::initWork(
    std::vector<at::Device> devices) {
  if (devices.size() != 1) {
    throw std::runtime_error(
        "ProcessGroupHCCL support one device per process only");
  }
  return c10::make_intrusive<ProcessGroupHCCL::WorkHCCL>(devices);
}

ProcessGroupHCCL::Options::Options()
    : opTimeout(kProcessGroupHCCLOpTimeoutMillis){}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post) {
  const auto devices = getDeviceList(inputs);
  const auto key = getKeyFromDevices(devices);
  auto& hcclComms = getHCCLComm(key, devices);
  // First let HCCL streams wait for input tensors allocation streams
  syncStreams(devices, hcclEvents_[key], hcclStreams_[key]);
  // Work itself will create the events on all NPUs of tensors
  auto work = initWork(devices);

  c10::npu::OptionalNPUGuard npuGuard;
  pre(hcclStreams_[key]);

  for (size_t i = 0; i < inputs.size(); ++i) {
    npuGuard.set_index(devices[i].index());
    c10::npu::NPUStream& hcclStream = hcclStreams_[key][i];

    // Both `inputs' and `outputs' are created on a worker stream and used in
    // different hcclStreams.  Hence, both must record the hcclStream to
    // prevent being freed before the collective finishes.
    //
    // We only record `inputs' here, and leave recording `outputs' to `fn' for
    // operations where `inputs' and `outputs' are not the same.
    //
    // See [Sync Streams].
    c10_npu::NPUCachingAllocator::recordStream(
        inputs[i].storage().data_ptr(), hcclStream);
  }
  {
    for (size_t i = 0; i < inputs.size(); ++i) {
      npuGuard.set_index(devices[i].index());
      // to avoid to much task pushed to the stream, leading to stream overflow
      // insert sync point fluxLimit(key, i)
      c10::npu::NPUStream& hcclStream = hcclStreams_[key][i];
      hcclUs startut = TIME_NOW();
      C10D_HCCL_CHECK(
          fn(inputs[i], outputs[i], hcclComms[i]->getHcclComm(), hcclStream));
    }
  }
  post(hcclStreams_[key]);

  for (size_t i = 0; i < inputs.size(); ++i) {
    c10::npu::NPUStream& hcclStream = hcclStreams_[key][i];
    work->npuEvents_[i].record(hcclStream);
    work->hcclComms_[i] = hcclComms[i];
    work->blockingWait_ = blockingWait_;
    work->opTimeout_ = opTimeout_;
  }

  return work;
}

template <typename Fn>
c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn) {
  return collective(
      inputs,
      outputs,
      fn,
      [](std::vector<c10::npu::NPUStream>&) {},
      [](std::vector<c10::npu::NPUStream>&) {});
}

int g_allreduceID = 0;
c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceOptions& opts) {
  check_npu_tensors(tensors);
  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          HcclComm comm,
          c10::npu::NPUStream& stream) {
        aclrtSetExceptionInfoCallback(exceptionCallback);
        RECORD_FUNCTION("HcclAllreduce", std::vector<c10::IValue>({input}));
        return HcclAllReduce(
            input.data_ptr(),
            output.data_ptr(),
            (uint64_t)physical_numel(input),
            getHcclDataType(input.scalar_type()),
            hcclOp[opts.reduceOp],
            comm,
            stream.stream());
      });
}
int g_broadcastID = 100000;
c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const c10d::BroadcastOptions& opts) {
  check_npu_tensors(tensors);
  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          HcclComm comm,
          c10::npu::NPUStream& stream) {
        RECORD_FUNCTION("HcclBroadcast", std::vector<c10::IValue>({input}));
        const auto root = opts.rootRank * tensors.size() + opts.rootTensor;
        return HcclBroadcast(
            input.data_ptr(),
            (uint64_t)physical_numel(input),
            getHcclDataType(input.scalar_type()),
            root,
            comm,
            stream.stream());
      });
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const c10d::AllreduceCoalescedOptions& /* unused */) {
  throw std::runtime_error(
      "ProcessGroupHCCL does not support allreduce_coalesced");
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::reduce(
    std::vector<at::Tensor>& /* unused */,
    const c10d::ReduceOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupHCCL does not support reduce");
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  check_npu_tensors(inputTensors);
  auto outputFlattened =
      flatten_for_scatter_gather(outputTensors, inputTensors, size_);
  check_npu_tensors(outputFlattened);

  return collective(
      inputTensors,
      outputFlattened,
      [&](at::Tensor& input,
          at::Tensor& output,
          HcclComm comm,
          c10::npu::NPUStream& stream) {
        RECORD_FUNCTION("HcclAllgather", std::vector<c10::IValue>({input}));
        c10_npu::NPUCachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
        return HcclAllGather(
            input.data_ptr(),
            output.data_ptr(),
            (uint64_t)physical_numel(input),
            getHcclDataType(input.scalar_type()),
            comm,
            stream.stream());
      },
      [&](std::vector<c10::npu::NPUStream>& hcclStreams) {},
      [&](std::vector<c10::npu::NPUStream>& hcclStreams) {
        // Copy the flattened output tensors to the outputs.
        for (size_t i = 0; i < outputTensors.size(); ++i) {
          c10::npu::NPUStreamGuard guard(hcclStreams[i]);
          for (size_t j = 0; j < outputTensors[0].size(); ++j) {
            // See [Sync Streams].
            c10_npu::NPUCachingAllocator::recordStream(
                outputTensors[i][j].storage().data_ptr(), hcclStreams[i]);

            outputTensors[i][j].copy_(outputFlattened[i][j], true);
          }
        }
      });
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::allgather_base(
    at::Tensor& /* unused */,
    at::Tensor& /* unused */,
    const c10d::AllgatherOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupHCCL does not support allgather_base");
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ReduceScatterOptions& opts) {
  check_npu_tensors(outputTensors);

  auto inputFlattened =
      flatten_for_scatter_gather(inputTensors, outputTensors, size_);
  check_npu_tensors(inputFlattened);

  return collective(
      inputFlattened,
      outputTensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          HcclComm comm,
          c10::npu::NPUStream& stream) {
        RECORD_FUNCTION("HcclReduceScatter", std::vector<c10::IValue>({input}));
        c10_npu::NPUCachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
        return HcclReduceScatter(
            input.data_ptr(),
            output.data_ptr(),
            (uint64_t)physical_numel(output),
            getHcclDataType(input.scalar_type()),
            hcclOp[opts.reduceOp],
            comm,
            stream.stream());
      },
      [&](std::vector<c10::npu::NPUStream>& hcclStreams) {
        // Copy the input tensors to the flattened inputs.
        for (size_t i = 0; i < inputTensors.size(); ++i) {
          c10::npu::NPUStreamGuard guard(hcclStreams[i]);
          for (size_t j = 0; j < inputTensors[0].size(); ++j) {
            // See [Sync Streams].
            c10_npu::NPUCachingAllocator::recordStream(
                inputTensors[i][j].storage().data_ptr(), hcclStreams[i]);

            inputFlattened[i][j].copy_(inputTensors[i][j], true);
          }
        }
      },
      [&](std::vector<c10::npu::NPUStream>& hcclStreams) {});
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::barrier(
    const c10d::BarrierOptions& opts) {
  std::vector<at::Device> devices;
  if (usedDeviceIdxs_.empty()) {
    auto numNPUs = c10::npu::device_count();
    int16_t deviceIdx = static_cast<int16_t>(rank_ % std::max(static_cast<int>(numNPUs), 1));
    devices.push_back(at::Device(at::DeviceType::NPU, deviceIdx));
  } else {
    for (auto usedDeviceIdx : usedDeviceIdxs_) {
      devices.push_back(at::Device(at::DeviceType::NPU, usedDeviceIdx));
    }
  }

  std::vector<at::Tensor> barrierTensors;
  barrierTensors.reserve(devices.size());

  at::npu::OptionalNPUGuard npuGuard;
  for (auto& device : devices) {
    npuGuard.set_index(device.index());
    barrierTensors.push_back(at::empty(
        {1},
        at::TensorOptions().device(at::DeviceType::NPU).dtype(at::kFloat)));
  }

  auto work = allreduce(barrierTensors);

  // Work will take over barrierTensors
  auto hcclWork = dynamic_cast<ProcessGroupHCCL::WorkHCCL*>(work.get());
  TORCH_CHECK(hcclWork);
  hcclWork->barrierTensors_ = std::move(barrierTensors);

  return work;
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const c10d::GatherOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupHCCL does not support gather");
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const c10d::ScatterOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupHCCL does not support scatter");
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::send(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */) {
  throw std::runtime_error("ProcessGroupHCCL does not support send");
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::recv(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */) {
  throw std::runtime_error("ProcessGroupHCCL does not support recv");
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */) {
  throw std::runtime_error("ProcessGroupHCCL does not support recv");
}

void ProcessGroupHCCL::release_resource() {
  c10::npu::npuSynchronizeDevice();
  this->hcclEvents_.clear();
  this->devHCCLCommMap_.clear();
}
} // namespace c10d_npu
