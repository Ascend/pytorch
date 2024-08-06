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

#include <ATen/record_function.h>
#include <algorithm>
#include <map>
#include <tuple>
#include <unordered_set>
#include <unistd.h>
#include <linux/limits.h>
#include <fstream>
#include <iostream>

#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <c10d/ParamCommsUtils.hpp>
#include <c10d/TraceUtils.h>

#include "op_plugin/OpInterface.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/distributed/HCCLUtils.hpp"
#include "torch_npu/csrc/distributed/HcclCompile.h"
#include "torch_npu/csrc/distributed/ProcessGroupHCCL.hpp"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/interface/HcclInterface.h"
#include <c10d/Utils.hpp>

namespace c10d_npu {

constexpr const char* const kHCCLAbortedCommStoreKey = "HCCLABORTEDCOMM";

std::string getHcclErrorDetailStr(HcclResult error, c10::optional<std::string> processGroupFailureReason)
{
    // Prioritize failure reason provided by PG HCCL first, as it can abort
    // communicators when it encounters collective timeouts, etc.
    if (processGroupFailureReason != c10::nullopt) {
        return *processGroupFailureReason;
    }
    std::string interpret;
    std::string err;

    switch (error) {
        case HCCL_E_REMOTE:
            interpret =
                "HCCL_E_REMOTE: A call failed possibly due to a network error or a remote process exiting prematurely.";
            break;
        default:
            interpret = "Unknown HCCL error!";
    }
    return interpret + err;
}

namespace {
static constexpr uint32_t kOpWaitTimeoutOffset = 30U; // second
static uint32_t kOpWaitTimeout = 1868U; // second

using hcclUs = std::chrono::steady_clock::time_point;
#define DURATION_US(x) (std::chrono::duration_cast<std::chrono::microseconds>(x))
#define TIME_NOW() ({ std::chrono::steady_clock::now(); })

// HCCL ReduceOp mapping
std::map<c10d::ReduceOp, HcclReduceOp> hcclOp = {
    {c10d::ReduceOp::MIN, HCCL_REDUCE_MIN},
    {c10d::ReduceOp::MAX, HCCL_REDUCE_MAX},
    {c10d::ReduceOp::SUM, HCCL_REDUCE_SUM},
    {c10d::ReduceOp::PRODUCT, HCCL_REDUCE_PROD},
};

// HCCL DataType mapping
std::map<at::ScalarType, HcclDataType> kScalarTypeToHcclDataType = {
    {at::kByte, HCCL_DATA_TYPE_UINT8},
    {at::kChar, HCCL_DATA_TYPE_INT8},
    {at::kShort, HCCL_DATA_TYPE_INT16},
    {at::kInt, HCCL_DATA_TYPE_INT32},
    {at::kLong, HCCL_DATA_TYPE_INT64},
    {at::kHalf, HCCL_DATA_TYPE_FP16},
    {at::kFloat, HCCL_DATA_TYPE_FP32},
    {at::kDouble, HCCL_DATA_TYPE_FP64},
    {at::kBool, HCCL_DATA_TYPE_UINT8},
    {at::kBFloat16, HCCL_DATA_TYPE_BFP16},
};

std::map<HcclDataType, std::string> kHcclDataTypeToStringMap = {
    {HCCL_DATA_TYPE_UINT8, "at::kByte/at::kBool"},
    {HCCL_DATA_TYPE_INT8, "at::kChar"},
    {HCCL_DATA_TYPE_INT16, "at::kShort"},
    {HCCL_DATA_TYPE_INT32, "at::kInt"},
    {HCCL_DATA_TYPE_INT64, "at::kLong"},
    {HCCL_DATA_TYPE_FP16, "at::kHalf"},
    {HCCL_DATA_TYPE_FP32, "at::kFloat"},
    {HCCL_DATA_TYPE_FP64, "at::kDouble"},
    {HCCL_DATA_TYPE_BFP16, "at::kBFloat16"},
};

bool nslb_is_end = false;
char* nslb_path = c10_npu::option::OptionsManager::GetNslbPath();

int64_t physical_numel(const at::Tensor& self)
{
    auto sizes = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.storage_sizes_;
    int64_t n = 1;
    for (auto s : sizes) {
        n *= s;
    }
    return n;
}

// use tensor numel when the format is ACL_FORMAT_ND or ACL_FORMAT_NCHW
uint64_t getNumelForHCCL(const at::Tensor& self)
{
    aclFormat format = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.npu_format_;
    if (!at_npu::native::FormatHelper::IsBaseFormatType(format)) {
        if (self.storage().data_ptr().get() != self.data_ptr()) {
            TORCH_WARN_ONCE(
                "The storage data_ptr is different from tensor data_ptr."
                "Maybe this tensor is not suitable for HCCL.");
        }
        return physical_numel(self);
    } else {
        return self.numel();
    }
}

// Helper function that gets the data type and issues error if not supported
HcclDataType getHcclDataType(at::ScalarType type)
{
    try {
        return kScalarTypeToHcclDataType.at(type);
    } catch (std::out_of_range& e) {
        throw std::runtime_error("Unsupported data type for HCCL process group" + DIST_ERROR(ErrCode::NOT_SUPPORT));
    }
}

std::string getHcclDataTypeSerialString(HcclDataType type)
{
    const auto& iter = kHcclDataTypeToStringMap.find(type);
    if (iter != kHcclDataTypeToStringMap.end()) {
        return iter->second;
    } else {
        TORCH_WARN_ONCE("Can not serialize undefined hccl data type.");
        return "";
    }
}

HcclReduceOp getHcclReduceOp(const c10d::ReduceOp reduceOp, at::Tensor& input)
{
    if (reduceOp == c10d::ReduceOp::SUM && input.scalar_type() == at::kBool) {
        // For bool tensors, map sum to max, which both represent a bitwise or.
        // This is to prevent overflow issues with sum, since we use uint8 to
        // represent a bool (see HCCLDataType mapping).
        return HCCL_REDUCE_MAX;
    }
    return hcclOp[reduceOp];
}

// AllGather & Broadcast support all data type, no need do more check.
void checkSupportedDataType(HcclDataType type, std::string functionName)
{
    static std::set<HcclDataType> supportedDataTypes = {
        HCCL_DATA_TYPE_INT8,
        HCCL_DATA_TYPE_INT16,
        HCCL_DATA_TYPE_INT32,
        HCCL_DATA_TYPE_FP16,
        HCCL_DATA_TYPE_FP32,
        HCCL_DATA_TYPE_BFP16,
        HCCL_DATA_TYPE_INT64};
    TORCH_CHECK(
        supportedDataTypes.count(type) != 0,
        "HCCL "+functionName+": Unsupported data type ",
        getHcclDataTypeSerialString(type), DIST_ERROR(ErrCode::NOT_SUPPORT));
}

// Get the deviceList String from the list of devices
std::string getKeyFromDevices(const std::vector<at::Device>& devices)
{
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
std::vector<at::Device> getDeviceList(const std::vector<at::Tensor>& tensors)
{
    std::vector<at::Device> res;
    res.reserve(tensors.size());
    for (auto& tensor : tensors) {
        res.push_back(tensor.device());
    }
    return res;
}

// Return device with ordinal given by input rank.
at::Device getDeviceForRank(int rank)
{
    TORCH_CHECK(rank >= 0, "Invalid rank ", rank, DIST_ERROR(ErrCode::VALUE));
    auto numNPUs = c10_npu::device_count();
    if (numNPUs == 0) {
        AT_ERROR("Number of NPU devices on the machine is zero. Please check it");
    }
    int16_t deviceIdx = static_cast<int16_t>(rank % numNPUs);
    return at::Device(at_npu::key::NativeDeviceType, deviceIdx);
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
    std::vector<c10_npu::NPUEvent>& hcclEvents,
    std::vector<c10_npu::NPUStream>& hcclStreams)
{
    for (size_t i = 0; i < devices.size(); ++i) {
        c10_npu::NPUStream& hcclStream = hcclStreams[i];
        c10_npu::NPUEvent& hcclEvent = hcclEvents[i];
        hcclEvent.record(c10_npu::getCurrentNPUStream(devices[i].index()));
        ASCEND_LOGI("Event: record hccl group is successfully executed, event=%p", hcclEvent.event());
        hcclEvent.block(hcclStream);
        ASCEND_LOGI("Event: block hccl group is successfully executed, event=%p", hcclEvent.event());
    }
}

// exit call back for allreduce error
void exceptionCallback(aclrtExceptionInfo* exceptionInfo)
{
    // notice: Do not raise error, otherwise we will get call stacks of the rts callback function.
    fprintf(stdout, "Inner error, see details in Ascend logs.");
}

// Returns exception's what() given an exception_ptr instance.
std::string getExceptionMsgFromExceptionPtr(const std::exception_ptr& exceptionPtr)
{
    TORCH_CHECK(exceptionPtr != nullptr, DIST_ERROR(ErrCode::PTR));
    try {
        std::rethrow_exception(exceptionPtr);
    } catch (const std::exception& e) {
        return e.what();
    } catch (...) {
        return "Unknown exception type";
    }
}

} // namespace

constexpr int64_t kSynchronizeBusyWaitMillis = 10;
constexpr int64_t maxOpNumPerSyncPoint = 2;
const int64_t ProcessGroupHCCL::kProcessGroupHCCLOpTimeoutMillis = 10 * 1000;
thread_local uint64_t ProcessGroupHCCL::hcclActiveGroupCounter_ = 0;
constexpr int64_t kWaitForAbortCommStoreKey = 1000;
const int64_t ProcessGroupHCCL::kWatchdogThreadSleepMillis = 10000;
const int64_t ProcessGroupHCCL::kWorkCleanupThreadSleepMillis = 1000;
// const int64_t ProcessGroupHCCL::kProcessGroupHCCLOpTimeoutMillis = 10 * 1000;
ProcessGroupHCCL::WorkHCCL::WorkHCCL(
    const std::vector<at::Device>& devices,
    int rank,
    c10d::OpType opType,
    uint64_t seq,
    bool desyncDebug)
    : Work(rank, opType), devices_(devices), workStartTime_(std::chrono::steady_clock::now()), seq_(seq)
{
    // Creates the npu event wrappers
    // Note: The actual events are lazily created when first recorded to with
    // DEFAULT_FLAGS = npuEventDisableTiming.
    if (desyncDebug) {
        hcclStartEvents_ = std::make_shared<std::vector<c10_npu::NPUEvent>>(devices.size());
    }
    npuEvents_ = std::make_shared<std::vector<c10_npu::NPUEvent>>(devices.size());
    hcclComms_.resize(devices.size());
}

ProcessGroupHCCL::WorkHCCL::WorkHCCL(const WorkHCCL& w)
    : Work(w.rank_, w.opType_),
      devices_(w.devices_),
      hcclComms_(w.hcclComms_),
      hcclStartEvents_(w.hcclStartEvents_),
      npuEvents_(w.npuEvents_),
      blockingWait_(w.blockingWait_),
      opTimeout_(w.opTimeout_),
      workStartTime_(w.workStartTime_),
      seq_(w.seq_),
      startTraceUpdated_(w.startTraceUpdated_),
      store_(w.store_),
    is_dispatched(w.is_dispatched),
    is_reported(w.is_reported)
{
    exception_ = w.exception_;
}

ProcessGroupHCCL::WorkHCCL::~WorkHCCL() {}

bool ProcessGroupHCCL::WorkHCCL::isCompleted()
{
    checkAndSetException();
    return exception() || finishedNPUExecutionInternal();
}

bool ProcessGroupHCCL::WorkHCCL::isStarted()
{
    checkAndSetException();
    return exception() || startedNPUExecutionInternal();
}

bool ProcessGroupHCCL::WorkHCCL::isSuccess() const
{
    if (exception()) {
        // Already detected an exception.
        return false;
    }
    return finishedNPUExecutionInternal();
}

void ProcessGroupHCCL::WorkHCCL::checkAndSetException()
{
    if (exception()) {
        // We already have an exception.
        return;
    }

    // Check rts error for specific devices. If no rts error found, then check hccl error of specific hcclcomms.
    std::exception_ptr rt_device_status = checkForRTSErrors(devices_);
    std::exception_ptr hcclErrorException = nullptr;
    if (!rt_device_status) {
        hcclErrorException = checkForHCCLErrors(hcclComms_);
    } else {
        LOG(INFO) << "[Rank " << rank_ << "]"
                  << " found exception when checking for RTS errors: "
                  << getExceptionMsgFromExceptionPtr(rt_device_status);
    }
    if (hcclErrorException) {
        LOG(INFO) << "[Rank " << rank_ << "]"
                  << " found async exception when checking for HCCL errors: "
                  << getExceptionMsgFromExceptionPtr(hcclErrorException);
    }

    std::unique_lock<std::mutex> lock(mutex_);
    exception_ = (hcclErrorException) ? (hcclErrorException) : (rt_device_status);
}

void ProcessGroupHCCL::WorkHCCL::setException(std::exception_ptr exception_ptr)
{
    std::unique_lock<std::mutex> lock(mutex_);
    exception_ = exception_ptr;
}

// Helper that checks if the HCCL kernels are completed on the NPU
bool ProcessGroupHCCL::WorkHCCL::finishedNPUExecution()
{
    checkAndSetException();
    return finishedNPUExecutionInternal();
}

bool ProcessGroupHCCL::WorkHCCL::startedNPUExecutionInternal() const
{
    for (const auto i : c10::irange(devices_.size())) {
        // Checking the work's corresponding NPU events' status
        if (!(*hcclStartEvents_)[i].query()) {
            return false;
        }
    }
    return true;
}

// check if HCCL task is finished
bool ProcessGroupHCCL::WorkHCCL::finishedNPUExecutionInternal() const
{
    try {
        for (const auto i : c10::irange(devices_.size())) {
            // Checking the work's corresponding CUDA events' status
            if (!(*npuEvents_)[i].query()) {
                return false;
            }
        }
    } catch (const std::exception& e) {
        if (std::string(e.what()).find("driver shutting down") == std::string::npos) {
            throw std::runtime_error(DIST_ERROR(ErrCode::INTERNAL));
        }
        LOG(INFO) << "[Rank " << rank_ << "] Event query failed with exception: " << e.what();
    }
    return true;
}

void ProcessGroupHCCL::WorkHCCL::checkAndThrowException()
{
    // Set the appropriate exception if found.
    checkAndSetException();

    // Throw an exception, only if we have a valid exception.
    if (exception()) {
        std::rethrow_exception(exception());
    }
}

// Waiting on the work's corresponding NPU events
void ProcessGroupHCCL::WorkHCCL::synchronize()
{
    for (const auto i : c10::irange(devices_.size())) {
        auto currentStream = c10_npu::getCurrentNPUStream(devices_[i].index());
        // Block the current stream on the HCCL stream
        (*npuEvents_)[i].block(currentStream);
        ASCEND_LOGI("Event: block hccl work is successfully executed, event=%p", (*npuEvents_)[i].event());
        // If we use the work to do barrier, we should block here
        if (!barrierTensors_.empty()) {
            c10_npu::NPUGuard npuGuard(devices_[i]);
            c10_npu::npuSynchronizeDevice();
        }
    }

    if (!recorded_inputs_.empty()) {
        for (auto it = recorded_inputs_.begin(); it != recorded_inputs_.end(); ++it) {
            auto storage = it->first.lock();
            if (storage) {
                c10_npu::NPUCachingAllocator::eraseStream(storage->data_ptr(), it->second);
            }
        }
    }
    if (!recorded_outputs_.empty()) {
        for (auto it = recorded_outputs_.begin(); it != recorded_outputs_.end(); ++it) {
            auto storage = it->first.lock();
            if (storage) {
                c10_npu::NPUCachingAllocator::eraseStream(storage->data_ptr(), it->second);
            }
        }
    }

    if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::ERASE_RECORD_STREAM) {
        lazy_destory_tensors_.clear();
    } else if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::AVOID_RECORD_STREAM) {
        stashed_for_allocator_safety_.clear();
    }

    // In case of blocking, wait for the operation to complete.
    if (blockingWait_) {
        // Wait for the operation to complete.
        while (!isCompleted()) {
            auto currentTimepoint = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(currentTimepoint - workStartTime_) > opTimeout_) {
                throw std::runtime_error("Operation timed out!" + DIST_ERROR(ErrCode::TIMEOUT));
            }
            // Check for errors and throw appropriate exception.
            checkAndThrowException();
            std::this_thread::sleep_for(std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
        }
        checkAndThrowException();
    }
}

void ProcessGroupHCCL::WorkHCCL::handleExceptionGuard(ErrorHandlingMode asyncErrorHandling)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (exception_) {
        auto exceptionMsg = c10::str(
            "Some RTS or HCCL operations have failed or timed out. Due to the ",
            "asynchronous nature of NPU kernels, subsequent NPU operations ",
            "might run on corrupted/incomplete data.");
        LOG(ERROR) << exceptionMsg;
        C10_LOG_API_USAGE_ONCE("ProcessGroupHCCL.WorkHCCL.handleExceptionGuard");
        if (asyncErrorHandling == TearDown) {
            auto tearDownMsg = c10::str("To avoid data inconsistency, we are taking the entire process down.");
            LOG(ERROR) << tearDownMsg;
            std::rethrow_exception(exception_);
        }
    }
}

// Same as calling synchronize().
bool ProcessGroupHCCL::WorkHCCL::wait(std::chrono::milliseconds timeout)
{
    synchronize();
    // Always return true, because abort API is not implemented.
    return true;
}

bool ProcessGroupHCCL::WorkHCCL::timedOut()
{
    auto currentTimepoint = std::chrono::steady_clock::now();
    return (std::chrono::duration_cast<std::chrono::milliseconds>(currentTimepoint - workStartTime_) >= opTimeout_);
}

std::chrono::milliseconds GetDispatchTimeout()
{
    uint32_t dispatchTimeout_ = 600U;
    uint32_t dispatchoffset = 30U;
    uint32_t mindispatchTimeout_ = 120U;

    uint32_t hccl_exec_timeout = c10_npu::option::OptionsManager::GetHCCLExecTimeout();
    if (hccl_exec_timeout > 0) {
        if (hccl_exec_timeout < dispatchTimeout_ + dispatchoffset && hccl_exec_timeout > mindispatchTimeout_ + dispatchoffset) {
            dispatchTimeout_ = hccl_exec_timeout - dispatchoffset;
        };
    };
    ASCEND_LOGI("set dispatchTimeout_ %u s.", dispatchTimeout_);
    return std::chrono::milliseconds(dispatchTimeout_ * 1000U);
}

std::chrono::milliseconds dispatchTimeout_ = GetDispatchTimeout();

void ProcessGroupHCCL::WorkHCCL::checkDispatch()
{
    if (!*is_dispatched && !is_reported) {
        auto currentTimepoint = std::chrono::steady_clock::now();
        auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTimepoint - workStartTime_);
        if (timeElapsed > dispatchTimeout_) {
            std::string repo_info = c10_npu::getRepoInfo();
            ASCEND_LOGE("Process group work %s, seq_num %u dispatch timeout. %s", opTypeToString(opType_).c_str(), seq_, repo_info.c_str());
            is_reported = true;
        }
    } else if (*is_dispatched && is_reported) {
        ASCEND_LOGE("Process group work %s, seq_num %u dispatch sucess. This error log can be ignored.", opTypeToString(opType_).c_str(), seq_);
        is_reported = false;
    }
}

void ProcessGroupHCCL::WorkHCCL::lazyDestory(std::vector<at::Tensor> tensors) {
    if (tensors.empty() ||
        (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() != c10_npu::option::ERASE_RECORD_STREAM)) {
        return;
    }

    for (const auto i : c10::irange(tensors.size())) {
        lazy_destory_tensors_.push_back(tensors[i]);
    }
}

void ProcessGroupHCCL::hcclCommWatchdog()
{
    try {
        LOG(INFO) << "[Rank " << rank_ << "] HCCL watchdog thread started!";
        hcclCommWatchdogInternal();
        LOG(INFO) << "[Rank " << rank_ << "] HCCL watchdog thread terminated normally";
    } catch (std::exception& e) {
        LOG(INFO) << "[Rank " << rank_ << "] HCCL watchdog thread terminated with exception: " << e.what();
    } catch (...) {
        LOG(INFO) << "[Rank " << rank_ << "] HCCL watchdog thread terminated with unknown exception";
    }
}

// Given a hcclUniqueId, convert it to a string representation that can be put
// in the store.
std::string buildHcclUniqueIdStr(const HcclRootInfo& hcclID)
{
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&hcclID);
    std::ostringstream oss;
    for (const auto i : c10::irange(HCCL_ROOT_INFO_BYTES)) {
        oss << std::hex << static_cast<int>(bytes[i]);
    }
    return oss.str();
}

std::string getHcclAbortedCommStoreKey(const std::string hcclIdStr)
{
    return std::string(kHCCLAbortedCommStoreKey) + ":" + hcclIdStr;
}

std::exception_ptr ProcessGroupHCCL::rts_device_error_query(int32_t devId)
{
    aclrtDeviceStatus deviceStatus = ACL_RT_DEVICE_STATUS_NORMAL;
    auto ret = c10_npu::acl::AclrtQueryDeviceStatus(devId, &deviceStatus);
    if (ret != 0) {
        TORCH_NPU_WARN("Device[", devId, "] AclrtQueryDeviceStatus interface get unexpected query, query code is [", ret, "]");
        ASCEND_LOGD("Device[", devId, "] AclrtQueryDeviceStatus interface get unexpected query, query code is [", ret, "]");
    }

    if (ret == 0 && deviceStatus != ACL_RT_DEVICE_STATUS_NORMAL) {
        return std::make_exception_ptr(std::runtime_error(c10::str("rts error: ", deviceStatus)));
    }
    return nullptr;
}

std::exception_ptr ProcessGroupHCCL::WorkHCCL::checkForRTSErrors(
    const std::vector<at::Device>& devicelist) const
{
    return checkForRTSErrorsInternal(devicelist);
}

std::exception_ptr ProcessGroupHCCL::checkForRTSErrors(const std::vector<at::Device>& devicelist)
{
    return checkForRTSErrorsInternal(devicelist);
}

std::exception_ptr ProcessGroupHCCL::checkForRTSErrorsInternal(const std::vector<at::Device>& devicelist)
{
    std::exception_ptr rt_device_status = nullptr;
    for (const auto& device : devicelist) {
        rt_device_status = rts_device_error_query(device.index());
        if (rt_device_status)
            break;
    }
    return rt_device_status;
}

std::exception_ptr ProcessGroupHCCL::WorkHCCL::checkForHCCLErrors(
    const std::vector<std::shared_ptr<HCCLComm>>& hcclComms) const
{
    return checkForHCCLErrorsInternal(hcclComms);
}

std::exception_ptr ProcessGroupHCCL::checkForHCCLErrors(const std::vector<std::shared_ptr<HCCLComm>>& hcclComms)
{
    return checkForHCCLErrorsInternal(hcclComms);
}

std::exception_ptr ProcessGroupHCCL::checkForHCCLErrorsInternal(const std::vector<std::shared_ptr<HCCLComm>>& hcclComms)
{
    for (auto& hcclComm : hcclComms) {
        //  checkForHcclError() result
        HcclResult hcclAsyncErr = hcclComm->checkForHcclError();
        if (hcclAsyncErr != HCCL_SUCCESS) {
            return std::make_exception_ptr(std::runtime_error("HCCL error: " + getHcclErrorDetailStr(hcclAsyncErr)));
        }
    }
    return nullptr;
}

// Get the list of devices from devicekey
std::vector<at::Device> get_device_list_from_devicekey(const std::string& devicesKey)
{
    std::vector<at::Device> res;
    std::stringstream ss(devicesKey);
    std::string token;
    while (std::getline(ss, token, ',')) {
        res.push_back(at::Device(at_npu::key::NativeDeviceType, std::stoi(token)));
    }
    return res;
}

void ProcessGroupHCCL::abortTimedOutCollectives(std::unordered_set<std::string>& abortedCommIds)
{
    std::unique_lock<std::mutex> lock(workMetaListMutex_);
    for (auto& work : workMetaList_) {
        work.checkAndSetException();
        work.checkDispatch();
        // Aborting HCCL Communicators due to errors is already handled above.
        if (work.exception()) {
            continue;
        }

        // Check for Timeouts in the WorkHCCL Operations, and abort all
        // communicators accordingly.
        if (work.timedOut()) {
            auto currentTimepoint = std::chrono::steady_clock::now();
            auto timeElapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(currentTimepoint - work.workStartTime_);
            std::string exceptionMsg = c10::str(
                "[Rank ",
                rank_,
                "] ",
                "Watchdog caught collective operation timeout: ",
                // work,
                " ran for ",
                timeElapsed.count(),
                " milliseconds before timing out.");
            if (desyncDebug_) {
                exceptionMsg += retrieveDesyncReport(store_, "HCCL", rank_, size_);
            }
            LOG(ERROR) << exceptionMsg;
            std::exception_ptr exception_ptr = std::make_exception_ptr(std::runtime_error(exceptionMsg));
            work.setException(exception_ptr);
            for (const auto& hcclComm : work.hcclComms_) {
                hcclComm->destropyHcclComm();
                abortedCommIds.emplace(buildHcclUniqueIdStr(hcclComm->getHcclId()));
            }
        }
    }
}

void ProcessGroupHCCL::hcclCommWatchdogInternal()
{
    while (!terminateProcessGroup_.load()) {
        std::unordered_set<std::string> abortedCommIds;
        std::unordered_set<std::string> allCommIds;

        {
            // Loop through the cache of communicators for HCCL errors.
            std::lock_guard<std::mutex> lock(mutex_);

            for (auto& it : devHCCLCommMap_) {
                auto& hcclComms = it.second;

                for (const auto& hcclComm : hcclComms) {
                    allCommIds.emplace(buildHcclUniqueIdStr(hcclComm->getHcclId()));
                }

                // Check rts error for specific devices. If no rts error found, then check hccl error of specific hcclcomms.
                std::vector<at::Device> devicelist = get_device_list_from_devicekey(it.first);
                std::exception_ptr rt_device_status = checkForRTSErrors(devicelist);
                std::exception_ptr hcclErrorException = nullptr;
                if (!rt_device_status) {
                    hcclErrorException = checkForHCCLErrors(hcclComms);
                } else {
                    LOG(INFO) << "[Rank " << rank_ << "] Received RTS errors for communicators in the cache: \n"
                              << "RTS error: \n"
                              << getExceptionMsgFromExceptionPtr(rt_device_status);
                }
                if (hcclErrorException) {
                    LOG(INFO) << "[Rank " << rank_ << "] Received HCCL errors for communicators in the cache: \n"
                              << "HCCL error: \n"
                              << getExceptionMsgFromExceptionPtr(hcclErrorException);
                }

                if ((hcclErrorException || rt_device_status) && (blockingWait_ || asyncErrorHandling_ != NoHandling)) {
                    LOG(INFO) << "[Rank " << rank_ << "] Aborting communicators that received errors";
                    // We abort HCCL communicators that have received errors
                    // or where the devices have received errors from this thread,
                    // and exceptions are set on the corresponding work objects.
                    // The workCleanupThread will then loop through the unfinished
                    // collectives and throw exceptions if an exception has been set on
                    // any of the work objects from the thread.
                    for (const auto& hcclComm : hcclComms) {
                        hcclComm->destropyHcclComm();
                        abortedCommIds.emplace(buildHcclUniqueIdStr(hcclComm->getHcclId()));
                    }
                }
            }
        }

        if (asyncErrorHandling_ != NoHandling) {
            abortTimedOutCollectives(abortedCommIds);
        }

        if (blockingWait_) {
            // When we abort a communicator on one rank, it is likely that might cause
            // other ranks to hang indefinitely. As a result, whenever we abort a
            // communicator, we write its ID to the store. The watchdog on other ranks
            // then monitor the store, find an aborted communicator ID and abort their
            // respective communicator as well.

            // Record the aborted communicators locally and in the store.
            for (const auto& abortedCommId : abortedCommIds) {
                abortedComms_.emplace(abortedCommId);
                const auto& storeKey = getHcclAbortedCommStoreKey(abortedCommId);
                auto rankStr = std::to_string(rank_);
                store_->set(
                    storeKey,
                    std::vector<uint8_t>(
                        reinterpret_cast<const uint8_t*>(rankStr.data()),
                        reinterpret_cast<const uint8_t*>(rankStr.data()) + rankStr.size()));
                LOG(INFO) << "[Rank " << rank_ << "] Watchdog wrote aborted communicator id to store: " << storeKey;
            }

            // Check for any communicators in the store and abort them if needed.
            for (const auto& commId : allCommIds) {
                if (abortedComms_.find(commId) == abortedComms_.end()) {
                    // Check if we need to abort them if not already aborted (shouldn't
                    // wait more than the watchdog sleep time.).
                    const auto& storeKey = getHcclAbortedCommStoreKey(commId);
                    try {
                        store_->wait({storeKey}, std::chrono::milliseconds(kWaitForAbortCommStoreKey));
                        auto val = store_->get(storeKey);
                        std::string rank(reinterpret_cast<char*>(val.data()), val.size());
                        std::stringstream ss;
                        ss << "[Rank " << rank_ << "] Found key in store: " << storeKey << ", from rank: " << rank
                           << ". This means that rank has aborted its HCCL communicators previously and is not in a healthy state."
                           << ". Aborting appropriate communicators";
                        std::string abortReason = ss.str();
                        LOG(WARNING) << abortReason;

                        // Now abort the appropriate communicators.
                        std::lock_guard<std::mutex> lock(mutex_);
                        auto it = hcclIdToCommMap_.find(commId);
                        TORCH_INTERNAL_ASSERT(it != hcclIdToCommMap_.end());
                        for (const auto& hcclComm : it->second) {
                            // The reason we are aborting is because some other ranks have
                            // aborted their communicators originally
                            hcclComm->destropyHcclComm();
                        }
                        abortedComms_.emplace(commId);
                        LOG(INFO) << "[Rank " << rank_ << "] Aborted communicators for key in store: " << storeKey;
                    } catch (std::exception& e) {
                        VLOG(1) << "Did not find key in store: " << storeKey << ", error: " << e.what();
                    }
                }
            }
        }

        std::unique_lock<std::mutex> lock(watchdogCVMutex_);

        watchdogCV_.wait_for(lock, std::chrono::milliseconds(kWatchdogThreadSleepMillis), [&]() -> bool {
            return terminateProcessGroup_.load();
        });
    }
}

ProcessGroupHCCL::ProcessGroupHCCL(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : c10d::ProcessGroup(rank, size),
      store_(store),
      options_(options),
      hcclCommCounter_(0),
      opTimeout_(options->opTimeout),
      terminateWatchdog_(false),

      traceKeyStart_("HCCL_" + std::to_string(rank) + "_trace_start"),
      traceKeyEnd_("HCCL_" + std::to_string(rank) + "_trace_end"),
      terminateProcessGroup_(false)
{
    uint32_t hccl_exec_timeout = c10_npu::option::OptionsManager::GetHCCLExecTimeout();
    // When no env, the default value is 0
    if (hccl_exec_timeout > 0) {
        kOpWaitTimeout = hccl_exec_timeout + kOpWaitTimeoutOffset;
        if (kOpWaitTimeout <= hccl_exec_timeout) {
            kOpWaitTimeout = UINT_MAX;
        }
    }
    NPU_CHECK_SUPPORTED_OR_ERROR(c10_npu::acl::AclrtSetOpWaitTimeout(kOpWaitTimeout));
    ASCEND_LOGI(
        "Get env HCCL_EXEC_TIMEOUT value %u, and set op wait timeout to %u.", hccl_exec_timeout, kOpWaitTimeout);

    char* blockingWait = getenv(HCCL_BLOCKING_WAIT);
    try {
        if (blockingWait != nullptr) {
            auto val = std::stoi(blockingWait);
            if (val == 1) {
                // Make wait() and synchronize() a blocking call.
                blockingWait_ = true;
            } else if (val != 0) {
                throw std::runtime_error("Invalid value for environment variable: " + std::string(HCCL_BLOCKING_WAIT) + DIST_ERROR(ErrCode::VALUE));
            }
        }
    } catch (std::exception& e) {
        throw std::runtime_error("Invalid value for environment variable: " + std::string(HCCL_BLOCKING_WAIT) + DIST_ERROR(ErrCode::VALUE));
    }

    asyncErrorHandling_ =
        static_cast<ErrorHandlingMode>(c10_npu::option::OptionsManager::CheckUseHcclAsyncErrorHandleEnable());
    desyncDebug_ = static_cast<bool>(c10_npu::option::OptionsManager::CheckUseDesyncDebugEnable());

    if (blockingWait_) {
        if (asyncErrorHandling_ != NoHandling || desyncDebug_) {
            LOG(INFO) << "[Rank " << rank_ << "] HCCL_BLOCKING_WAIT and "
                      << "HCCL_ASYNC_ERROR_HANDLING|HCCL_DESYNC_DEBUG"
                      << "should not both be enabled. "
                      << "Only HCCL_BLOCKING_WAIT is being used in this process.";
            asyncErrorHandling_ = NoHandling;
            desyncDebug_ = false;
        }
    } else {
        if (desyncDebug_ && asyncErrorHandling_ == NoHandling) {
            LOG(INFO) << "[Rank " << rank_ << "] HCCL_DESYNC_DEBUG and HCCL_ASYNC_ERROR_HANDLING "
                      << "must both be enabled. "
                      << "Enabling HCCL_ASYNC_ERROR_HANDLING.";
            asyncErrorHandling_ = TearDown;
        }
    }

#ifdef ENABLE_HCCL_ERROR_CHECKING
    if (hccl_exec_timeout > 0) {
        if ((hccl_exec_timeout * 1000) > (options_->opTimeout).count()) {
            TORCH_NPU_WARN("The HCCL execution timeout ", hccl_exec_timeout*1000, "ms is bigger than watchdog timeout ",
                (options_->opTimeout).count(), "ms which is set by init_process_group! The plog may not be recorded.");
        }
    } else {
        if ((options_->opTimeout).count() == DEFAULT_TIMEOUT) {
            // Only when the timeout is default, we will change it.
            options_->opTimeout = std::chrono::milliseconds(DEFAULT_TIMEOUT*2);
        }
        if ((options_->opTimeout).count() < DEFAULT_TIMEOUT) {
            TORCH_NPU_WARN("The HCCL execution timeout ", DEFAULT_TIMEOUT, "ms is bigger than watchdog timeout ",
                (options_->opTimeout).count(), "ms which is set by init_process_group! The plog may not be recorded.");
        }
    }
    hcclCommWatchdogThread_ = std::thread(&ProcessGroupHCCL::hcclCommWatchdog, this);
#endif
    if (asyncErrorHandling_ != NoHandling) {
        workCleanupThread_ = std::thread(&ProcessGroupHCCL::workCleanupLoop, this);
    }
}

void ProcessGroupHCCL::workCleanupLoop()
{
    bool threadSetDeviceFlag = true;
    while (!terminateProcessGroup_.load()) {
        std::list<WorkHCCL> doneWorks;
        {
            std::unique_lock<std::mutex> lock(workMetaListMutex_);
            // We busy-poll the work vector every kWatchdogThreadSleepMillis
            // milliseconds as long as the atomic is True.
            workMetaListCV_.wait_for(lock, std::chrono::milliseconds(kWorkCleanupThreadSleepMillis), [&]() -> bool {
                return terminateProcessGroup_.load();
            });

            for (auto it = workMetaList_.begin(); it != workMetaList_.end();) {
                auto& work = *it;
                if (threadSetDeviceFlag) {
                    NPU_CHECK_ERROR(c10_npu::SetDevice(static_cast<int>(work.devices_[0].index())));
                    threadSetDeviceFlag = false;
                }

                if (desyncDebug_ && !work.exception()) {
                    if (!work.startTraceUpdated_ && work.isStarted() && !terminateProcessGroup_.load() &&
                        !storeError_) {
                        work.startTraceUpdated_ = true;
                        storeError_ =
                            !c10d::traceUpdate(store_, traceKeyStart_, work.seq_, opTypeToString(work.opType_));
                    }
                }

                if (work.isCompleted()) {
                    if (desyncDebug_ && !work.exception()) {
                        // To close the window between the check of work.isStarted() and
                        // the check of work.isCompleted().
                        if (!work.startTraceUpdated_ && !terminateProcessGroup_.load() && !storeError_) {
                            storeError_ =
                                !c10d::traceUpdate(store_, traceKeyStart_, work.seq_, opTypeToString(work.opType_));
                        }
                        if (!terminateProcessGroup_.load() && !storeError_) {
                            storeError_ =
                                !c10d::traceUpdate(store_, traceKeyEnd_, work.seq_, opTypeToString(work.opType_));
                        }
                    }
                    // Handle Exceptions on failed GPU operations and remove completed
                    // workHCCL objects from work vector.
                    if (!terminateProcessGroup_.load()) {
                        work.handleExceptionGuard(asyncErrorHandling_);
                    }
                    doneWorks.emplace_back(std::move(*it));
                    if (*(work.is_dispatched) && work.is_reported) {
                        ASCEND_LOGE("Process group work %s, seq_num %u dispatch sucess. This error log can be ignored.", opTypeToString(work.opType_).c_str(), work.seq_);
                        work.is_reported = false;
                    }
                    it = workMetaList_.erase(it);
                } else {
                    // Increment the iterator if the current WorkHCCL object is not
                    // completed.
                    ++it;
                }
            }
        }
        doneWorks.clear();
    }
}

void ProcessGroupHCCL::workEnqueue(c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL> work)
{
    if (!terminateProcessGroup_.load()) {
        std::lock_guard<std::mutex> lock(workMetaListMutex_);
        // Avoid view tensors to be processed in cleanup thread.
        // View tensors' destruction invokes autograd_meta, which
        // needs to be destructed in user thread. Otherwise will
        // get deadlock. Here we enqueue work without outputs_.
        workMetaList_.emplace_back(*work);
    }
}

void ProcessGroupHCCL::setSequenceNumberForGroup() {}

uint64_t ProcessGroupHCCL::getSequenceNumberForGroup()
{
    return seq_;
}

ProcessGroupHCCL::~ProcessGroupHCCL()
{
    terminateProcessGroup_.store(true);
    watchdogCV_.notify_one();

#ifdef ENABLE_HCCL_ERROR_CHECKING
    hcclCommWatchdogThread_.join();
#endif
    if (asyncErrorHandling_ != NoHandling) {
        workMetaListCV_.notify_one();
        workCleanupThread_.join();
    }
    {
        // Destropy all HCCL Communicators on Process Group Destruction
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& it : devHCCLCommMap_) {
            auto& hcclComms = it.second;

            for (const auto& hcclComm : hcclComms) {
                hcclComm->destropyHcclComm();
            }
        }
    }
}

void ProcessGroupHCCL::broadcastMasterID(HcclRootInfo* hcclID)
{
    // For every HCCL communicator that we create we need to broadcast
    // a unique ID from rank 0 to all other ranks. This broadcast is
    // done by rank 0 setting a key in the store and all other ranks
    // retrieving the contents of that key. A single process group
    // may create multiple HCCL communicators, so we use a sequence
    // number to differentiate between them.
    std::string storeKey = std::to_string(hcclCommCounter_++);
    std::string ver_key = "version_key";
    auto date_list = __DATE__ != nullptr ? __DATE__ : "";
    std::vector<uint8_t> ver_list;
#ifdef PYTORCH_NPU_VERSION
    auto py_list = PYTORCH_NPU_VERSION != nullptr ? PYTORCH_NPU_VERSION : "";
    ver_list.insert(ver_list.end(), py_list, py_list + strlen(py_list));
#endif
    ver_list.insert(ver_list.end(), date_list, date_list + strlen(date_list));
    if (rank_ == 0) {
        auto vec = std::vector<uint8_t>(
            reinterpret_cast<uint8_t*>(hcclID), reinterpret_cast<uint8_t*>(hcclID) + HCCL_ROOT_INFO_BYTES);
        store_->set(storeKey, vec);
        store_->set(ver_key, ver_list);
    } else {
        try {
            auto vec = store_->get(storeKey);
            TORCH_CHECK(vec.size() == HCCL_ROOT_INFO_BYTES, DIST_ERROR(ErrCode::PARAM));
            std::memcpy(hcclID, vec.data(), vec.size());
        } catch (const std::exception& e) {
            throw std::runtime_error("store->get() got error: " + std::string(HCCL_BLOCKING_WAIT) + DIST_ERROR(ErrCode::INTERNAL));
        } catch (...) {
            throw std::runtime_error("Unknown exception: " + std::string(HCCL_BLOCKING_WAIT) + DIST_ERROR(ErrCode::INTERNAL));
        }
        auto main_list = store_->get(ver_key);
        if (main_list != ver_list) {
            TORCH_WARN("PTA version mismatch");
        }
    }
}

// record data volume for HCCL op.
void ProcessGroupHCCL::recordDataVol(std::string opName, const std::string dataVol, const int currRank,
    std::vector<std::shared_ptr<HCCLComm>>& hcclComms)
{
    std::ofstream outfile;
    std::stringstream fileName;
    std::string commName = getHcclCommNameWithoutInit(currRank, hcclComms);
    auto master_addr = getenv("MASTER_ADDR");
    TORCH_CHECK(master_addr != nullptr, "Unable to fetch master IP addr, environment variable is null.", DIST_ERROR(ErrCode::NOT_FOUND));
    fileName << master_addr << "_" << commName << "_" << std::to_string(currRank) << ".log";
    std::string out_file_path = c10::str(nslb_path, "/", fileName.str());
    try {
        if (access(nslb_path, W_OK) != 0 && mkdir(nslb_path, S_IRWXU | S_IRGRP | S_IXGRP) != 0) {
            throw std::exception();
        }
        outfile.open(out_file_path, std::ios::app);
    } catch (std::exception& e) {
        throw std::runtime_error("Open shared directory failed. Please check whether input path is valid."
                                 + DIST_ERROR(ErrCode::NOT_FOUND));
    }
    std::transform(opName.begin(), opName.end(), opName.begin(), ::tolower);
    outfile << opName << " " << dataVol << " " << std::to_string(currRank) << "\n";
    outfile.close();
}

std::vector<std::shared_ptr<HCCLComm>>& ProcessGroupHCCL::getHCCLComm(
    const std::string& devicesKey,
    const std::vector<at::Device>& devices)
{
    // Sanity check
    if (devicesKey.empty()) {
        throw std::runtime_error(
            "Not able to create/get the HCCL Communicator since "
            "the NPU devices are not known" + DIST_ERROR(ErrCode::HCCL));
    }

    for (auto& device : devices) {
        usedDeviceIdxs_.insert(device.index());
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
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
        HCCL_CHECK_ERROR(HcclGetRootInfo(&hcclID));
    }
    broadcastMasterID(&hcclID);

    c10_npu::OptionalNPUGuard npuGuard;
    std::vector<c10_npu::NPUStream> streamVal;
    streamVal.reserve(devices.size());

    for (size_t i = 0; i < devices.size(); ++i) {
        int numRanks = getSize();
        int rank = getRank() * static_cast<int>(devices.size()) + static_cast<int>(i);

        npuGuard.set_index(devices[i].index());
        hcclComms[i] = HCCLComm::create(numRanks, rank, hcclID);

        // Creates the HCCL streams
        streamVal.push_back(c10_npu::getNPUStreamFromPool(devices[i].index()));
    }

    hcclStreams_.emplace(devicesKey, std::move(streamVal));

    // Note: these events are created with the (default) cudaEventDisableTiming
    // flag This flag provides the best performance when used with
    // StreamWaitEvent() and EventQuery(). Since we here don't measure the
    // performance using npuEvent, this should be set.
    hcclEvents_.emplace(std::piecewise_construct, std::make_tuple(devicesKey), std::make_tuple(devices.size()));

    // stream length is 1024,
    rateCtrlEvents_.emplace(std::piecewise_construct, std::make_tuple(devicesKey), std::make_tuple(devices.size()));

    // record collectiveCnts.
    collectiveCnts_.emplace(std::piecewise_construct, std::make_tuple(devicesKey), std::make_tuple(devices.size()));

    // Hold the lock before modifying the cache.
    std::lock_guard<std::mutex> lock(mutex_);

    // Record the communicators based on HCCLUniqueId.
    hcclIdToCommMap_.emplace(buildHcclUniqueIdStr(hcclID), hcclComms);

    // Move the HCCL resource to cache
    devHCCLCommMap_.emplace(devicesKey, std::move(hcclComms));
    return devHCCLCommMap_[devicesKey];
}

int64_t ProcessGroupHCCL::getStreamId(bool p2p)
{
    int device;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
    std::vector<at::Device> devices = {at::Device(at_npu::key::NativeDeviceType, device)};
    auto key = getKeyFromDevices(devices);
    if ((!hcclStreams_.count(key)) || hcclStreams_[key].empty()) {
        return -1;
    }
    return hcclStreams_[key][0].id();
}

namespace {

// Check that all `tensors' have the same type and shape and are distributed
// across distinct NPUs.
void check_npu_tensors_different_devices(const std::vector<at::Tensor>& tensors)
{
    if (tensors.size() == 0) {
        TORCH_CHECK(false, "Tensor list must be nonempty", DIST_ERROR(ErrCode::PARAM));
    }
    // HCCL support one NPU per process only
    if (tensors.size() != 1) {
        TORCH_CHECK(false, "Tensor list mustn't be larger than the number of available NPUs", DIST_ERROR(ErrCode::PARAM));
    }
    const auto& first = tensors.front();
    // Set for ensuring that tensors are on separate devices.
    std::unordered_set<decltype(first.get_device())> usedDevices;
    usedDevices.reserve(tensors.size());

    for (const auto& t : tensors) {
        if (!torch_npu::utils::is_npu(t) || t.is_sparse()) {
            TORCH_CHECK(false, "Tensors must be NPU and dense", DIST_ERROR(ErrCode::TYPE));
        }
        if (t.scalar_type() != first.scalar_type()) {
            TORCH_CHECK(false, "Tensors must have identical type", DIST_ERROR(ErrCode::TYPE));
        }
        if (t.sizes() != first.sizes()) {
            TORCH_CHECK(false, "Tensors must have identical size", DIST_ERROR(ErrCode::TYPE));
        }
        if (t.strides() != first.strides()) {
            TORCH_CHECK(false, "Tensors must have identical strides", DIST_ERROR(ErrCode::TYPE));
        }
        if (!t.is_contiguous(t.suggest_memory_format())) {
            TORCH_CHECK(false, "Tensors must be contiguous", DIST_ERROR(ErrCode::TYPE));
        }
        const auto inserted = usedDevices.insert(t.get_device()).second;
        if (!inserted) {
            TORCH_CHECK(false, "Tensors must be on distinct NPU devices", DIST_ERROR(ErrCode::TYPE));
        }
    }
}

// check validity of single tensor
void check_npu_single_tensor(const at::Tensor& tensor)
{
    if (!torch_npu::utils::is_npu(tensor) || tensor.is_sparse()) {
        TORCH_CHECK(false, "Tensors must be NPU and dense", DIST_ERROR(ErrCode::TYPE));
    }
    if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
        TORCH_CHECK(false, "Tensors must be contiguous", DIST_ERROR(ErrCode::TYPE));
    }
}

std::vector<at::Tensor> cast_to_origin_format(const std::vector<at::Tensor>& inputTensors)
{
    std::vector<at::Tensor> inputTensors_;
    inputTensors_.resize(inputTensors.size());
    size_t index = 0;
    for (auto& tensor : inputTensors) {
        if (at_npu::native::FormatHelper::IsBaseFormatType(tensor)) {
            inputTensors_[index] = tensor;
        } else {
            auto origin_format = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_.origin_format_;
            inputTensors_[index] = at_npu::native::custom_ops::npu_format_cast(tensor, origin_format);
        }
        index++;
    }
    return inputTensors_;
}

std::vector<at::Tensor> create_base_format_tensors(const std::vector<at::Tensor>& inputTensors)
{
    std::vector<at::Tensor> inputTensors_;
    inputTensors_.resize(inputTensors.size());
    for (size_t i = 0; i < inputTensors.size(); ++i) {
        if (at_npu::native::FormatHelper::IsBaseFormatType(inputTensors[i])) {
            inputTensors_[i] = inputTensors[i];
        } else {
            auto options = at::TensorOptions().dtype(inputTensors[i].dtype()).device(inputTensors[i].device());
            inputTensors_[i] = at_npu::native::NPUNativeFunctions::empty(
                inputTensors[i].sizes(),
                options.dtype().toScalarType(),
                options.layout_opt(),
                options.device_opt(),
                options.pinned_memory_opt(),
                options.memory_format_opt());
        }
    }
    return inputTensors_;
}

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_for_scatter_gather(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other,
    size_t world_size)
{
    if (tensor_lists.size() != other.size()) {
        TORCH_CHECK(false, "Tensor list operands to scatter/gather must have the same length", DIST_ERROR(ErrCode::PARAM));
    }
    const auto num_devices = tensor_lists.size();

    std::vector<at::Tensor> flattened;
    flattened.resize(num_devices);

    for (auto i = size_t{}; i < num_devices; ++i) {
        if (tensor_lists[i].size() != world_size * num_devices) {
            TORCH_CHECK(
                false,
                "Tensor list input to scatter/gather must match number of collective"
                " participants", DIST_ERROR(ErrCode::PARAM));
        }

        // Only check device match for the first tensor in the list; the call to
        // newLikeFlat() below will check the rest.
        if (tensor_lists[i].front().get_device() != other[i].get_device()) {
            TORCH_CHECK(
                false,
                "Corresponding input/output tensors to scatter/gather must all reside"
                " on the same device", DIST_ERROR(ErrCode::PARAM));
        }

        for (const auto& t : tensor_lists[i]) {
            if (t.numel() != other[i].numel()) {
                TORCH_CHECK(false, "All tensor operands to scatter/gather must have the same size", DIST_ERROR(ErrCode::PARAM));
            }
        }
        // Flatten the tensors (from all ranks) into a single big tensor.
        flattened[i] = c10d::newLikeFlat(tensor_lists, i);
    }
    return flattened;
}

void nslb_record_end()
{
    std::string end_file_path;
    std::ofstream endfile;
    end_file_path = c10::str(nslb_path, "/end_", getenv("MASTER_ADDR"), "_", getpid(), ".log");
    try {
        if (access(nslb_path, W_OK) != 0 && mkdir(nslb_path, S_IRWXU | S_IRGRP | S_IXGRP) != 0) {
            throw std::exception();
        }
        endfile.open(end_file_path.c_str(), std::ios::out);
        endfile.close();
    } catch (std::exception& e) {
        throw std::runtime_error("NSLB set end failed." + DIST_ERROR(ErrCode::NOT_FOUND));
    }
}

} // namespace

c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL> ProcessGroupHCCL::initWork(
    std::vector<at::Device> devices,
    int rank,
    c10d::OpType opType)
{
    if (devices.size() != 1) {
        throw std::runtime_error("ProcessGroupHCCL support one device per process only" + DIST_ERROR(ErrCode::NOT_SUPPORT));
    }
    return c10::make_intrusive<ProcessGroupHCCL::WorkHCCL>(devices, rank, opType, seq_, desyncDebug_);
}

ProcessGroupHCCL::Options::Options(bool is_high_priority_stream)
    : opTimeout(kProcessGroupHCCLOpTimeoutMillis), is_high_priority_stream(is_high_priority_stream)
{
}

int64_t ProcessGroupHCCL::getHcclComm(int rankid) {
  at::Device device = getDeviceForRank(rankid);
  std::vector<at::Device> devices = {device};
  const auto key = getKeyFromDevices(devices);
  auto& hcclComms = getHCCLComm(key, devices);
  TORCH_CHECK(hcclComms.size() == 1, "expect hcclComms.size() = 1, but hcclComms.size() = ",
      hcclComms.size(), DIST_ERROR(ErrCode::PARAM));
  auto ret_hcom = hcclComms[0]->getHcclComm();
  int64_t hccl_comm = static_cast<int64_t>(reinterpret_cast<intptr_t>(ret_hcom));
  return hccl_comm; 
}

std::string ProcessGroupHCCL::getHcclCommName(int rankid) {
  at::Device device = getDeviceForRank(rankid);
  std::vector<at::Device> devices = {device};
  const auto key = getKeyFromDevices(devices);
  auto& hcclComms = getHCCLComm(key, devices);
  auto name_str = getHcclCommNameWithoutInit(rankid, hcclComms);
  return name_str;
}

std::string ProcessGroupHCCL::getHcclCommNameWithoutInit(int rankid, std::vector<std::shared_ptr<HCCLComm>>& hcclComms)
{
    TORCH_CHECK(hcclComms.size() == 1, "expect hcclComms.size() = 1, but hcclComms.size() = ",
        hcclComms.size(), DIST_ERROR(ErrCode::PARAM));
    HcclComm ret_hcom = hcclComms[0]->getHcclComm();
    char commName[MAX_GROUP_NAME];
    HCCL_CHECK_ERROR(at_npu::native::hccl::HcclGetCommNameFace(ret_hcom, commName));
    std::string name_str(commName);
    return name_str;
}

void ProcessGroupHCCL::silenceCheck(at::Tensor &input, c10d::OpType opType)
{
    if (input.scalar_type() != at::kFloat && input.scalar_type() != at::kBFloat16) {
        return;
    }

    if (input.requires_grad()) {
        return;
    }

    if (opType != c10d::OpType::SEND && opType != c10d::OpType::RECV && opType != c10d::OpType::UNKNOWN) {
        if (c10_npu::model_state().get_call_state() != c10_npu::CallStateMode::L_BACKWARD) {
            return;
        }
        // Barrier will call allreduce. This is used to filter out invalid allreduce operator calls.
        if (opType == c10d::OpType::ALLREDUCE && input.numel() <= 1) {
            return;
        }
    }
    if (silenceCheckCache_.find(opType) == silenceCheckCache_.end()) {
        at::Tensor stepTensor = at::zeros({1}, input.options().dtype(at::kLong));
        at::Tensor cacheTensor = at::zeros({3}, input.options().dtype(at::kFloat));
        silenceCheckCache_.emplace(opType, std::make_pair(std::move(stepTensor), std::move(cacheTensor)));
    }
    at::Tensor val = at::norm(input);
    static double min_steps = 100.0;
    op_plugin::_npu_silent_check_v2(val, input, silenceCheckCache_[opType].second, silenceCheckCache_[opType].first, min_steps,
        c10_npu::option::OptionsManager::GetSilenceUpperThresh().first, c10_npu::option::OptionsManager::GetSilenceSigmaThresh().first,
        c10_npu::option::OptionsManager::GetSilenceUpperThresh().second, c10_npu::option::OptionsManager::GetSilenceSigmaThresh().second,
        static_cast<int64_t>(c10_npu::option::OptionsManager::GetSilenceCheckFlag()));
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    c10d::OpType opType)
{
    // Bump collective counter
    seq_++;

    const auto devices = getDeviceList(inputs);
    const auto key = getKeyFromDevices(devices);
    auto& hcclComms = getHCCLComm(key, devices);
    // Used many times below, so we stash the unordered_map lookup
    auto& hcclStreams = hcclStreams_[key];
    // First let HCCL streams wait for input tensors allocation streams
    syncStreams(devices, hcclEvents_[key], hcclStreams);
    // Work itself will create the events on all NPUs of tensors
    auto work = initWork(devices, rank_, opType);

    c10_npu::OptionalNPUGuard npuGuard;

    if (desyncDebug_) {
        for (const auto i : c10::irange(devices.size())) {
            c10_npu::NPUStream& hcclStream = hcclStreams[i];
            (*work->hcclStartEvents_)[i].record(hcclStream);
        }
    }

    if (c10_npu::model_state().get_model_mode() == c10_npu::ModelMode::L_TRAIN
        && c10_npu::option::OptionsManager::GetSilenceCheckFlag() != c10_npu::option::CHECK_CLOSE) {
        for (const auto i : c10::irange(inputs.size())) {
            npuGuard.set_index(devices[i].index());
            c10_npu::NPUStreamGuard guard(hcclStreams[i]);
            silenceCheck(inputs[i], opType);
        }
    }

    pre(hcclStreams, work);

    if (nslb_path != nullptr && !nslb_is_end) {
        auto nslb_num = c10_npu::option::OptionsManager::GetNslbCntVal();
        if (seq_ <= nslb_num) {
            size_t dataVol = 0;
            for (auto tensor:inputs) {
                dataVol += tensor.storage().nbytes();
            }
            char* global_rank = getenv("RANK");
            TORCH_CHECK(global_rank != nullptr, "Unable to fetch global rank for NSLB.");
            recordDataVol(opTypeToString(opType), std::to_string(dataVol), atoi(global_rank), hcclComms);
        }
        if (seq_ >= nslb_num) {
            nslb_is_end = true;
            nslb_record_end();

        }
    }

    for (const auto i : c10::irange(inputs.size())) {
        npuGuard.set_index(devices[i].index());
        c10_npu::NPUStream& hcclStream = hcclStreams[i];

        // Both `inputs' and `outputs' are created on a worker stream and used in
        // different hcclStreams.  Hence, both must record the hcclStream to
        // prevent being freed before the collective finishes.
        //
        // We only record `inputs' here, and leave recording `outputs' to `fn' for
        // operations where `inputs' and `outputs' are not the same.
        //
        // See [Sync Streams].
        if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::AVOID_RECORD_STREAM) {
            work->stashed_for_allocator_safety_.push_back(inputs[i]);
        } else {
            c10_npu::NPUCachingAllocator::recordStream(inputs[i].storage().data_ptr(), hcclStream);
            if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::ERASE_RECORD_STREAM) {
                work->recorded_inputs_.push_back(std::make_pair(inputs[i].storage().getWeakStorageImpl(), hcclStream));
            }
        }
    }
    {
        for (const auto i : c10::irange(inputs.size())) {
            npuGuard.set_index(devices[i].index());
            // to avoid to much task pushed to the stream, leading to stream overflow
            // insert sync point fluxLimit(key, i)
            c10_npu::NPUStream& hcclStream = hcclStreams[i];
            hcclUs startut = TIME_NOW();
            HCCL_CHECK_ERROR(fn(inputs[i], outputs[i], hcclComms[i]->getHcclComm(), hcclStream, work->is_dispatched), opTypeToString(opType).c_str());
            if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::ERASE_RECORD_STREAM) {
                work->recorded_outputs_.push_back(
                    std::make_pair(outputs[i].storage().getWeakStorageImpl(), hcclStream));
            } else if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::AVOID_RECORD_STREAM) {
                work->stashed_for_allocator_safety_.push_back(outputs[i]);
            }
        }
    }
    post(hcclStreams, work);

    for (size_t i = 0; i < inputs.size(); ++i) {
        c10_npu::NPUStream& hcclStream = hcclStreams_[key][i];
        (*work->npuEvents_)[i].record(hcclStream);
        ASCEND_LOGI("Event: record hccl work is successfully executed, event=%p", (*(work->npuEvents_))[i].event());
        work->hcclComms_[i] = hcclComms[i];
    }
    work->blockingWait_ = blockingWait_;
    work->opTimeout_ = opTimeout_;
    work->store_ = store_;
    if (asyncErrorHandling_ != NoHandling) {
        workEnqueue(work);
    }
    return work;
}

template <typename Fn>
c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    c10d::OpType opType)
{
    return collective(
        inputs,
        outputs,
        fn,
        [](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {},
        [](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {},
        opType);
}

int g_allreduceID = 0;
c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceOptions& opts)
{
    check_npu_tensors_different_devices(tensors);
    std::string functionName = __FUNCTION__;
    return collective(
        tensors,
        tensors,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            aclrtSetExceptionInfoCallback(exceptionCallback);

            auto hcclType = getHcclDataType(input.scalar_type());
            checkSupportedDataType(hcclType, functionName);
            RECORD_FUNCTION("HcclAllreduce", std::vector<c10::IValue>({input}));

            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto hcclReduceOp = getHcclReduceOp(opts.reduceOp, input);
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream, is_dispatched]() -> int {
                auto hccl_result = HcclAllReduce(
                    inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclAllreduce");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

            return HCCL_SUCCESS;
        },
        c10d::OpType::ALLREDUCE);
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::allreduce_out(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int64_t fusion_id,
    const c10d::AllreduceOptions& opts)
{
    check_npu_tensors_different_devices(inputs);
    check_npu_tensors_different_devices(outputs);
    std::string functionName = __FUNCTION__;
    return collective(
        inputs,
        outputs,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            aclrtSetExceptionInfoCallback(exceptionCallback);

            auto hcclType = getHcclDataType(input.scalar_type());
            checkSupportedDataType(hcclType, functionName);
            RECORD_FUNCTION("HcclAllreduce", std::vector<c10::IValue>({input}));
            int64_t hccl_comm = static_cast<int64_t>(reinterpret_cast<intptr_t>(comm));
            at_npu::native::NPUNativeFunctions::npu_hcom_allreduce_out(
                input, "sum", "hccl_world_group", 2, fusion_id, 1, 0, hccl_comm, output);
            *is_dispatched = true;
            return HCCL_SUCCESS;
        },
        c10d::OpType::ALLREDUCE);
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::batch_isend_irecv(
    std::vector<std::string>& op_type,
    std::vector<at::Tensor>& tensors,
    std::vector<uint32_t> remote_rank_list)
{
    std::vector<at::Tensor> tensors_tmp = {tensors[0]};
    return collective(
        tensors_tmp,
        tensors_tmp,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            RECORD_FUNCTION("HcclBatchSendRecv", std::vector<c10::IValue>({input}));
			auto itemNum = static_cast<uint32_t>(op_type.size());
			std::vector<void *> tensor_ptr_list;
			std::vector<uint64_t> numel_list;
			std::vector<HcclDataType> type_list;
			for (size_t i = 0; i < op_type.size(); ++i) {
			    tensor_ptr_list.push_back(tensors[i].data_ptr());
			    numel_list.push_back(getNumelForHCCL(tensors[i]));
			    type_list.push_back(getHcclDataType(tensors[i].scalar_type()));
			}
			auto hccl_call = [tensor_ptr_list, numel_list, type_list, remote_rank_list, op_type, itemNum, comm, stream, is_dispatched]() -> int {
			HcclSendRecvItem sendRecvInfo[itemNum];
		    HcclSendRecvType currType;
		    for (size_t i = 0; i < op_type.size(); ++i) {
		        if (op_type[i] == "isend") {
		            currType = HcclSendRecvType::HCCL_SEND;
		        } else if (op_type[i] == "irecv") {
		            currType = HcclSendRecvType::HCCL_RECV;
		        } else {
		            currType = HcclSendRecvType::HCCL_SEND_RECV_RESERVED;
		        }
		        sendRecvInfo[i] = HcclSendRecvItem{currType,
		                                           tensor_ptr_list[i],
		                                           numel_list[i],
		                                           type_list[i],
		                                           remote_rank_list[i]
		                                           };
            }
			    auto hccl_result = hcclBatchIsendIrecv(sendRecvInfo, itemNum, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
			};
			at_npu::native::OpCommand cmd;
            cmd.Name("HcclBatchSendRecv");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();
            return HCCL_SUCCESS;
        },
        c10d::OpType::UNKNOWN);
}

int g_broadcastID = 100000;
c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const c10d::BroadcastOptions& opts)
{
    check_npu_tensors_different_devices(tensors);
    return collective(
        tensors,
        tensors,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            RECORD_FUNCTION("HcclBroadcast", std::vector<c10::IValue>({input}));
            const auto root = opts.rootRank * tensors.size() + opts.rootTensor;

            auto inputDataPtr = input.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto hcclType = getHcclDataType(input.scalar_type());
            auto hccl_call = [inputDataPtr, numel, hcclType, root, comm, stream, is_dispatched]() -> int {
                auto hccl_result = HcclBroadcast(inputDataPtr, numel, hcclType, root, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclBroadcast");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

            return HCCL_SUCCESS;
        },
        c10d::OpType::BROADCAST);
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const c10d::AllreduceCoalescedOptions& /* unused */)
{
    throw std::runtime_error("ProcessGroupHCCL does not support allreduce_coalesced" + DIST_ERROR(ErrCode::NOT_SUPPORT));
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const c10d::ReduceOptions& opts)
{
    check_npu_tensors_different_devices(tensors);
    std::string functionName = __FUNCTION__;
    uint64_t rank = opts.rootRank;
    return collective(
        tensors,
        tensors,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            auto hcclType = getHcclDataType(input.scalar_type());
            checkSupportedDataType(hcclType, functionName);
            RECORD_FUNCTION("HcclReduce", std::vector<c10::IValue>({input}));

            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto reduceOp = getHcclReduceOp(opts.reduceOp, input);
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, reduceOp, rank, comm, stream, is_dispatched]() -> int {
                auto hccl_result = hcclReduce(
                    inputDataPtr, outputDataPtr, numel, hcclType, reduceOp, rank, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclReduce");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

            return HCCL_SUCCESS;
        },
        c10d::OpType::REDUCE);
}

#define ADDRESS_ALIGNMENT_BYTE 512
at::Tensor ProcessGroupHCCL::byte_alignment(at::Tensor& tensors)
{
    at::Tensor inter_tensors = at::reshape(tensors, {1, tensors.numel()});
    if (tensors.element_size() == 0) {
        return inter_tensors;
    }

    int64_t tensor_byte = tensors.numel() * tensors.element_size();
    int64_t byte_add = (tensor_byte % ADDRESS_ALIGNMENT_BYTE == 0)
        ? 0
        : (ADDRESS_ALIGNMENT_BYTE - tensor_byte % ADDRESS_ALIGNMENT_BYTE);
    int64_t num_add = byte_add / tensors.element_size();
    if (num_add != 0) {
        bool transflag = false;
        if (inter_tensors.scalar_type() == at::ScalarType::Bool) {
            inter_tensors = at_npu::native::custom_ops::npu_dtype_cast(inter_tensors, at::ScalarType::Int);
            transflag = true;
        }

        inter_tensors = op_plugin::constant_pad_nd(inter_tensors, {0, num_add}, 0);

        if (transflag == true) {
            inter_tensors = at_npu::native::custom_ops::npu_dtype_cast(inter_tensors, at::ScalarType::Bool);
        }
    }
    return inter_tensors;
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts)
{
    check_npu_tensors_different_devices(inputTensors);
    auto inputTensors_ = cast_to_origin_format(inputTensors);

    int outsize = static_cast<int>(outputTensors[0].size());
    uint64_t output_nums[outsize];
    for (const auto i : c10::irange(outputTensors.size())) {
        for (const auto j : c10::irange(outsize)) {
            output_nums[j] = static_cast<uint64_t>(outputTensors[0][j].numel());
        }
    }

    std::vector<at::Tensor> byte_alignment_inputTensors_ = {byte_alignment(inputTensors_[0])};
    std::vector<at::Tensor> byte_alignment_outputTensors_;
    for (unsigned int i = 0; i < outputTensors[0].size(); i++) {
        byte_alignment_outputTensors_.push_back(byte_alignment(outputTensors[0][i]));
    }
    std::vector<std::vector<at::Tensor>> byte_alignment_outputTensors;
    byte_alignment_outputTensors.push_back(byte_alignment_outputTensors_);

    auto outputFlattened =
        flatten_for_scatter_gather(byte_alignment_outputTensors, byte_alignment_inputTensors_, size_);
    check_npu_tensors_different_devices(outputFlattened);

    return collective(
        byte_alignment_inputTensors_,
        outputFlattened,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            RECORD_FUNCTION("HcclAllgather", std::vector<c10::IValue>({input}));

            if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() != c10_npu::option::AVOID_RECORD_STREAM) {
                c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
            }

            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto hcclType = getHcclDataType(input.scalar_type());
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, comm, stream, is_dispatched]() -> int {
                auto hccl_result = HcclAllGather(inputDataPtr, outputDataPtr, numel, hcclType, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclAllgather");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {},
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {
            work->lazyDestory(byte_alignment_inputTensors_);
            work->lazyDestory(outputFlattened);
            // Copy the flattened output tensors to the outputs.
            for (const auto i : c10::irange(outputTensors.size())) {
                c10_npu::NPUStreamGuard guard(hcclStreams[i]);
                for (const auto j : c10::irange(outputTensors[0].size())) {
                    // See [Sync Streams].
                    if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::AVOID_RECORD_STREAM) {
                        work->stashed_for_allocator_safety_.push_back(outputTensors[i][j]);
                    } else {
                        c10_npu::NPUCachingAllocator::recordStream(
                            outputTensors[i][j].storage().data_ptr(), hcclStreams[i]);
                        if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::ERASE_RECORD_STREAM) {
                            work->recorded_outputs_.push_back(
                                std::make_pair(outputTensors[i][j].storage().getWeakStorageImpl(), hcclStreams[i]));
                        }
                    }
                    at::Tensor output_tensor = outputFlattened[i][j].slice(1, 0, output_nums[j]);
                    at::Tensor output_tensor_shape = at::reshape(output_tensor, outputTensors[i][j].sizes());
                    outputTensors[i][j].copy_(output_tensor_shape, true);
                }
            }
        },
        c10d::OpType::ALLGATHER);
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::allgather_togather(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts)
{
    check_npu_tensors_different_devices(inputTensors);
    check_npu_tensors_different_devices(outputTensors);
    auto inputTensors_ = cast_to_origin_format(inputTensors);

    return collective(
        inputTensors_,
        outputTensors,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            RECORD_FUNCTION("HcclAllgatherTogather", std::vector<c10::IValue>({input}));
            if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() != c10_npu::option::AVOID_RECORD_STREAM) {
                c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
            }
            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto hcclType = getHcclDataType(input.scalar_type());
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, comm, stream, is_dispatched]() -> int {
                auto hccl_result = HcclAllGather(inputDataPtr, outputDataPtr, numel, hcclType, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclAllGather");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {},
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {},
        c10d::OpType::ALLGATHER);
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::_allgather_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const c10d::AllgatherOptions& opts)
{
    if (inputTensor.dtype() != outputTensor.dtype()) {
        TORCH_CHECK(false, "output tensor must have the same type as input tensor", DIST_ERROR(ErrCode::PARAM));
    }
    if (inputTensor.numel() * size_ != outputTensor.numel()) {
        TORCH_CHECK(false, "output tensor size must be equal to world_size times input tensor size", DIST_ERROR(ErrCode::PARAM));
    }
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    check_npu_tensors_different_devices(inputTensors);
    check_npu_tensors_different_devices(outputTensors);
    auto inputTensors_ = cast_to_origin_format(inputTensors);

    return collective(
        inputTensors_,
        outputTensors,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            RECORD_FUNCTION("HcclAllgatherBase", std::vector<c10::IValue>({input}));
            if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() != c10_npu::option::AVOID_RECORD_STREAM) {
                c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
            }
            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto hcclType = getHcclDataType(input.scalar_type());
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, comm, stream, is_dispatched]() -> int {
                auto hccl_result = HcclAllGather(inputDataPtr, outputDataPtr, numel, hcclType, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclAllGather");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {},
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {},
        c10d::OpType::ALLGATHER);
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ReduceScatterOptions& opts)
{
    check_npu_tensors_different_devices(outputTensors);

    auto inputFlattened = flatten_for_scatter_gather(inputTensors, outputTensors, size_);
    check_npu_tensors_different_devices(inputFlattened);
    std::string functionName = __FUNCTION__;
    return collective(
        inputFlattened,
        outputTensors,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            auto hcclType = getHcclDataType(input.scalar_type());
            checkSupportedDataType(hcclType, functionName);
            RECORD_FUNCTION("HcclReduceScatter", std::vector<c10::IValue>({input}));
            if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() != c10_npu::option::AVOID_RECORD_STREAM) {
                c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
            }
            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(output);
            auto hcclReduceOp = getHcclReduceOp(opts.reduceOp, input);
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream, is_dispatched]() -> int {
                auto hccl_result = HcclReduceScatter(
                    inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclReduceScatter");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {
            work->lazyDestory(inputFlattened);
            // Copy the input tensors to the flattened inputs.
            for (const auto i : c10::irange(inputTensors.size())) {
                c10_npu::NPUStreamGuard guard(hcclStreams[i]);
                for (const auto j : c10::irange(inputTensors[0].size())) {
                    // See [Sync Streams].
                    if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::AVOID_RECORD_STREAM) {
                        work->stashed_for_allocator_safety_.push_back(inputTensors[i][j]);
                    } else {
                        c10_npu::NPUCachingAllocator::recordStream(inputTensors[i][j].storage().data_ptr(), hcclStreams[i]);
                        if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::ERASE_RECORD_STREAM) {
                            work->recorded_inputs_.push_back(
                                std::make_pair(inputTensors[i][j].storage().getWeakStorageImpl(), hcclStreams[i]));
                        }
                    }
                    inputFlattened[i][j].copy_(inputTensors[i][j], true);
                }
            }
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {},
        c10d::OpType::REDUCE_SCATTER);
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const c10d::ReduceScatterOptions& opts)
{
    if (inputTensor.dtype() != outputTensor.dtype()) {
        TORCH_CHECK(false, "input tensor must be the same type as the output tensor.", DIST_ERROR(ErrCode::TYPE));
    }

    if (inputTensor.numel() != outputTensor.numel() * size_) {
        TORCH_CHECK(false, "input tensor must be the same size as output size times world size", DIST_ERROR(ErrCode::PARAM));
    }

    auto inputs = std::vector<at::Tensor>{inputTensor};
    auto outputs = std::vector<at::Tensor>{outputTensor};
    std::string functionName = __FUNCTION__;
    return collective(
        inputs,
        outputs,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() != c10_npu::option::AVOID_RECORD_STREAM) {
                c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
            }
            auto hcclType = getHcclDataType(input.scalar_type());
            checkSupportedDataType(hcclType, functionName);
            RECORD_FUNCTION("HcclReduceScatterBase", std::vector<c10::IValue>({input}));

            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(output);
            auto hcclReduceOp = hcclOp[opts.reduceOp];
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream, is_dispatched]() -> int {
                auto hccl_result = HcclReduceScatter(
                    inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclReduceScatter");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {},
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {},
        c10d::OpType::REDUCE_SCATTER);
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::barrier(const c10d::BarrierOptions& opts)
{
    std::vector<at::Device> devices;
    if (usedDeviceIdxs_.empty()) {
        auto numNPUs = c10_npu::device_count();
        int16_t deviceIdx = static_cast<int16_t>(rank_ % std::max(static_cast<int>(numNPUs), 1));
        devices.push_back(at::Device(at_npu::key::NativeDeviceType));
    } else {
        for (auto usedDeviceIdx : usedDeviceIdxs_) {
            devices.push_back(at::Device(at_npu::key::NativeDeviceType, usedDeviceIdx));
        }
    }

    std::vector<at::Tensor> barrierTensors;
    barrierTensors.reserve(devices.size());

    c10_npu::OptionalNPUGuard npuGuard;
    for (auto& device : devices) {
        npuGuard.set_index(device.index());
        barrierTensors.push_back(
            at::ones({1}, at::TensorOptions().device(at_npu::key::NativeDeviceType).dtype(at::kFloat)));
    }

    auto work = allreduce(barrierTensors);

    // Work will take over barrierTensors
    auto hcclWork = dynamic_cast<ProcessGroupHCCL::WorkHCCL*>(work.get());
    TORCH_CHECK(hcclWork, DIST_ERROR(ErrCode::HCCL));
    hcclWork->barrierTensors_ = std::move(barrierTensors);

    return work;
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const c10d::GatherOptions& /* unused */)
{
    throw std::runtime_error("ProcessGroupHCCL does not support gather" + DIST_ERROR(ErrCode::NOT_SUPPORT));
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ScatterOptions& opts)
{
    static auto invalidArgument = [](const std::string& msg) {
        C10_THROW_ERROR(ValueError, "ProcessGroupHCCL::scatter: " + msg);
    };

    c10d::assertRootRank(invalidArgument, opts.rootRank, size_);
    check_npu_tensors_different_devices(outputTensors);
    c10d::assertSingleElementInput(invalidArgument, outputTensors);

    if (getRank() == opts.rootRank) {
        if (inputTensors.size() != 1) {
            std::stringstream ss;
            ss << "requires a single-element input list containing a list with "
                << getSize() << " tensors.";
            invalidArgument(ss.str());
        } else if (inputTensors[0].size() != static_cast<size_t>(getSize())) {
            std::stringstream ss;
            ss << "Incorrect input list size " << inputTensors[0].size()
                << ". Input list size should be " << getSize()
                << ", same as size of the process group.";
            invalidArgument(ss.str());
        }

        const auto& options = outputTensors[0].options();
        const auto& sizes = outputTensors[0].sizes();
        c10d::assertTypeAndSizesMatch(invalidArgument, inputTensors[0], options, sizes);
    } else {
        // if not in the root rank, initialize inputTensors as empty place holder
        // with an empty list
        if (inputTensors.size() != 0) {
            invalidArgument("requires empty input on non-root");
        }
    }

    std::vector<at::Tensor> inputFlattened;
    if (getRank() == opts.rootRank) {
        inputFlattened = flatten_for_scatter_gather(inputTensors, outputTensors, size_);
    } else {
        std::vector<at::Tensor> empty;
        for (int i = 0; i < size_; i++) {
            empty.push_back(at::empty_like(outputTensors[0]));
        }
        inputTensors.push_back(empty);
        inputFlattened = flatten_for_scatter_gather(inputTensors, outputTensors, size_);
    }

    return collective(
        inputFlattened,
        outputTensors,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            RECORD_FUNCTION("HcclScatter", std::vector<c10::IValue>({input}));
            const auto root = opts.rootRank;
            if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() != c10_npu::option::AVOID_RECORD_STREAM) {
                c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
            }
            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(output);
            auto hcclType = getHcclDataType(input.scalar_type());
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, root, comm, stream, is_dispatched]() -> int {
                auto hccl_result = hcclScatter(inputDataPtr, outputDataPtr, numel, hcclType, root, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclScatter");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {
            work->lazyDestory(inputFlattened);
            // Copy the input tensors to the flattened inputs.
            for (const auto i : c10::irange(inputTensors.size())) {
                c10_npu::NPUStreamGuard guard(hcclStreams[i]);
                for (const auto j : c10::irange(inputTensors[0].size())) {
                    // See [Sync Streams].
                    if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::AVOID_RECORD_STREAM) {
                        work->stashed_for_allocator_safety_.push_back(inputTensors[i][j]);
                    } else {
                        c10_npu::NPUCachingAllocator::recordStream(inputTensors[i][j].storage().data_ptr(), hcclStreams[i]);
                        if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::ERASE_RECORD_STREAM) {
                            work->recorded_inputs_.push_back(
                                std::make_pair(inputTensors[i][j].storage().getWeakStorageImpl(), hcclStreams[i]));
                        }
                    }
                    inputFlattened[i][j].copy_(inputTensors[i][j], true);
                }
            }
        },
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        c10d::OpType::SCATTER);
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag)
{
    check_npu_tensors_different_devices(tensors);
    auto tensors_ = cast_to_origin_format(tensors);
    return collective(
        tensors_,
        tensors_,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            RECORD_FUNCTION("HcclSend", std::vector<c10::IValue>({input}));

            auto inputDataPtr = input.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto hcclType = getHcclDataType(input.scalar_type());
            auto hccl_call = [inputDataPtr, numel, hcclType, dstRank, comm, stream, is_dispatched]() -> int {
                auto hccl_result = HcclSend(inputDataPtr, numel, hcclType, dstRank, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclSend");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

            return HCCL_SUCCESS;
        },
        c10d::OpType::SEND);
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag)
{
    check_npu_tensors_different_devices(tensors);
    auto tensors_ = create_base_format_tensors(tensors);

    return collective(
        tensors,
        tensors_,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            RECORD_FUNCTION("HcclRecv", std::vector<c10::IValue>({input}));
            if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() != c10_npu::option::AVOID_RECORD_STREAM) {
                c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
            }
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(output);
            auto hcclType = getHcclDataType(output.scalar_type());
            auto hccl_call = [outputDataPtr, numel, hcclType, srcRank, comm, stream, is_dispatched]() -> int {
                auto hccl_result = HcclRecv(outputDataPtr, numel, hcclType, srcRank, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclRecv");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {},
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {
            for (size_t i = 0; i < tensors_.size(); ++i) {
                c10_npu::NPUStreamGuard guard(hcclStreams[i]);
                if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::AVOID_RECORD_STREAM) {
                    work->stashed_for_allocator_safety_.push_back(tensors_[i]);
                } else {
                    c10_npu::NPUCachingAllocator::recordStream(tensors_[i].storage().data_ptr(), hcclStreams[i]);
                    if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::ERASE_RECORD_STREAM) {
                        work->recorded_outputs_.push_back(
                            std::make_pair(tensors_[i].storage().getWeakStorageImpl(), hcclStreams[i]));
                    }
                }
                if (!at_npu::native::FormatHelper::IsBaseFormatType(tensors[i])) {
                    tensors[i].copy_(tensors_[i], true);
                }
            }
        },
        c10d::OpType::RECV);
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */)
{
    TORCH_CHECK(false, "ProcessGroupHCCL does not support recv", DIST_ERROR(ErrCode::NOT_SUPPORT));
}

void check_split_sizes(const std::vector<int64_t>& split_sizes, const at::Tensor& tensor, int group_size)
{
    if (split_sizes.empty()) {
        TORCH_CHECK(tensor.size(0) % group_size == 0, "Tensor's dim 0 does not divide equally across group size",
                    DIST_ERROR(ErrCode::VALUE));
    } else {
        TORCH_CHECK(
            split_sizes.size() == static_cast<size_t>(group_size), "Number of tensor splits not equal to group size",
            DIST_ERROR(ErrCode::VALUE));
    }
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const c10d::AllToAllOptions& opts)
{
    check_npu_single_tensor(outputTensor);
    check_npu_single_tensor(inputTensor);
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    int ranks = getSize();
    TORCH_CHECK(ranks > 0, "Invalid ranks", ranks, DIST_ERROR(ErrCode::VALUE));
    if (inputSplitSizes.empty() && outputSplitSizes.empty()) {
        // We can use alltoall
        TORCH_CHECK(
            outputTensor.numel() == inputTensor.numel() && outputTensor.type() == inputTensor.type(),
            "Tensors are not equal in size or data type", DIST_ERROR(ErrCode::PARAM));
        TORCH_CHECK(outputTensor.size(0) % ranks == 0, "Tensor's dim 0 does not divide equally across group size", DIST_ERROR(ErrCode::PARAM));
        uint64_t output_counts = static_cast<uint64_t>(outputTensor.numel() / ranks);
        uint64_t input_counts = static_cast<uint64_t>(inputTensor.numel() / ranks);

        check_npu_tensors_different_devices(inputTensors);
        check_npu_tensors_different_devices(outputTensors);
        return collective(
            inputTensors,
            outputTensors,
            [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
                    RECORD_FUNCTION("HcclAlltoAll", std::vector<c10::IValue>({input}));
                auto inputDataPtr = input.data_ptr();
                auto outputDataPtr = output.data_ptr();
                auto inputhcclDataType = getHcclDataType(input.scalar_type());
                auto outputhcclDataType = getHcclDataType(output.scalar_type());
                auto hccl_call = [inputDataPtr,
                                  input_counts,
                                  inputhcclDataType,
                                  outputDataPtr,
                                  output_counts,
                                  outputhcclDataType,
                                  comm,
                                  stream,
                                  is_dispatched]() -> int {
                    auto hccl_result = hcclAlltoAll(
                        inputDataPtr,
                        input_counts,
                        inputhcclDataType,
                        outputDataPtr,
                        output_counts,
                        outputhcclDataType,
                        comm,
                        stream.stream(false));
                    *is_dispatched = true;
                    return hccl_result;
                };
                at_npu::native::OpCommand cmd;
                cmd.Name("HcclAlltoAll");
                cmd.SetCustomHandler(hccl_call);
                cmd.Run();

                return HCCL_SUCCESS;
            },
            c10d::OpType::ALLTOALL_BASE);
    } else {
        check_split_sizes(inputSplitSizes, inputTensor, size_);
        check_split_sizes(outputSplitSizes, outputTensor, size_);
        uint64_t index = static_cast<uint64_t>(outputTensor.numel() / ranks);
        if (outputSplitSizes.empty()) {
            for (int i = 0; i < ranks; i++) {
                outputSplitSizes.push_back(index);
            }
        }
        index = static_cast<uint64_t>(inputTensor.numel() / ranks);
        if (inputSplitSizes.empty()) {
            for (int i = 0; i < ranks; i++) {
                inputSplitSizes.push_back(index);
            }
        }
        int inputSize = static_cast<int>(inputSplitSizes.size());
        int outSize = static_cast<int>(outputSplitSizes.size());
        std::vector<uint64_t> inputCounts;
        std::vector<uint64_t> inputSpl;
        std::vector<uint64_t> outputCounts;
        std::vector<uint64_t> outputSpl;
        inputSpl.push_back(0);
        outputSpl.push_back(0);
        for (int i = 0; i < outSize; i++) {
            outputCounts.push_back(static_cast<uint64_t>(outputSplitSizes[i]));
            if (i > 0) {
                outputSpl.push_back(outputSpl[i - 1] + outputCounts[i - 1]);
            }
        }
        for (int i = 0; i < inputSize; i++) {
            inputCounts.push_back(static_cast<uint64_t>(inputSplitSizes[i]));
            if (i > 0) {
                inputSpl.push_back(inputSpl[i - 1] + inputCounts[i - 1]);
            }
        }

        check_npu_tensors_different_devices(inputTensors);
        check_npu_tensors_different_devices(outputTensors);
        return collective(
            inputTensors,
            outputTensors,
            [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
                RECORD_FUNCTION("HcclAlltoAllV", std::vector<c10::IValue>({input}));
                auto inputDataPtr = input.data_ptr();
                auto outputDataPtr = output.data_ptr();
                auto inputhcclDataType = getHcclDataType(input.scalar_type());
                auto outputhcclDataType = getHcclDataType(output.scalar_type());
                auto hccl_call = [inputDataPtr,
                                  inputCounts,
                                  inputSpl,
                                  inputhcclDataType,
                                  outputDataPtr,
                                  outputCounts,
                                  outputSpl,
                                  outputhcclDataType,
                                  comm,
                                  stream,
                                  is_dispatched]() -> int {
                    auto hccl_result = hcclAlltoAllV(
                        inputDataPtr,
                        inputCounts.data(),
                        inputSpl.data(),
                        inputhcclDataType,
                        outputDataPtr,
                        outputCounts.data(),
                        outputSpl.data(),
                        outputhcclDataType,
                        comm,
                        stream.stream(false));
                    *is_dispatched = true;
                    return hccl_result;
                };
                at_npu::native::OpCommand cmd;
                cmd.Name("HcclAlltoAllV");
                cmd.SetCustomHandler(hccl_call);
                cmd.Run();

                return HCCL_SUCCESS;
            },
            c10d::OpType::ALLTOALL);
    }
}

c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroupHCCL::alltoall(
    std::vector<at::Tensor>& output_tensors,
    std::vector<at::Tensor>& input_tensors,
    const c10d::AllToAllOptions& opts)
{
    auto device = output_tensors[0].device();
    for (const auto r : c10::irange(output_tensors.size())) {
        check_npu_single_tensor(output_tensors[r]);
        check_npu_single_tensor(input_tensors[r]);
        TORCH_CHECK(
            device == output_tensors[r].device() && device == input_tensors[r].device(),
            "tensors must be on the same device", DIST_ERROR(ErrCode::PARAM));
    }
    std::vector<int64_t> output_split_sizes;
    std::vector<int64_t> input_split_sizes;
    std::vector<at::Tensor> output_results;
    std::vector<at::Tensor> input_tensors_after;
    std::vector<at::Tensor> output_tensors_after;

    for (size_t i = 0; i < input_tensors.size(); i++) {
        int64_t inputlist_tensor_size = input_tensors[i].numel();
        input_split_sizes.push_back(inputlist_tensor_size);
        input_tensors_after.push_back(at::reshape(input_tensors[i], {inputlist_tensor_size,  1}));
    }

    for (size_t i = 0; i < output_tensors.size(); i++) {
        output_split_sizes.push_back(output_tensors[i].numel());
        output_tensors_after.push_back(at::reshape(output_tensors[i], {output_tensors[i].numel(),  1}));
    }

    int ranks = getSize();

    int inputsize = static_cast<int>(input_split_sizes.size());
    int outsize = static_cast<int>(output_split_sizes.size());
    std::vector<uint64_t> input_counts;
    std::vector<uint64_t> input_spl;
    std::vector<uint64_t> output_counts;
    std::vector<uint64_t> output_spl;
    input_spl.push_back(0);
    output_spl.push_back(0);
    output_counts.push_back(static_cast<uint64_t>(output_split_sizes[0]));
    input_counts.push_back(static_cast<uint64_t>(input_split_sizes[0]));
    for (int i = 1; i < outsize; i++) {
        output_counts.push_back(static_cast<uint64_t>(output_split_sizes[i]));
        output_spl.push_back(output_spl[i - 1] + static_cast<uint64_t>(output_split_sizes[i - 1]));
    }
    for (int i = 1; i < inputsize; i++) {
        input_counts.push_back(static_cast<uint64_t>(input_split_sizes[i]));
        input_spl.push_back(input_spl[i - 1] + static_cast<uint64_t>(input_split_sizes[i - 1]));
    }

    std::vector<at::Tensor> in_tensors = {at::cat(input_tensors_after, 0)};
    std::vector<at::Tensor> out_tensors = {at::cat(output_tensors_after, 0)};

    auto input_tensors_ = cast_to_origin_format(in_tensors);
    auto output_tensors_ = cast_to_origin_format(out_tensors);

    check_npu_tensors_different_devices(in_tensors);
    check_npu_tensors_different_devices(out_tensors);

    return collective(
        input_tensors_,
        output_tensors_,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            RECORD_FUNCTION("HcclAlltoAllV", std::vector<c10::IValue>({input}));
            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto inputhcclDataType = getHcclDataType(input.scalar_type());
            auto outputhcclDataType = getHcclDataType(output.scalar_type());
            auto hccl_call = [inputDataPtr,
                              input_counts,
                              input_spl,
                              inputhcclDataType,
                              outputDataPtr,
                              output_counts,
                              output_spl,
                              outputhcclDataType,
                              comm,
                              stream,
                              is_dispatched]() -> int {
                auto hccl_result = hcclAlltoAllV(
                    inputDataPtr,
                    input_counts.data(),
                    input_spl.data(),
                    inputhcclDataType,
                    outputDataPtr,
                    output_counts.data(),
                    output_spl.data(),
                    outputhcclDataType,
                    comm,
                    stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclAlltoAllV");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {
            c10_npu::NPUStreamGuard guard(hcclStreams[0]);
            if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::AVOID_RECORD_STREAM) {
                work->stashed_for_allocator_safety_.push_back(output_tensors_[0]);
            } else {
                c10_npu::NPUCachingAllocator::recordStream(output_tensors_[0].storage().data_ptr(), hcclStreams[0]);
				if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::ERASE_RECORD_STREAM) {
                    work->recorded_outputs_.push_back(
                        std::make_pair(output_tensors_[0].storage().getWeakStorageImpl(), hcclStreams[0]));
                }
            }
            if (!at_npu::native::FormatHelper::IsBaseFormatType(out_tensors[0])) {
                out_tensors[0].copy_(output_tensors_[0], true);
            }
            output_results = at::split_with_sizes(out_tensors[0], output_split_sizes, 0);
            for (int i = 0; i < output_results.size(); i++) {
                at::Tensor output_results_shaped = at::reshape(output_results[i], output_tensors[i].sizes());
                output_tensors[i].copy_(output_results_shaped, true);
            }
        },
        c10d::OpType::ALLTOALL);
}

} // namespace c10d_npu