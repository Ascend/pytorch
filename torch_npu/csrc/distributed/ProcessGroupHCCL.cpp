#include <ATen/record_function.h>
#include <algorithm>
#include <map>
#include <tuple>
#include <unordered_set>
#include <unistd.h>
#include <linux/limits.h>
#include <fstream>
#include <iostream>
#include <functional>
#include <cstdlib>

#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <c10d/ParamCommsUtils.hpp>
#include <c10d/TraceUtils.h>

#include "op_plugin/OpInterface.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUAffinityController.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/interface/OpInterface.h"
#include "torch_npu/csrc/distributed/HCCLUtils.hpp"
#include "torch_npu/csrc/distributed/HcclCompile.h"
#include "torch_npu/csrc/distributed/ProcessGroupHCCL.hpp"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
#include "torch_npu/csrc/framework/OpHook.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/profiler/npu_profiler.h"
#include <c10d/Utils.hpp>

namespace c10d_npu {
namespace {
static constexpr uint32_t kOpWaitTimeoutOffset = 30U; // second
static uint32_t kOpWaitTimeout = 1868U; // second
constexpr const char* P2P_DEVICE_KEY = "_p2p";

using hcclUs = std::chrono::steady_clock::time_point;
#define DURATION_US(x) (std::chrono::duration_cast<std::chrono::microseconds>(x))
#define TIME_NOW() ({ std::chrono::steady_clock::now(); })

#define MAX_GROUP_NAME_LEN 128

// HCCL ReduceOp mapping
std::map<c10d::ReduceOp, HcclReduceOp> hcclOp = {
    {c10d::ReduceOp::MIN, HCCL_REDUCE_MIN},
    {c10d::ReduceOp::MAX, HCCL_REDUCE_MAX},
    {c10d::ReduceOp::SUM, HCCL_REDUCE_SUM},
    {c10d::ReduceOp::PRODUCT, HCCL_REDUCE_PROD},
};

std::map<c10d::ReduceOp, std::string> unsupportedOp = {
    {c10d::ReduceOp::AVG, "AVG"},
    {c10d::ReduceOp::BAND, "BAND"},
    {c10d::ReduceOp::BOR, "BOR"},
    {c10d::ReduceOp::BXOR, "BXOR"}
};

bool nslb_is_end = false;
bool uce_error_flag = false;
bool force_stop_error_flag = false;
char* nslb_path = c10_npu::option::OptionsManager::GetNslbPath();
bool status_save_enable = c10_npu::option::OptionsManager::CheckStatusSaveEnable();
std::string status_save_path = c10_npu::option::OptionsManager::GetStatusSavePath();

inline c10_npu::NPUStream getNPUStreamByCurrentType(c10::DeviceIndex device = -1)
{
    auto current_Stream = c10_npu::getCurrentNPUStream(device);
    if (!current_Stream.isSyncLaunchStream()) {
        return c10_npu::getNPUStreamFromPool(device);
    }
    return c10_npu::getNPUStreamFromSyncLaunchPool(device);
}

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
            TORCH_NPU_WARN_ONCE(
                "The storage data_ptr is different from tensor data_ptr."
                "Maybe this tensor is not suitable for HCCL.");
        }
        return physical_numel(self);
    } else {
        return self.numel();
    }
}

HcclReduceOp getHcclReduceOp(const c10d::ReduceOp reduceOp, at::Tensor& input)
{
    if (reduceOp == c10d::ReduceOp::SUM && input.scalar_type() == at::kBool) {
        // For bool tensors, map sum to max, which both represent a bitwise or.
        // This is to prevent overflow issues with sum, since we use uint8 to
        // represent a bool (see hcclDataType mapping).
        return HCCL_REDUCE_MAX;
    }
    
    if (unsupportedOp.find(reduceOp) != unsupportedOp.end()) {
        TORCH_CHECK(false,
            "Cannot use ReduceOp." + unsupportedOp[reduceOp] + " with HCCL",
            DIST_ERROR(ErrCode::NOT_SUPPORT));
    } else if (hcclOp.find(reduceOp) == hcclOp.end()) {
        TORCH_CHECK(false, "Unhandled ReduceOp", DIST_ERROR(ErrCode::NOT_FOUND));
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

std::string getKeyFromDevice(const std::vector<at::Device>& devices)
{
    return std::to_string(devices[0].index());
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

std::vector<at::Device> getDevice(const std::vector<at::Tensor>& tensors)
{
    std::vector<at::Device> res;
    res.reserve(1);
    res.push_back(tensors[0].device());
    return res;
}

// Return device with ordinal given by input rank.
at::Device getDeviceForRank(int rank)
{
    TORCH_CHECK(rank >= 0, "Invalid rank ", rank, DIST_ERROR(ErrCode::VALUE));
    auto numNPUs = c10_npu::device_count();
    TORCH_CHECK(numNPUs > 0, "Invalid device number", numNPUs, DIST_ERROR(ErrCode::VALUE));
    int16_t deviceIdx = static_cast<int16_t>(rank % numNPUs);
    return at::Device(c10::DeviceType::PrivateUse1, deviceIdx);
}

std::string getKeySendRecv(int myRank, int peer)
{
    int lowRank = myRank < peer ? myRank : peer;
    int highRank = myRank < peer ? peer : myRank;
    std::string sendRecvPair = std::to_string(lowRank) + ":" + std::to_string(highRank);
    return sendRecvPair;
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

void getHcclCommConfig(HcclCommConfig* config)
{
    HcclCommConfigInit(config);
    config->hcclBufferSize = c10_npu::option::OptionsManager::GetHcclBufferSize();
}

void getP2PHcclCommCofig(HcclCommConfig* config)
{
    HcclCommConfigInit(config);
    config->hcclBufferSize = c10_npu::option::OptionsManager::GetP2PBufferSize();
    // Compatible with the size check of the old version of HCCL, forcibly convert
    // the config object to a size_t=32 object, and retain the N ± 2 version
    if (!isHcclFeatureSupported(HcclCommConfigCapability::HCCL_COMM_CONFIG_COMM_NAME)) {
        size_t *configSize = reinterpret_cast<size_t *>(config);
        *configSize = 32;
    }
}

void checkHcclCommConfigValid(const HcclCommConfig* config)
{
    if (strlen(config->hcclCommName) > 0) {
        TORCH_CHECK(isHcclFeatureSupported(HcclCommConfigCapability::HCCL_COMM_CONFIG_COMM_NAME),
                    "The current version of CANN does not support the hcclCommName:", config->hcclCommName,
                    DIST_ERROR(ErrCode::NOT_SUPPORT));
    }
}

void fill_equal_split_sizes_when_empty(std::vector<int64_t>& split_sizes, at::Tensor tensor, int group_size)
{
    if (!split_sizes.empty()) {
        return;
    }
    TORCH_CHECK(group_size > 0, "Invalid group size within current process group", group_size, DIST_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        tensor.size(0) % group_size == 0,
        "Tensor's dim 0 does not divide equally across group size",
        DIST_ERROR(ErrCode::PARAM));
    int64_t equal_split_size = static_cast<int64_t>(tensor.size(0) / group_size);
    for (int i = 0; i < group_size; i++) {
        split_sizes.push_back(equal_split_size);
    }
}

void check_split_sizes(const std::vector<int64_t>& split_sizes, const at::Tensor& tensor, int group_size)
{
    if (split_sizes.empty()) {
        TORCH_CHECK(tensor.size(0) % group_size == 0, "Tensor's dim 0 does not divide equally across group size",
                    DIST_ERROR(ErrCode::PARAM));
    } else {
        TORCH_CHECK(
            split_sizes.size() == static_cast<size_t>(group_size), "Number of tensor splits not equal to group size",
            DIST_ERROR(ErrCode::TYPE));
        const auto sum = c10::sum_integers(split_sizes);
        TORCH_CHECK(sum == tensor.size(0), "Split sizes dosen't match total dim 0 size", DIST_ERROR(ErrCode::TYPE));
    }
}

void checkAndMakePath(const char* path, std::string errormessage)
{
    try {
        if (access(path, W_OK) != 0 && mkdir(path, S_IRWXU | S_IRGRP | S_IXGRP) != 0) {
            throw std::exception();
        }
    } catch (std::exception& e) {
        throw std::runtime_error(errormessage + DIST_ERROR(ErrCode::NOT_FOUND));
    }
}

void createFile(const char* path)
{
    int fd = open(path, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP);
    if (fd == -1) {
        throw std::runtime_error("Create file failed. Please check whether input file is valid." + DIST_ERROR(ErrCode::NOT_FOUND));
    }
    close(fd);
}
} // namespace

constexpr int64_t kSynchronizeBusyWaitMillis = 10;
constexpr int64_t maxOpNumPerSyncPoint = 2;
const int64_t ProcessGroupHCCL::kProcessGroupHCCLOpTimeoutMillis = 10 * 1000;
thread_local uint64_t ProcessGroupHCCL::hcclActiveGroupCounter_ = 0;
const int64_t ProcessGroupHCCL::kWatchdogThreadSleepMillis = 1000;
std::string ProcessGroupHCCL::perfdumppath = "";
ProcessGroupHCCL* ProcessGroupHCCL::global_ = nullptr;
std::unordered_map<std::string, ProcessGroupHCCL::StatusStruct> ProcessGroupHCCL::StatusOutput_;
int ProcessGroupHCCL::deviceId_ = -1;
int ProcessGroupHCCL::numRanks_ = -1;
std::string ProcessGroupHCCL::exceptionMessage_ = "";

std::ostream& operator<<(std::ostream& output, const ProcessGroupHCCL::WorkHCCL& workHCCL)
{
    std::string workInfo = c10::str(
        "WorkHCCL(",
        "SeqNum=",
        workHCCL.seq_,
        ", OpType=",
        opTypeToString(workHCCL.opType_),
        ", Timeout(ms)=",
        workHCCL.opTimeout_.count(),
        ")");
    return output << workInfo;
}

ProcessGroupHCCL::WorkHCCL::WorkHCCL(
    const std::vector<at::Device>& devices,
    int rank,
    c10d::OpType opType,
    uint64_t seq,
    bool desyncDebug)
    : Work(rank, opType),
    devices_(devices),
    workStartTime_(std::chrono::steady_clock::now()),
    seq_(seq)
{
    // Creates the npu event wrappers
    // Note: The actual events are lazily created when first recorded to with
    // DEFAULT_FLAGS = npuEventDisableTiming.
    if (desyncDebug || (status_save_enable)) {
        hcclStartEvents_ = std::make_shared<std::vector<c10_npu::NPUEvent>>();
        hcclStartEvents_->reserve(devices.size());
        for (int i = 0; i < devices.size(); i++) {
            hcclStartEvents_->emplace_back(ACL_EVENT_CAPTURE_STREAM_PROGRESS);
        }
    }

    hcclEndEvents_ = std::make_shared<std::vector<c10_npu::NPUEvent>>(devices.size());
    hcclComms_.resize(devices.size());
}

ProcessGroupHCCL::WorkHCCL::WorkHCCL(const WorkHCCL& w)
    : Work(w.rank_, w.opType_),
    std::enable_shared_from_this<WorkHCCL>(w),
    devices_(w.devices_),
    hcclStartEvents_(w.hcclStartEvents_),
    hcclComms_(w.hcclComms_),
    hcclEndEvents_(w.hcclEndEvents_),
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

bool ProcessGroupHCCL::WorkHCCL::isStarted() {
    checkAndSetException();
    return exception() || startedNPUExecutionInternal();
}

bool ProcessGroupHCCL::WorkHCCL::isSuccess() const
{
    if (exception()) {
        // Already detected an exception.
        return false;
    }
    return !checkForHCCLErrors(hcclComms_) && finishedNPUExecutionInternal();
}

void ProcessGroupHCCL::WorkHCCL::checkAndSetException()
{
    if (exception()) {
        // We already have an exception.
        return;
    }
    auto exception_ptr = checkForHCCLErrors(hcclComms_);
    std::unique_lock<std::mutex> lock(mutex_);
    exception_ = exception_ptr;
    if (exception_) {
        ASCEND_LOGE("[Rank %d], found async exception when checking for HCCL errors: %s", rank_,
            getExceptionMsgFromExceptionPtr(exception_).c_str());
        LOG(INFO) << "[Rank " << rank_ << "]"
              << " found async exception when checking for HCCL errors: "
              << getExceptionMsgFromExceptionPtr(exception_);
    }
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

bool ProcessGroupHCCL::WorkHCCL::startedNPUExecutionInternal() const {
    try {
        for (const auto i : c10::irange(devices_.size())) {
            // Checking the work's corresponding ASCEND events' status
            if (!(*hcclStartEvents_)[i].query()) {
                return false;
            }
        }
    } catch (const std::exception& e) {
        if (std::string(e.what()).find("UCE ERROR") == std::string::npos) {
            uce_error_flag = true;
            return false;
        }

        if (std::string(e.what()).find("FORCE STOP") == std::string::npos) {
            force_stop_error_flag = true;
            return false;
        }

        if (std::string(e.what()).find("driver shutting down") == std::string::npos) {
            throw std::runtime_error(DIST_ERROR(ErrCode::INTERNAL));
        }
        LOG(INFO) << "[Rank " << rank_ << "] Event query failed with exception: " << e.what();
    }

    return true;
}

// check if HCCL task is finished
bool ProcessGroupHCCL::WorkHCCL::finishedNPUExecutionInternal() const
{
    // If in the Finalize, should not query event
    if (!c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
        return false;
    }
    try {
        for (const auto i : c10::irange(devices_.size())) {
            // Checking the work's corresponding ASCEND events' status
            if (!(*hcclEndEvents_)[i].query()) {
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

bool ProcessGroupHCCL::WorkHCCL::checkTimeout(c10::optional<std::chrono::milliseconds> timeout)
{
    auto currentTimepoint = std::chrono::steady_clock::now();
    auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTimepoint - workStartTime_);
    auto workTimeout = timeout ? *timeout : opTimeout_;

    if (timeElapsed < workTimeout)
        return false;

    // Timed out

    // There is already an error, we don't override it
    if (exception())
        return true;

    std::string exceptionMsg = c10::str(
        "[Rank ",
        rank_,
        "] ",
        "Watchdog caught collective operation timeout: ",
        *this,
        " ran for ",
        timeElapsed.count(),
        " milliseconds before timing out.");

    LOG(ERROR) << exceptionMsg;
    std::exception_ptr exception_ptr =
        std::make_exception_ptr(std::runtime_error(exceptionMsg));
    setException(exception_ptr);
    return true;
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

void ProcessGroupHCCL::WorkHCCL::synchronize() {
    // Call Synchronize without a timeout. We use this method to avoid adding a
    // timeout argument to the public synchronize API.
    synchronizeInternal(kNoTimeout);
}

void ProcessGroupHCCL::WorkHCCL::handleException(ErrorHandlingMode errorHandling)
{
    if (exception_) {
        auto exceptionMsg = c10::str(
            "Some HCCL operations have failed or timed out. Due to the ",
            "asynchronous nature of ASCEND kernels, subsequent NPU operations ",
            "might run on corrupted/incomplete data.");
        LOG(ERROR) << exceptionMsg;
        C10_LOG_API_USAGE_ONCE("ProcessGroupHCCL.WorkHCCL.handleException");

        if (SHOULD_TEAR_DOWN(errorHandling)) {
            auto tearDownMsg = c10::str(
                "To avoid data inconsistency, we are taking the entire process down.");
            LOG(ERROR) << tearDownMsg;
            std::rethrow_exception(exception_);
        }
    }
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
void ProcessGroupHCCL::WorkHCCL::synchronizeInternal(std::chrono::milliseconds timeout)
{
    for (const auto i : c10::irange(devices_.size())) {
        auto currentStream = c10_npu::getCurrentNPUStream(devices_[i].index());
        // Block the current stream on the HCCL stream
        (*hcclEndEvents_)[i].block(currentStream);
        ASCEND_LOGI("Event: block hccl work is successfully executed, event=%p", (*hcclEndEvents_)[i].event());
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
        lazy_destroy_tensors_.clear();
    } else if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::AVOID_RECORD_STREAM) {
        stashed_for_allocator_safety_.clear();
    }

    // In case of blocking, wait for the operation to complete.
    if (blockingWait_) {
        // Wait for the operation to complete.
        while (!isCompleted()) {
            bool timedOut = checkTimeout(
                timeout == kNoTimeout ? c10::nullopt : c10::make_optional(timeout));
            // Explicitly abort hcclComms here before throwing this timed out
            // exception to users.
            // If throwing timed out excepiton without aborting hccl communicators
            // here, ASCEND NPU may not run new events successfully.
            if (timedOut) {
                std::string exceptionMsg = c10::str(
                    "[Rank ",
                    rank_,
                    "] Work ",
                    (*this),
                    " timed out in blocking wait (HCCL_BLOCKING_WAIT=1).");
                LOG(ERROR) << exceptionMsg;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
        }
        if (exception()) {
            // Abort HCCL communicators
            abort();
            // Throw exception (from main thread here)
            handleException(TearDown);
        }
    }

    if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {
        at_npu::native::OpHook::GetInstance().PostHook();
    }
}

void ProcessGroupHCCL::WorkHCCL::lazyDestroy(std::vector<at::Tensor> tensors)
{
    if (tensors.empty() ||
        (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() != c10_npu::option::ERASE_RECORD_STREAM)) {
        return;
    }

    for (const auto i : c10::irange(tensors.size())) {
        lazy_destroy_tensors_.push_back(tensors[i]);
    }
}

// Same as calling synchronize().
bool ProcessGroupHCCL::WorkHCCL::wait(std::chrono::milliseconds timeout)
{
    synchronizeInternal(timeout);
    // Always return true, because abort API is not implemented.
    return true;
}

void ProcessGroupHCCL::WorkHCCL::abort()
{
    // Abort all communicators of this work
    for (const auto& hcclComm : hcclComms_) {
        hcclComm->destroyHcclComm();
    }
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupHCCL::WorkHCCL::getFuture()
{
    return future_;
}

std::vector<at::Tensor> ProcessGroupHCCL::WorkHCCL::result()
{
    return *outputs_;
}

ProcessGroupHCCL::ProcessGroupHCCL(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : c10d::Backend(rank, size),
    store_(store),
    options_(c10::make_intrusive<Options>(*options.get())),
    hcclCommCounter_(0),
    traceKeyStart_("HCCL_" + std::to_string(rank) + "_trace_start"),
    traceKeyEnd_("HCCL_" + std::to_string(rank) + "_trace_end"),
    terminateProcessGroup_(false)
{
    uint32_t hccl_event_timeout = c10_npu::option::OptionsManager::GetHCCLEventTimeout();
    uint32_t hccl_exec_timeout = c10_npu::option::OptionsManager::GetHCCLExecTimeout();
    if (hccl_event_timeout > 0) {
        kOpWaitTimeout = hccl_event_timeout;
        ASCEND_LOGI("Set op wait timeout to %u.", hccl_event_timeout);
    } else {
        // When no env, the default value is 0
        if (hccl_exec_timeout > 0) {
            kOpWaitTimeout = hccl_exec_timeout + kOpWaitTimeoutOffset;
            if (kOpWaitTimeout <= hccl_exec_timeout) {
                kOpWaitTimeout = UINT_MAX;
            }
        }
        ASCEND_LOGI(
            "Get env HCCL_EXEC_TIMEOUT value %u, and set op wait timeout to %u.", hccl_exec_timeout, kOpWaitTimeout);
    }
    NPU_CHECK_SUPPORTED_OR_ERROR(c10_npu::acl::AclrtSetOpWaitTimeout(kOpWaitTimeout));
    char* blockingWait = getenv(HCCL_BLOCKING_WAIT);
    try {
        if (blockingWait != nullptr) {
            auto val = std::stoi(blockingWait);
            if (val == 1) {
                // Make wait() and synchronize() a blocking call.
                blockingWait_ = true;
            } else if (val != 0) {
                throw std::runtime_error("Invalid value for environment variable: " + std::string(HCCL_BLOCKING_WAIT)
                + DIST_ERROR(ErrCode::VALUE));
            }
        }
    } catch (std::exception& e) {
        throw std::runtime_error("Invalid value for environment variable: " + std::string(HCCL_BLOCKING_WAIT)
        + DIST_ERROR(ErrCode::VALUE));
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
        LOG(INFO) << "[Rank " << rank_
                    << "] HCCL_DESYNC_DEBUG and HCCL_ASYNC_ERROR_HANDLING "
                    << "must both be enabled. "
                    << "Enabling HCCL_ASYNC_ERROR_HANDLING.";
        asyncErrorHandling_ = TearDown;
        }
    }

#ifdef ENABLE_HCCL_ERROR_CHECKING
    if (asyncErrorHandling_ == TearDown) {
        if (hccl_exec_timeout > 0) {
            if ((hccl_exec_timeout * 1000) > (options_->timeout).count()) {
                TORCH_NPU_WARN("The HCCL execution timeout ", hccl_exec_timeout * 1000, "ms is bigger than watchdog timeout ",
                    (options_->timeout).count(), "ms which is set by init_process_group! The plog may not be recorded.");
            }
        } else {
            if ((options_->timeout).count() == DEFAULT_TIMEOUT) {
                // Only when the timeout is default, we will change it.
                options_->timeout = std::chrono::milliseconds(DEFAULT_TIMEOUT * 2);
            }
            if ((options_->timeout).count() < DEFAULT_TIMEOUT) {
                TORCH_NPU_WARN("The HCCL execution timeout 1800000ms is bigger than watchdog timeout ",
                    (options_->timeout).count(), "ms which is set by init_process_group! The plog may not be recorded.");
            }
        }
    }
    hcclCommWatchdogThread_ = std::thread(&ProcessGroupHCCL::hcclCommWatchdog, this);
#endif

    if (options_->global_ranks_in_group.empty()) {
        global_ = this;
    }
    ASCEND_LOGI("process group created, group id is %s.", options_->group_id.c_str());
}

void ProcessGroupHCCL::setSequenceNumberForGroup() {}

uint64_t ProcessGroupHCCL::getSequenceNumberForGroup()
{
    return seq_;
}

void abortCommsFromMap(
    std::unordered_map<std::string, std::vector<std::shared_ptr<HCCLComm>>>& hcclCommsMap,
    const int rank,
    c10::optional<std::string> abortReason)
{
    // The process may control multiple devices, loop through the communicators on
    // each device
    for (auto& it : hcclCommsMap) {
        auto& devName = it.first;
        auto& hcclComms = it.second;

        for (const auto& hcclComm : hcclComms) {
            hcclComm->destroyHcclComm();
        }
        // Note that we don't remove the aborted communicators from the
        // cache. The reason is that if we do remove the communicator
        // from the cache, it is possible that a new collective operation
        // calls `hcclCommInitRank` to create a new communicator whereas
        // other ranks might have failed/timed out and didn't enter
        // `hcclCommInitRank`. As a result, when there is a failure on
        // a communicator the application receives an exception and its
        // their responsibility to destroy the process group and recreate
        // it to recover from errors.

        if (abortReason.has_value()) {
            LOG(INFO) << "[Rank " << rank << "] Destroyed " << hcclComms.size()
                << "communicators on ASCEND device " << devName
                << " for reason: " << *abortReason;
        } else {
            LOG(INFO) << "[Rank " << rank << "] Destroyed " << hcclComms.size()
                << "communicators on ASCEND device " << devName;
        }
    }
}

// Abort all communicators on this rank
void ProcessGroupHCCL::abort(c10::optional<std::string> abortReason)
{
    std::lock_guard<std::mutex> lock(mutex_);
    abortCommsFromMap(devHCCLCommMap_, rank_, abortReason);
}

void ProcessGroupHCCL::deleteTCPStoreKey()
{
    try {
        // all processes in a group may be killed, so delete key 0 as a last resort
        store_->deleteKey("0");
        for (const auto &key : TCPStoreKeyList_) {
            store_->deleteKey(key);
        }
    } catch(...) {
        // different ranks may delete at the same time, which could cause exception
        TCPStoreKeyList_.clear();
        return;
    }
    
    TCPStoreKeyList_.clear();
}

void ProcessGroupHCCL::abortAndClearHcclComm(c10::optional<std::string> abortReason)
{
    std::lock_guard<std::mutex> lock(mutex_);
    abortCommsFromMap(devHCCLCommMap_, rank_, abortReason);
    devHCCLCommMap_.clear();
    devHCCLCommNameMap_.clear();
    hcclCommCounter_ = 0;
    return;
}

ProcessGroupHCCL::~ProcessGroupHCCL()
{
    if (options_->global_ranks_in_group.empty()) {
        global_ = nullptr;
    }

    terminateProcessGroup_.store(true);

    workMetaListCV_.notify_one();
#ifdef ENABLE_HCCL_ERROR_CHECKING
    hcclCommWatchdogThread_.join();
#endif
    {
        // Destropy all HCCL Communicators on Process Group Destruction
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& it : devHCCLCommMap_) {
            auto& hcclComms = it.second;

            for (const auto& hcclComm : hcclComms) {
                hcclComm->destroyHcclComm();
            }
        }
        devHCCLCommMap_.clear();
    }
    ASCEND_LOGI("process group destroyed, group id is %s.", options_->group_id.c_str());
}

void ProcessGroupHCCL::hcclCommWatchdog()
{
    try {
        c10_npu::SetThreadName(c10_npu::ThreadType::hcclCommWatchdogThread);

        VLOG(2) << "[Rank " << rank_ << "] HCCL watchdog thread started!";
        workCleanupLoop();
        VLOG(2) << "[Rank " << rank_
                << "] HCCL watchdog thread terminated normally";
    } catch (std::exception& e) {
        // Append error message reported from workCleanupLoop
        const auto exitMsg = c10::str(
            "[Rank ",
            rank_,
            "] HCCL watchdog thread terminated with exception: ",
            e.what());
        LOG(ERROR) << exitMsg;
        if (status_save_enable) {
            if (exceptionMessage_.empty()) {
                exceptionMessage_ = e.what();
            }
            recordHcclStatus(status_save_path, true, true);
        }
        watchDogException_ = std::make_exception_ptr(std::runtime_error(exitMsg));
        std::rethrow_exception(watchDogException_);
    } catch (...) {
        const auto exitMsg = c10::str(
            "[Rank ",
            rank_,
            "] HCCL watchdog thread terminated with exception: unknown");
        LOG(ERROR) << exitMsg;
        if (status_save_enable) {
            recordHcclStatus(status_save_path, true);
        }
        watchDogException_ = std::make_exception_ptr(std::runtime_error(exitMsg));
        std::rethrow_exception(watchDogException_);
    }
}

void ProcessGroupHCCL::logWorkStart(WorkHCCL& work)
{
    if (work.startTraceUpdated_) {
        return;
    }

    if (terminateProcessGroup_.load() || storeError_) {
        return;
    }

    work.startTraceUpdated_ = true;
    storeError_ = !c10d::traceUpdate(store_, traceKeyStart_, work.seq_, opTypeToString(work.opType_));
}

void ProcessGroupHCCL::logWorkEnd(WorkHCCL& work)
{
    if (terminateProcessGroup_.load() || storeError_) {
        return;
    }

    // In case the start of the work hasn't been logged
    if (!work.startTraceUpdated_) {
        logWorkStart(work);
    }

    storeError_ = !c10d::traceUpdate(store_, traceKeyEnd_, work.seq_, opTypeToString(work.opType_));
}

const std::vector<uint32_t>& ProcessGroupHCCL::groupRanks() const
{
    if (options_->global_ranks_in_group.empty()) {
        static std::vector<uint32_t> globalRanks(size_);
        std::iota(globalRanks.begin(), globalRanks.end(), 0);
        return globalRanks;
    }
    return options_->global_ranks_in_group;
}

void ProcessGroupHCCL::workCleanupLoop()
{
    bool needSetDevice = true;
    std::list<ProcessGroupHCCL::WorkHCCL> completedWorkList;
    auto lastrecordtime = std::chrono::steady_clock::now();
    auto timenow = std::chrono::steady_clock::now();
    bool recordflag = false;
    
    while (!terminateProcessGroup_.load()) {
        if (status_save_enable) {
            checkAndMakePath(status_save_path.c_str(), "Open shared directory failed. Please check whether input path is valid.");
            timenow = std::chrono::steady_clock::now();
            recordflag = (std::chrono::duration_cast<std::chrono::milliseconds>(timenow - lastrecordtime).count() > (c10_npu::option::OptionsManager::GetStatusSaveInterval() * 1000));
        }

        {
        std::unique_lock<std::mutex> lock(workMetaListMutex_);
        // We busy-poll the work vector every kWatchdogThreadSleepMillis
        // milliseconds as long as the atomic is True.
        workMetaListCV_.wait_for(lock, std::chrono::milliseconds(kWatchdogThreadSleepMillis),
                                 [&]() -> bool { return terminateProcessGroup_.load(); });
        if (watchdogStatus == WatchdogStatus::STOP) {
            continue;
        }

        for (auto it = workMetaList_.begin(); it != workMetaList_.end();
             /* no increment */) {
            auto& work = *it;
            try {
                if (needSetDevice) {
                    c10::DeviceIndex device = static_cast<int>(work.devices_[0].index());
                    c10_npu::SetThreadAffinity(device);
                    NPU_CHECK_ERROR(c10_npu::SetDevice(device));
                    deviceId_ = static_cast<int>(work.devices_[0].index());
                    needSetDevice = false;
                }
            } catch (const std::exception& e) {
                if (std::string(e.what()).find("UCE ERROR") == std::string::npos) {
                    uce_error_flag = true;
                }

                if (std::string(e.what()).find("FORCE STOP") == std::string::npos) {
                    force_stop_error_flag = true;
                }
            }
            work.checkAndSetException();
            work.checkDispatch();
            bool timedOut = work.checkTimeout();

            // If work hits an exception (either an error or timeout)
            if (work.exception()) {
                // Report desync state in case of timeout
                if (desyncDebug_ && timedOut) {
                    try {
                        auto desyncMsg = retrieveDesyncReport(store_, "HCCL", rank_, size_);
                        LOG(ERROR) << desyncMsg;
                    } catch (const std::exception& e) {
                        LOG(ERROR) << "Failed to retrieve HCCL_DESYNC_DEBUG report. "
                                   << " Please file an issue. Error: " << e.what();
                    } catch (...) {
                        LOG(ERROR) << "Failed to rerieve HCCL_DESYNC_DEBUG report with unknown error."
                                   << " Please file an issue.";
                    }
                }
                // Throw exception
                work.handleException(asyncErrorHandling_);
            }

            // Work status logging for desync debug
            if (desyncDebug_) {
                if (work.isStarted()) {
                    logWorkStart(work);
                }
                if (work.isCompleted()) {
                    logWorkEnd(work);
                }
            }

            // Clean up completed work
            if (work.isCompleted()) {
                if (*(work.is_dispatched) && work.is_reported) {
                    ASCEND_LOGE("Process group work %s, seq_num %u dispatch sucess. This error log can be ignored.", opTypeToString(work.opType_).c_str(), work.seq_);
                    work.is_reported = false;
                }
                if (status_save_enable) {
                    refreshStatusInfo(work, "end"); // Update Statusinfo，but not write into the map
                }
                it = workMetaList_.erase(it);
            } else {
                if (status_save_enable && work.isStarted()) {
                    refreshStatusInfo(work, "start"); // Update Statusinfo，but not write into the map
                }
                // Increment the iterator if the current WorkHCCL object is not
                // completed.
                ++it;
            }
        }
        }

        if (recordflag && recordHcclStatus(status_save_path)) {
            lastrecordtime = std::chrono::steady_clock::now();
        }
    }

    if (status_save_enable) {
        recordHcclStatus(status_save_path);
    }

    if (terminateProcessGroup_.load()) {
        if (status_save_enable) {
            recordHcclStatus(status_save_path, true);
        }
        std::unique_lock<std::mutex> lock(workMetaListMutex_);
        workMetaList_.clear();
    }
}

std::exception_ptr ProcessGroupHCCL::WorkHCCL::checkForHCCLErrors(
    const std::vector<std::shared_ptr<HCCLComm>>& hcclComms) const {
    return checkForHCCLErrorsInternal(hcclComms);
}

std::exception_ptr ProcessGroupHCCL::checkForHCCLErrors(
    const std::vector<std::shared_ptr<HCCLComm>>& hcclComms) {
    return checkForHCCLErrorsInternal(hcclComms);
}

std::exception_ptr ProcessGroupHCCL::checkForHCCLErrorsInternal(
    const std::vector<std::shared_ptr<HCCLComm>>& hcclComms) {
    for (const auto& hcclComm : hcclComms) {
        HcclResult hcclAsyncErr = hcclComm->checkForHcclError();
        if (hcclAsyncErr != HCCL_SUCCESS) {
            return std::make_exception_ptr(std::runtime_error(
                "HCCL error: " + getHcclErrorDetailStr(hcclAsyncErr)));
        }
    }
    return nullptr;
}

void ProcessGroupHCCL::broadcastMasterID(
    HcclRootInfo* hcclID,
    bool isSingleP2POp,
    const std::string& devicesKey,
    int p2pRank)
{
    // For every HCCL communicator that we create we need to broadcast
    // a unique ID from rank 0 to all other ranks. This broadcast is
    // done by rank 0 setting a key in the store and all other ranks
    // retrieving the contents of that key. A single process group
    // may create multiple HCCL communicators, so we use a sequence
    // number to differentiate between them.
    std::string storeKey;
    if (!isSingleP2POp) {
        storeKey = std::to_string(hcclCommCounter_++);
    } else {
        storeKey = devicesKey;
    }
    std::string ver_key = "version_key";
    auto date_list = __DATE__ != nullptr ? __DATE__ : "";
    std::vector<uint8_t> ver_list;
#ifdef PYTORCH_NPU_VERSION
    auto py_list = PYTORCH_NPU_VERSION != nullptr ? PYTORCH_NPU_VERSION : "";
    ver_list.insert(ver_list.end(), py_list, py_list + strlen(py_list));
#endif
    ver_list.insert(ver_list.end(), date_list, date_list + strlen(date_list));
    if (rank_ == 0 || (isSingleP2POp && p2pRank == 0)) {
        auto vec = std::vector<uint8_t>(
            reinterpret_cast<uint8_t*>(hcclID), reinterpret_cast<uint8_t*>(hcclID) + HCCL_ROOT_INFO_BYTES);
        store_->set(storeKey, vec);
        store_->set(ver_key, ver_list);
        TCPStoreKeyList_.emplace(storeKey);
    } else {
        try {
            auto vec = store_->get(storeKey);
            TORCH_CHECK(vec.size() == HCCL_ROOT_INFO_BYTES, DIST_ERROR(ErrCode::PARAM));
            TCPStoreKeyList_.emplace(storeKey);
            std::memcpy(hcclID, vec.data(), vec.size());
        } catch (const std::exception& e) {
            std::string exceptionMsg = c10::str(
                "[",
                rank_,
                "] is setting up HCCL communicator and "
                "retrieving hcclUniqueId from [0] via c10d key-value store by key '",
                storeKey,
                "', but store->get('",
                storeKey,
                "') got error: ");
            throw std::runtime_error(exceptionMsg + e.what() +
                ". This may indicate a possible application crash on rank 0 or a network set up issue." +
                DIST_ERROR(ErrCode::INTERNAL));
        } catch (...) {
            throw std::runtime_error(c10::str(
                "Unknown exception while [",
                rank_,
                "] is setting up HCCL communicator and "
                "retrieving hcclUniqueId from [0] via c10d key-value store by key '",
                storeKey,
                "'",
                ". This may indicate a possible application crash on rank 0 or a network set up issue.") +
                DIST_ERROR(ErrCode::INTERNAL));
        }
        auto main_list = store_->get(ver_key);
        if (main_list != ver_list) {
            TORCH_NPU_WARN("PTA version mismatch");
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
    auto hccl_algo = getenv("HCCL_ALGO");
    TORCH_CHECK(master_addr != nullptr, "Unable to fetch master IP addr, environment variable is null.", DIST_ERROR(ErrCode::NOT_FOUND));
    fileName << master_addr << "_" << commName << "_" << std::to_string(currRank) << ".log";
    std::string out_file_path = c10::str(nslb_path, "/", fileName.str());
    bool need_algo = hccl_algo != nullptr && access(out_file_path.c_str(), W_OK) != 0;
    try {
        if (access(nslb_path, W_OK) != 0 && mkdir(nslb_path, S_IRWXU | S_IRGRP | S_IXGRP) != 0) {
            throw std::exception();
        }
        if (access(out_file_path.c_str(), W_OK) != 0) {
            int fd = open(out_file_path.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP);
            if (fd == -1) {
                throw std::exception();
            }
            close(fd);
        }
        outfile.open(out_file_path, std::ios::app);
    } catch (std::exception& e) {
        throw std::runtime_error("Open shared directory failed. Please check whether input path is valid." + DIST_ERROR(ErrCode::NOT_FOUND));
    }
    std::transform(opName.begin(), opName.end(), opName.begin(), ::tolower);
    if (need_algo) {
        outfile << "HCCL_ALGO=" << hccl_algo << "\n";
    }
    outfile << opName << " " << dataVol << " " << std::to_string(currRank) << "\n";
    outfile.close();
}

void ProcessGroupHCCL::refreshStatusInfo(ProcessGroupHCCL::WorkHCCL work, std::string status)
{
    StatusInfo.seq = work.seq_;
    StatusInfo.pgId = options_->group_id;
    StatusInfo.opType = opTypeToString(work.opType_);
    if (StatusInfo.commIds == "") {
        for (auto i : options_->global_ranks_in_group) {
            StatusInfo.commIds += (std::to_string(i) + " ");
        }
    }
    if (StatusInfo.commIds == "") {
        StatusInfo.commIds = "all";
    }
    StatusInfo.status = status;
}

void ProcessGroupHCCL::updateStatusOutput()
{
    if (!StatusInfo.pgId.empty()) {
        StatusOutput_[options_->group_id] = StatusInfo;
    }
}

bool ProcessGroupHCCL::recordHcclStatus(const std::string path, bool end, bool error)
{
    std::unique_lock<std::mutex> lock(StatusMapmutex_);
    updateStatusOutput();
    if (!options_->global_ranks_in_group.empty() && !error) {
        return true;
    } else if (!StatusOutput_.empty()) {
        static auto pid = getpid();
        static std::chrono::time_point<std::chrono::system_clock> firstrecordtime = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(firstrecordtime.time_since_epoch()).count();
        auto end_duration = duration;
        if (end) {
            static std::chrono::time_point<std::chrono::system_clock> endrecordtime = std::chrono::system_clock::now();
            end_duration = std::chrono::duration_cast<std::chrono::milliseconds>(endrecordtime.time_since_epoch()).count();
        }
        std::ofstream outfile;
        std::stringstream fileName;
        static auto master_addr = getenv("MASTER_ADDR");
        if (master_addr == nullptr) {
            master_addr = "127.0.0.1";
            ASCEND_LOGW("Unable to fetch master IP addr, environment variable is null, it will use 127.0.0.1");
        }
        int global_rank = rank_;
        if (!options_->global_ranks_in_group.empty()) {
            global_rank = options_->global_ranks_in_group[rank_];
        }
        fileName << "torch_hccl_status-" << std::to_string(global_rank) << "_" << master_addr << "_" << std::to_string(deviceId_) << "_";
        fileName << std::to_string(numRanks_) << "_" << std::to_string(pid) << "_" << std::to_string(duration) << ".log";
        std::string isMaster = "false";
        if (global_rank == 0) {
            isMaster = "true";
        }
        std::string out_file_path = c10::str(path, "/", fileName.str());
        checkAndMakePath(path.c_str(), "Open shared directory failed. Please check whether input path is valid.");
        createFile(out_file_path.c_str());
        outfile.open(out_file_path.c_str(), std::ios::trunc);
        outfile << "{\"last_comm_op\":[";
        bool first_op = true;
        for (auto info = StatusOutput_.begin(); info != StatusOutput_.end(); info++) {
            if (first_op) {
                outfile << "{";
            } else {
                outfile << ", {";
            }
            outfile << "\"seq\":" << info->second.seq << ", \"op_type\":\"" << info->second.opType;
            outfile << "\", \"pg_id\":" << info->second.pgId << ", \"comm_ids\":\"" << info->second.commIds;
            outfile << "\", \"status\":\""<< info->second.status << "\"}";
            first_op = false;
        }
        outfile << "], \"is_master\":" << isMaster;
        outfile << ", \"exception_message\":\"" << exceptionMessage_;
        outfile << "\", \"global_pg_end_time\":" << end_duration << "}" << std::endl;
        outfile.close();
        return true;
    }
    return false;
}

void ProcessGroupHCCL::recordComm(std::string filename, std::string opName, const int currRank, std::vector<std::shared_ptr<HCCLComm>>& hcclComms)
{
    std::ofstream outfile;
    std::string commName = getHcclCommNameWithoutInit(currRank, hcclComms);
    if (isFileExists(filename)) {
        try {
            outfile.open(filename, std::ios::app);
        } catch (std::exception& e) {
            throw std::runtime_error("Open shared directory failed. Please check whether file is valid." + DIST_ERROR(ErrCode::UNAVAIL));
        }
    } else {
        TORCH_CHECK(false, filename, " is not exist!", DIST_ERROR(ErrCode::NOT_FOUND));
    }

    std::transform(opName.begin(), opName.end(), opName.begin(), ::tolower);
    const std::vector<uint32_t>& ranks = groupRanks();
    std::stringstream ss;
    for (size_t i = 0; i < ranks.size(); ++i) {
        ss << ranks[i];
        if (i != ranks.size() - 1) {
            ss << ", ";
        }
    }

    std::string group_ranks = ss.str();
    CommStruct comm_struct {commName, opName};
    if (commset.find(comm_struct) == commset.end()) {
        outfile << "[COMM]:" << commName << "," << opName << "," << group_ranks << "\n";
        outfile.close();
        commset.insert(comm_struct);
    }
}

std::vector<std::shared_ptr<HCCLComm>>& ProcessGroupHCCL::getHCCLComm(
    const std::string& devicesKey,
    const std::vector<at::Device>& devices,
    HcclCommType commType,
    HcclCommConfig* commConfig,
    int p2pRank)
{
    // Sanity check
    if (devicesKey.empty()) {
        throw std::runtime_error(
            "Not able to create/get the HCCL Communicator since "
            "the NPU devices are not known" + DIST_ERROR(ErrCode::PARAM));
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
    return createHCCLComm(devicesKey, devices, commType, commConfig, p2pRank);
}

void ProcessGroupHCCL::createHCCLComm(
    const std::string& devicesKey,
    const std::vector<at::Device>& devices,
    HcclCommType commType,
    HcclCommConfig* commConfig,
    std::vector<std::shared_ptr<HCCLComm>> &hcclComms,
    std::vector<c10_npu::NPUStream> &streamVal,
    int p2pRank)
{
    HcclRootInfo hcclID;
    bool isSingleP2POp = commType == HcclCommType::P2P ? true : false;
    if (rank_ == 0 || (isSingleP2POp && p2pRank == 0)) {
        HCCL_CHECK_ERROR(HcclGetRootInfo(&hcclID));
    }
    broadcastMasterID(&hcclID, isSingleP2POp, devicesKey, p2pRank);

    c10_npu::OptionalNPUGuard npuGuard;
    for (size_t i = 0; i < devices.size(); ++i) {
        int numRanks = getSize();
        int rank = getRank() * static_cast<int>(devices.size()) + static_cast<int>(i);

        HcclCommConfig config;

        npuGuard.set_index(devices[i].index());
        switch (commType) {
            case HcclCommType::DEFAULT:
                if (commConfig != nullptr) {
                    checkHcclCommConfigValid(commConfig);
                    hcclComms[i] = HCCLComm::create_config(numRanks, rank, hcclID, commConfig);
                } else {
                    if (!options_->hccl_config.empty()) {
                        config = createHcclCommConfigWithOptions();
                        hcclComms[i] = HCCLComm::create_config(numRanks, rank, hcclID, &config);
                    } else {
                        hcclComms[i] = HCCLComm::create(numRanks, rank, hcclID);
                    }
                }
                break;
            case HcclCommType::P2P: // P2P not support set hcclCommName
                numRanks = 2;
                rank = p2pRank;
                getP2PHcclCommCofig(&config);
                hcclComms[i] = HCCLComm::create_config(numRanks, rank, hcclID, &config);
                break;
            default:
                throw std::runtime_error(
                    "create/get the HCCL Communicator failed for comm type:" +
                    std::to_string((int)commType) + DIST_ERROR(ErrCode::PARAM));
        }

        // Creates the HCCL streams
        streamVal.push_back(getNPUStreamByCurrentType(devices[i].index()));
    }
}

bool ProcessGroupHCCL::createHCCLCommEx(
    const std::vector<at::Device>& devices,
    HcclCommType commType,
    HcclCommConfig* commConfig,
    std::vector<std::shared_ptr<HCCLComm>> &hcclComms,
    std::vector<c10_npu::NPUStream> &streamVal,
    int p2pRank)
{
    std::string rankTableFile = c10_npu::option::OptionsManager::GetRankTableFilePath();
    if (rankTableFile.empty() || !checkFilePathReadable(rankTableFile)) {
        ASCEND_LOGI("The rank_table_file is not available, switch to original interface.");
        return false;
    }
    if (c10_npu::option::OptionsManager::GetHCCLConnectTimeout() < 300) {
        TORCH_NPU_WARN_ONCE("When creating an HCCL process group using the RANK_TABLE_FILE method, the connection may time out. ",
            "It is recommended to set the timeout duration of HCCL_CONNECT_TIMEOUT to 300 seconds or more.");
    }
    if (!hcclCommInitClusterInfoConfigExist()) {
        ASCEND_LOGI("The hcclCommInitClusterInfoConfig is not exist, switch to original interface.");
        return false;
    }
    c10_npu::OptionalNPUGuard npuGuard;
    // global process group
    if (options_->global_ranks_in_group.empty() && commType == HcclCommType::DEFAULT) {
        auto startTime = std::chrono::steady_clock::now();
        for (size_t i = 0; i < devices.size(); ++i) {
            int rank = getRank() * static_cast<int>(devices.size()) + static_cast<int>(i);

            npuGuard.set_index(devices[i].index());
            HcclCommConfig config;
            if (commConfig == nullptr) {
                config = createHcclCommConfigWithOptions();
                commConfig = &config;
            }
            auto comm = HCCLComm::createGlobalHcclComm(rankTableFile.c_str(), rank, commConfig);
            if (comm == nullptr) {
                ASCEND_LOGI("Create global hccl comm with ranktable failed.");
                return false;
            }
            hcclComms[i] = comm;
            // Creates the HCCL streams
            streamVal.push_back(getNPUStreamByCurrentType(devices[i].index()));
        }
        auto endTime = std::chrono::steady_clock::now();
        auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        ASCEND_LOGI("Create global hccl comm with ranktable success, take %d milliseconds", timeElapsed.count());
        return true;
    }

    // sub process group
    if (!hcclCreateSubCommConfigExist()) {
        ASCEND_LOGI("The hcclCreateSubCommConfig is not exist, switch to original interface.");
        return false;
    }
    if (global_ == nullptr) {
        ASCEND_LOGI("The global process group is not exist, switch to original interface.");
        return false;
    }
    std::shared_ptr<HCCLComm> globalHcclComm = nullptr;
    try {
        globalHcclComm = global_->getHcclCommByDevices(devices);
    } catch (const std::exception& e) {
        ASCEND_LOGI("create the global HCCL Communicator failed, the exception info is %s.", e.what());
        return false;
    }
    if (!globalHcclComm) {
        ASCEND_LOGI("Create sub hccl comm by hcclCreateSubCommConfig failed, globalHcclComm is nullptr.");
        return false;
    }

    uint64_t hcclid = (std::hash<string>{}(options_->group_id));
    auto subStartTime = std::chrono::steady_clock::now();
    for (size_t i = 0; i < devices.size(); ++i) {
        int numRanks = getSize();
        int rank = getRank() * static_cast<int>(devices.size()) + static_cast<int>(i);

        npuGuard.set_index(devices[i].index());
        HcclCommConfig config;
        if (commConfig == nullptr) {
            config = createHcclCommConfigWithOptions();
            if (commType == HcclCommType::P2P) {
                numRanks = 2;
                rank = p2pRank;
                config.hcclBufferSize = c10_npu::option::OptionsManager::GetP2PBufferSize();
            }
            commConfig = &config;
        }
        auto subComm = HCCLComm::createSubHcclComm(globalHcclComm, numRanks, options_->global_ranks_in_group.data(), hcclid, rank, commConfig);
        if (subComm == nullptr) {
            ASCEND_LOGI("Create sub hccl comm by hcclCreateSubCommConfig failed, group id is %s, subCommId is %llu.",
                options_->group_id.c_str(), hcclid);
            return false;
        }
        hcclComms[i] = subComm;
        // Creates the HCCL streams
        streamVal.push_back(getNPUStreamByCurrentType(devices[i].index()));
    }
    auto subEndTime = std::chrono::steady_clock::now();
    auto subTimeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(subEndTime - subStartTime);
    ASCEND_LOGI("Create sub hccl comm by hcclCreateSubCommConfig success, group id is %s, subCommId is %llu, use %d ms.",
        options_->group_id.c_str(), hcclid, subTimeElapsed.count());
    return true;
}

std::vector<std::shared_ptr<HCCLComm>>& ProcessGroupHCCL::createHCCLComm(
    const std::string& devicesKey,
    const std::vector<at::Device>& devices,
    HcclCommType commType,
    HcclCommConfig* commConfig,
    int p2pRank)
{
    // HCCL communicator not cached, create a new entry
    std::vector<std::shared_ptr<HCCLComm>> hcclComms;
    hcclComms.resize(devices.size());
    std::vector<c10_npu::NPUStream> streamVal;
    streamVal.reserve(devices.size());

    if (!createHCCLCommEx(devices, commType, commConfig, hcclComms, streamVal, p2pRank)) {
        createHCCLComm(devicesKey, devices, commType, commConfig, hcclComms, streamVal, p2pRank);
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

    // Move the HCCL resource to cache
    devHCCLCommMap_.emplace(devicesKey, std::move(hcclComms));
    return devHCCLCommMap_[devicesKey];
}

int64_t ProcessGroupHCCL::getStreamId(bool p2p, int peer)
{
    int device;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
    std::vector<at::Device> devices = {at::Device(c10::DeviceType::PrivateUse1, device)};
    auto key = getKeyFromDevices(devices);
    if (p2p && hcclCommInitRootInfoConfigExist() && c10_npu::option::OptionsManager::GetP2PBufferSize() != 0) {
        TORCH_CHECK(peer >= 0, "In p2p scenarios, the passed 'dst rank id' is error.", DIST_ERROR(ErrCode::PARAM));
        key = getKeySendRecv(rank_, peer);
    }
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
        TORCH_CHECK(false, "Tensor list mustn't be larger than the number of available NPUs", DIST_ERROR(ErrCode::VALUE));
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

// Check that all `tensors' have the same type and shape and reside on the same NPU.
void check_npu_tensors_same_device(const std::vector<at::Tensor>& tensors)
{
    if (tensors.size() == 0) {
        TORCH_CHECK(false, "Tensor list must be nonempty", DIST_ERROR(ErrCode::PARAM));
    }

    const auto& first = tensors.front();

    for (const auto& t : tensors) {
        if (!torch_npu::utils::is_npu(t) || t.is_sparse()) {
            TORCH_CHECK(false, "Tensors must be NPU and dense", DIST_ERROR(ErrCode::TYPE));
        }

        TORCH_CHECK(
            t.scalar_type() == first.scalar_type(),
            "Tensors must have identical type",
            DIST_ERROR(ErrCode::TYPE));

        TORCH_CHECK(
            t.is_non_overlapping_and_dense(),
            "Tensors must be non-overlapping and dense",
            DIST_ERROR(ErrCode::TYPE));

        TORCH_CHECK(
            t.get_device() == first.get_device(),
            "Tensors must be on same NPU device",
            DIST_ERROR(ErrCode::TYPE));
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

bool check_same_size(const std::vector<at::Tensor>& input_tensors)
{
    for (const auto& input_tensor : input_tensors) {
        if (!input_tensors[0].is_same_size(input_tensor)) {
            return false;
        }
    }
    return true;
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
        TORCH_CHECK(false, "Tensor list operands to scatter/gather must have the same length", DIST_ERROR(ErrCode::VALUE));
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
        if (access(end_file_path.c_str(), W_OK) != 0) {
            int fd = open(end_file_path.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP);
            if (fd == -1) {
                throw std::exception();
            }
            close(fd);
        }
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
    return c10::make_intrusive<ProcessGroupHCCL::WorkHCCL>(devices, rank, opType, op_id_, desyncDebug_);
}

void ProcessGroupHCCL::workEnqueue(c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL> work)
{
    if (uce_error_flag) {
        uce_error_flag = false;
        ASCEND_LOGE("uce_error_flag is true when workEnqueue, throw UCE ERROR.");
        throw std::runtime_error("UCE ERROR." + PTA_ERROR(ErrCode::ACL));
        return;
    }
    if (force_stop_error_flag) {
        force_stop_error_flag = false;
        ASCEND_LOGE("force_stop_error_flag is true when workEnqueue, throw FORCE STOP.");
        throw std::runtime_error("FORCE STOP." + PTA_ERROR(ErrCode::ACL));
        return;
    }
    if (watchdogStatus == WatchdogStatus::STOP) {
        return;
    }
    if (!terminateProcessGroup_.load()) {
        std::lock_guard<std::mutex> lock(workMetaListMutex_);
        // Avoid view tensors to be processed in cleanup thread.
        // View tensors' destruction invokes autograd_meta, which
        // needs to be destructed in user thread. Otherwise will
        // get deadlock. Here we enqueue work without outputs_.
        workMetaList_.emplace_back(*work);
    }
}

ProcessGroupHCCL::Options::Options(bool is_high_priority_stream)
    : c10d::Backend::Options(HCCL_BACKEND_NAME),
      opTimeout(kProcessGroupHCCLOpTimeoutMillis),
      is_high_priority_stream(is_high_priority_stream)
{
}

std::shared_ptr<HCCLComm> ProcessGroupHCCL::getHcclCommByDevices(const std::vector<at::Device>& devices)
{
    const auto key = getKeyFromDevices(devices);
    auto& hcclComms = getHCCLComm(key, devices);
    TORCH_CHECK(hcclComms.size() == 1, "expect hcclComms.size() = 1, but hcclComms.size() = ",
        hcclComms.size(), DIST_ERROR(ErrCode::VALUE));
    return hcclComms[0];
}

int64_t ProcessGroupHCCL::getHcclComm(int rankid)
{
    at::Device device = getDeviceForRank(rankid);
    std::vector<at::Device> devices = {device};
    const auto key = getKeyFromDevices(devices);
    auto& hcclComms = getHCCLComm(key, devices);
    TORCH_CHECK(hcclComms.size() == 1, "expect hcclComms.size() = 1, but hcclComms.size() = ",
                hcclComms.size(), DIST_ERROR(ErrCode::VALUE));
    auto ret_hcom = hcclComms[0]->getHcclComm();
    int64_t hccl_comm = static_cast<int64_t>(reinterpret_cast<intptr_t>(ret_hcom));
    return hccl_comm;
}

void ProcessGroupHCCL::resumeHcclComm(int device_id)
{
    at::Device device = at::Device(c10::DeviceType::PrivateUse1, device_id);
    std::vector<at::Device> devices = {device};
    const auto key = getKeyFromDevices(devices);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (devHCCLCommMap_.find(key) != devHCCLCommMap_.end()) {
            // Reuse the cached communicator if there is one.
            auto& hcclComms = devHCCLCommMap_[key];
            for (const auto& hcclComm : hcclComms) {
                auto comm = hcclComm->getHcclComm();
                HCCL_CHECK_ERROR(at_npu::hccl::HcclCommResumeFace(comm));
            }
        }
    }
    ASCEND_LOGI("resumeHcclComm success, group id is %s.", options_->group_id.c_str());
}

void ProcessGroupHCCL::setWatchdogStatus(int status)
{
    watchdogStatus = WatchdogStatus(status);
}


void ProcessGroupHCCL::clearWorkMetaList()
{
    std::unique_lock<std::mutex> lock(workMetaListMutex_);
    workMetaList_.clear();
}


void ProcessGroupHCCL::setHcclCommName(const std::string& hccl_comm_name)
{
    auto nameSize = hccl_comm_name.size();
    TORCH_CHECK(nameSize > 0 && nameSize < COMM_NAME_MAX_LENGTH,
                "The length of the name must be between 1 and ", COMM_NAME_MAX_LENGTH - 1, ", Invalid hcclCommName:",
                hccl_comm_name, DIST_ERROR(ErrCode::VALUE));
    c10::DeviceIndex indexFromCurDevice = c10_npu::current_device();
    at::Device device = at::Device(c10::DeviceType::PrivateUse1, indexFromCurDevice);
    std::vector <at::Device> devices = {device};
    const auto key = getKeyFromDevices(devices);
    std::lock_guard <std::mutex> lock(mutex_);
    auto hcclCommNameIter = devHCCLCommNameMap_.emplace(key, hccl_comm_name);
    auto currentHcclCommName = hcclCommNameIter.first->second;
    TORCH_CHECK(currentHcclCommName == hccl_comm_name,
                "The current ProcessGroup has already set the name and cannot be duplicated, Invalid hcclCommName:",
                hccl_comm_name, ", current hcclCommName:", currentHcclCommName, DIST_ERROR(ErrCode::VALUE));
}

std::string ProcessGroupHCCL::getHcclCommName(int rankid, bool init_comm)
{
    TORCH_CHECK(rankid >= 0, "Invalid rank ", rankid, DIST_ERROR(ErrCode::VALUE));
    auto numNPUs = c10_npu::device_count();
    TORCH_CHECK(numNPUs > 0, "Invalid device number", numNPUs, DIST_ERROR(ErrCode::VALUE));
    c10::DeviceIndex indexFromRank = static_cast<c10::DeviceIndex>(rankid % numNPUs);
    c10::DeviceIndex indexFromCurDevice = c10_npu::current_device();
    if (indexFromRank != indexFromCurDevice) {
        std::string warning_message = "The indexFromRank " + std::to_string(indexFromRank) +
        "is not equal indexFromCurDevice " + std::to_string(indexFromCurDevice) +
        " , which might be normal if the number of devices on your collective communication server is inconsistent." +
        "Otherwise, you need to check if the current device is correct when calling the interface." +
        "If it's incorrect, it might have introduced an error.";
        TORCH_WARN_ONCE(warning_message);
    }
    at::Device device = at::Device(c10::DeviceType::PrivateUse1, indexFromCurDevice);
    std::vector<at::Device> devices = {device};
    const auto key = getKeyFromDevices(devices);
    if (!init_comm) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (devHCCLCommMap_.find(key) == devHCCLCommMap_.end()) {
            return "";
        }
    }
    std::string hcclCommName = "";
    std::vector <std::shared_ptr<HCCLComm>> hcclComms;
    {
        std::lock_guard <std::mutex> lock(mutex_);
        hcclCommName = devHCCLCommNameMap_[key];
    }
    if (!hcclCommName.empty()) {
        HcclCommConfig config = createHcclCommConfigWithOptions();
        torch_npu::toolkit::profiler::Utils::safe_strcpy_s(config.hcclCommName, hcclCommName.c_str(),
                                                           COMM_NAME_MAX_LENGTH);
        hcclComms = getHCCLComm(key, devices, HcclCommType::DEFAULT, &config);
    } else {
        hcclComms = getHCCLComm(key, devices);
    }
    TORCH_CHECK(hcclComms.size() == 1, "expect hcclComms.size() = 1, but hcclComms.size() = ",
        hcclComms.size(), DIST_ERROR(ErrCode::VALUE));
    HcclComm hcom = hcclComms[0]->getHcclComm();
    char commName[MAX_GROUP_NAME_LEN] = {};
    HCCL_CHECK_ERROR(at_npu::hccl::HcclGetCommNameFace(hcom, commName));
    return std::string(commName);
}

std::string ProcessGroupHCCL::getHcclCommNameWithoutInit(int rankid, std::vector<std::shared_ptr<HCCLComm>>& hcclComms)
{
    TORCH_CHECK(hcclComms.size() == 1, "expect hcclComms.size() = 1, but hcclComms.size() = ",
        hcclComms.size(), DIST_ERROR(ErrCode::VALUE));
    HcclComm ret_hcom = hcclComms[0]->getHcclComm();
    char commName[MAX_GROUP_NAME_LEN];
    HCCL_CHECK_ERROR(at_npu::hccl::HcclGetCommNameFace(ret_hcom, commName));
    std::string name_str(commName);
    return name_str;
}

std::string ProcessGroupHCCL::getMstxHcclMsg(
    const std::string &opName, uint64_t dataCnt, HcclDataType dataType, HcclComm comm)
{
    const static std::map<HcclDataType, std::string> dataTypes = {
        {HCCL_DATA_TYPE_INT8, "int8"},
        {HCCL_DATA_TYPE_INT16, "int16"},
        {HCCL_DATA_TYPE_INT32, "int32"},
        {HCCL_DATA_TYPE_FP16, "fp16"},
        {HCCL_DATA_TYPE_FP32, "fp32"},
        {HCCL_DATA_TYPE_INT64, "int64"},
        {HCCL_DATA_TYPE_UINT64, "uint64"},
        {HCCL_DATA_TYPE_UINT8, "uint8"},
        {HCCL_DATA_TYPE_UINT16, "uint16"},
        {HCCL_DATA_TYPE_UINT32, "uint32"},
        {HCCL_DATA_TYPE_FP64, "fp64"},
        {HCCL_DATA_TYPE_BFP16, "bfp16"}
    };
    static std::map<HcclComm, std::string> commNames;
    if (!torch_npu::profiler::mstxEnable()) {
        return "";
    }
    std::string hccl_message_str = "comm:" + opName + ",";
    auto nameIter = commNames.find(comm);
    if (nameIter == commNames.end()) {
        char commName[MAX_GROUP_NAME_LEN];
        HCCL_CHECK_ERROR(at_npu::hccl::HcclGetCommNameFace(comm, commName));
        std::string name(commName);
        commNames.insert({comm, name});
        hccl_message_str += name;
    } else {
        hccl_message_str += nameIter->second;
    }
    hccl_message_str += ",";
    std::string data_type_str = "na";
    auto iter = dataTypes.find(dataType);
    if (iter != dataTypes.end()) {
        data_type_str = iter->second;
    }
    hccl_message_str = hccl_message_str + data_type_str + "," + std::to_string(dataCnt);
    return hccl_message_str;
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
    if (c10_npu::opapi::IsExistAclnnSilentCheckV2()) {
        at::Tensor val = input.detach().pow(2).max().view(-1);
        at::Tensor max;
        if (silenceCheckCache_.find(opType) == silenceCheckCache_.end()) {
            at::Tensor stepTensor = at::zeros({1}, input.options().dtype(at::kLong));
            at::Tensor avg = input.detach().pow(2).max().view(-1);
            max = avg;
            silenceCheckCache_.emplace(opType, std::make_pair(std::move(stepTensor), std::move(avg)));
        } else {
            max = val;
        }
        static double beta1 = 0.99;
        op_plugin::_npu_silent_check_v3(val, input, silenceCheckCache_[opType].first, max, silenceCheckCache_[opType].second,
            c10_npu::option::OptionsManager::GetSilenceUpperThresh().first, c10_npu::option::OptionsManager::GetSilenceUpperThresh().second,
            beta1, static_cast<int64_t>(c10_npu::option::OptionsManager::GetSilenceCheckFlag()));
    } else {
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
}

HcclCommConfig ProcessGroupHCCL::createHcclCommConfigWithOptions()
{
    HcclCommConfig config;
    getHcclCommConfig(&config);

    if (options_->hccl_config.find("hccl_buffer_size") != options_->hccl_config.end()) {
        config.hcclBufferSize = std::get<uint32_t>(options_->hccl_config["hccl_buffer_size"]);
    }

    if (options_->hccl_config.find("group_name") != options_->hccl_config.end()) {
        auto groupName = std::get<std::string>(options_->hccl_config["group_name"]);
        uint32_t udiLength = groupName.length();
        if (groupName.length() >= UDI_MAX_LENGTH) {
            udiLength = UDI_MAX_LENGTH - 1;
            TORCH_NPU_WARN("The length of group_name has exceeded the limit UDI_MAX_LENGTH which will be truncated to UDI_MAX_LENGTH - 1.");
        }
        strncpy(config.hcclUdi, groupName.c_str(), udiLength);
        config.hcclUdi[udiLength] = '\0';
    }

    return config;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    c10d::OpType opType)
{
    // Bump collective counter
    seq_++;
    op_id_++;

    const auto devices = getDeviceList(inputs);
    auto key = getKeyFromDevices(devices);
    std::vector<std::shared_ptr<HCCLComm>> hcclComms;
    if (!options_->hccl_config.empty()) {
        HcclCommConfig config = createHcclCommConfigWithOptions();
        hcclComms = getHCCLComm(key, devices, HcclCommType::DEFAULT, &config);
    } else {
        hcclComms = getHCCLComm(key, devices);
    }

    // Used many times below, so we stash the unordered_map lookup
    auto& hcclStreams = hcclStreams_[key];
    // First let HCCL streams wait for input tensors allocation streams
    syncStreams(devices, hcclEvents_[key], hcclStreams);
    // Work itself will create the events on all NPUs of tensors
    auto work = initWork(devices, rank_, opType);
    // Store references to outputs to be used by WorkHCCL::result and operator<<.
    work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);
    c10_npu::OptionalNPUGuard npuGuard;

    if (desyncDebug_ || status_save_enable) {
        for (const auto i : c10::irange(devices.size())) {
            c10_npu::NPUStream& hcclStream = hcclStreams[i];
            (*(work->hcclStartEvents_))[i].record(hcclStream);
        }
    }

    // No need to detect batch_isend_irecv inputs is incorrect, need require special treatments.
    if (c10_npu::model_state().get_model_mode() == c10_npu::ModelMode::L_TRAIN
        && c10_npu::option::OptionsManager::GetSilenceCheckFlag() != c10_npu::option::CHECK_CLOSE
        && opType != c10d::OpType::UNKNOWN) {
        for (const auto i : c10::irange(inputs.size())) {
            npuGuard.set_index(devices[i].index());
            c10_npu::NPUStreamGuard guard(hcclStreams[i]);
            silenceCheck(inputs[i], opType);
        }
    }

    pre(hcclStreams, work);

    if (nslb_path != nullptr && !nslb_is_end) {
        auto nslb_num = c10_npu::option::OptionsManager::GetNslbCntVal();
        if (op_id_ <= nslb_num) {
            size_t dataVol = 0;
            for (auto tensor:inputs) {
                dataVol += tensor.storage().nbytes();
            }
            char* global_rank = getenv("RANK");
            TORCH_CHECK(global_rank != nullptr, "Unable to fetch global rank for NSLB.", DIST_ERROR(ErrCode::NOT_FOUND));
            recordDataVol(opTypeToString(opType), std::to_string(dataVol), atoi(global_rank), hcclComms);
        }
        if (op_id_ >= nslb_num) {
            nslb_is_end = true;
            nslb_record_end();
        }
    }

    static bool perf_dump_enable = c10_npu::option::OptionsManager::CheckPerfDumpEnable();
    if (perf_dump_enable) {
        if (perfdumppath.empty()) {
            auto pid = getpid();
            int device_id = c10_npu::current_device();
            std::ostringstream oss;
            oss << "perf_pt_" << pid << "_" << device_id << ".log";
            std::string log_file_name = oss.str();
            auto perfDumpPath = c10_npu::option::OptionsManager::GetPerfDumpPath();
            char abs_path[PATH_MAX] = {'\0'};
            if (realpath(perfDumpPath.c_str(), abs_path) == nullptr) {
                TORCH_CHECK(0, "perfDumpPath is not realpath.", DIST_ERROR(ErrCode::NOT_FOUND));
            }
            auto path_temp = c10::str(perfDumpPath, "/", log_file_name);
            if (isFileExists(path_temp)) {
                perfdumppath = path_temp;
                std::ofstream outfile;
                try {
                    outfile.open(perfdumppath, std::ios::app);
                } catch (std::exception& e) {
                    throw std::runtime_error("Open shared directory failed. Please check whether perfdumppath is valid." + DIST_ERROR(ErrCode::NOT_FOUND));
                }

                const std::vector<uint32_t>& ranks = groupRanks();
                outfile << "[GLOBAL RANKID]:" << ranks[rank_] << "\n";
                
                outfile.close();
            }
        } else {
            recordComm(perfdumppath, opTypeToString(opType), rank_, hcclComms);
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
    {
        c10_npu::NPUMultiStreamGuard guard(hcclStreams);
        work->future_ = c10::make_intrusive<at::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()),
            devices);
        work->future_->markCompleted(at::IValue(*work->outputs_));
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        c10_npu::NPUStream& hcclStream = hcclStreams_[key][i];
        (*(work->hcclEndEvents_))[i].record(hcclStream);
        ASCEND_LOGI("Event: record hccl work is successfully executed, event=%p", (*(work->hcclEndEvents_))[i].event());
        work->hcclComms_[i] = hcclComms[i];
    }
    work->blockingWait_ = blockingWait_;
    work->opTimeout_ = options_->timeout;
    work->store_ = store_;
    if (asyncErrorHandling_ != NoHandling) {
        workEnqueue(work);
    }
    
    return work;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::collectiveCoalesced(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    c10d::OpType opType)
{
        // Bump collective counter
    seq_++;
    op_id_++;

    const auto devices = getDevice(inputs);
    auto key = getKeyFromDevice(devices);
    std::vector<std::shared_ptr<HCCLComm>> hcclComms;
    if (!options_->hccl_config.empty()) {
        HcclCommConfig config = createHcclCommConfigWithOptions();
        hcclComms = getHCCLComm(key, devices, HcclCommType::DEFAULT, &config);
    } else {
        hcclComms = getHCCLComm(key, devices);
    }

    // Used many times below, so we stash the unordered_map lookup
    auto& hcclStreams = hcclStreams_[key];
    // First let HCCL streams wait for input tensors allocation streams
    syncStreams(devices, hcclEvents_[key], hcclStreams);
    // Work itself will create the events on all NPUs of tensors
    auto work = initWork(devices, rank_, opType);
    // Store references to outputs to be used by WorkHCCL::result and operator<<.
    work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);
    c10_npu::OptionalNPUGuard npuGuard;

    if (desyncDebug_) {
        c10_npu::NPUStream& hcclStream = hcclStreams[0];
        (*(work->hcclStartEvents_))[0].record(hcclStream);
    }

    // No need to detect batch_isend_irecv inputs is incorrect, need require special treatments.
    if (c10_npu::model_state().get_model_mode() == c10_npu::ModelMode::L_TRAIN
        && c10_npu::option::OptionsManager::GetSilenceCheckFlag() != c10_npu::option::CHECK_CLOSE
        && opType != c10d::OpType::UNKNOWN) {
        for (const auto i : c10::irange(inputs.size())) {
            npuGuard.set_index(devices[0].index());
            c10_npu::NPUStreamGuard guard(hcclStreams[0]);
            silenceCheck(inputs[i], opType);
        }
    }

    pre(hcclStreams, work);

    if (nslb_path != nullptr && !nslb_is_end) {
        auto nslb_num = c10_npu::option::OptionsManager::GetNslbCntVal();
        if (op_id_ <= nslb_num) {
            size_t dataVol = 0;
            for (auto tensor:inputs) {
                dataVol += tensor.storage().nbytes();
            }
            char* global_rank = getenv("RANK");
            TORCH_CHECK(global_rank != nullptr, "Unable to fetch global rank for NSLB.", DIST_ERROR(ErrCode::NOT_FOUND));
            recordDataVol(opTypeToString(opType), std::to_string(dataVol), atoi(global_rank), hcclComms);
        }
        if (op_id_ >= nslb_num) {
            nslb_is_end = true;
            nslb_record_end();
        }
    }

    static bool perf_dump_enable = c10_npu::option::OptionsManager::CheckPerfDumpEnable();
    if (perf_dump_enable) {
        if (perfdumppath.empty()) {
            auto pid = getpid();
            int device_id = c10_npu::current_device();
            std::ostringstream oss;
            oss << "perf_pt_" << pid << "_" << device_id << ".log";
            std::string log_file_name = oss.str();
            auto perfDumpPath = c10_npu::option::OptionsManager::GetPerfDumpPath();
            char abs_path[PATH_MAX] = {'\0'};
            if (realpath(perfDumpPath.c_str(), abs_path) == nullptr) {
                TORCH_CHECK(0, "perfDumpPath is not realpath.", DIST_ERROR(ErrCode::NOT_FOUND));
            }
            auto path_temp = c10::str(perfDumpPath, "/", log_file_name);
            if (isFileExists(path_temp)) {
                perfdumppath = path_temp;
                std::ofstream outfile;
                try {
                    outfile.open(perfdumppath, std::ios::app);
                } catch (std::exception& e) {
                    throw std::runtime_error("Open shared directory failed. Please check whether perfdumppath is valid." + DIST_ERROR(ErrCode::NOT_FOUND));
                }

                const std::vector<uint32_t>& ranks = groupRanks();
                outfile << "[GLOBAL RANKID]:" << ranks[rank_] << "\n";
                
                outfile.close();
            }
        } else {
            recordComm(perfdumppath, opTypeToString(opType), rank_, hcclComms);
        }
    }

    for (const auto i : c10::irange(inputs.size())) {
        npuGuard.set_index(devices[0].index());
        c10_npu::NPUStream& hcclStream = hcclStreams[0];

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
            npuGuard.set_index(devices[0].index());
            // to avoid to much task pushed to the stream, leading to stream overflow
            // insert sync point fluxLimit(key, i)
            c10_npu::NPUStream& hcclStream = hcclStreams[0];
            hcclUs startut = TIME_NOW();
            HCCL_CHECK_ERROR(fn(inputs[i], outputs[i], hcclComms[0]->getHcclComm(), hcclStream, work->is_dispatched), opTypeToString(opType).c_str());
            if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::ERASE_RECORD_STREAM) {
                work->recorded_outputs_.push_back(
                    std::make_pair(outputs[i].storage().getWeakStorageImpl(), hcclStream));
            } else if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::AVOID_RECORD_STREAM) {
                work->stashed_for_allocator_safety_.push_back(outputs[i]);
            }
        }
    }
    post(hcclStreams, work);
    {
        c10_npu::NPUMultiStreamGuard guard(hcclStreams);
        work->future_ = c10::make_intrusive<at::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()),
            devices);
        work->future_->markCompleted(at::IValue(*work->outputs_));
    }

    c10_npu::NPUStream& hcclStream = hcclStreams_[key][0];
    (*(work->hcclEndEvents_))[0].record(hcclStream);
    ASCEND_LOGI("Event: record hccl work is successfully executed, event=%p", (*(work->hcclEndEvents_))[0].event());
    work->hcclComms_[0] = hcclComms[0];

    work->blockingWait_ = blockingWait_;
    work->opTimeout_ = options_->timeout;
    work->store_ = store_;
    if (asyncErrorHandling_ != NoHandling) {
        workEnqueue(work);
    }
    
    return work;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::pointToPoint(
    std::vector<at::Tensor>& tensors,
    Fn fn,
    int peer,
    c10d::OpType opType,
    PreProcess pre,
    PostProcess post)
{
    const auto devices = getDeviceList(tensors);
    int p2pRank = 0, p2pTargetRank = 0;
    bool isSendRecvSelf = false;

    std::string key;
    std::vector<std::shared_ptr<HCCLComm>> hcclComms;

    if (hcclCommInitRootInfoConfigExist() && c10_npu::option::OptionsManager::GetP2PBufferSize() != 0) {
        key = getKeySendRecv(rank_, peer);
        p2pRank = rank_ <= peer ? 0 : 1;
        isSendRecvSelf = rank_ == peer;
        p2pTargetRank = isSendRecvSelf ? 0 : 1 - p2pRank;
        hcclComms = getHCCLComm(key, devices, HcclCommType::P2P, nullptr, p2pRank);
    } else {
        p2pTargetRank = peer;
        key = getKeyFromDevices(devices);
        hcclComms = getHCCLComm(key, devices);
    }

    // Bump the logical operation counter regardless of whether this op is
    // coalesced or individual
    op_id_++;

    // First let HCCL streams wait for input tensors allocation streams
    syncStreams(devices, hcclEvents_[key], hcclStreams_[key]);

    // Work itself will create the CUDA events on all NPUs of tensors
    auto work = initWork(devices, rank_, opType);
    // This bypasses something in Work() that crashes if {tensor} is given as
    // output, not sure what
    work->outputs_ = std::make_shared<std::vector<at::Tensor>>(tensors);

    // is npuGuard needed for the if block below, or can i swap them
    c10_npu::OptionalNPUGuard npuGuard;

    if (desyncDebug_) {
        for (const auto i : c10::irange(devices.size())) {
            c10_npu::NPUStream& hcclStream = hcclStreams_[key][i];
            (*(work->hcclStartEvents_))[i].record(hcclStream);
        }
    }

    // No need to detect recv. batch_isend_irecv inputs is incorrect, need require special treatments.
    if (c10_npu::model_state().get_model_mode() == c10_npu::ModelMode::L_TRAIN
        && c10_npu::option::OptionsManager::GetSilenceCheckFlag() != c10_npu::option::CHECK_CLOSE
        && opType == c10d::OpType::SEND) {
        for (const auto i : c10::irange(tensors.size())) {
            npuGuard.set_index(devices[i].index());
            c10_npu::NPUStreamGuard guard(hcclStreams_[key][i]);
            silenceCheck(tensors[i], opType);
        }
    }

    pre(hcclStreams_[key], work);

    if (nslb_path != nullptr && !nslb_is_end) {
        auto nslb_num = c10_npu::option::OptionsManager::GetNslbCntVal();
        if (op_id_ <= nslb_num) {
            size_t dataVol = 0;
            for (auto tensor : tensors) {
                dataVol += tensor.storage().nbytes();
            }
            char* global_rank = getenv("RANK");
            TORCH_CHECK(global_rank != nullptr, "Unable to fetch global rank for NSLB.",
                        DIST_ERROR(ErrCode::NOT_FOUND));
            recordDataVol(opTypeToString(opType), std::to_string(dataVol), atoi(global_rank), hcclComms);
        }
        if (op_id_ >= nslb_num) {
            nslb_is_end = true;
            nslb_record_end();
        }
    }

    static bool perf_dump_enable = c10_npu::option::OptionsManager::CheckPerfDumpEnable();
    if (perf_dump_enable) {
        if (perfdumppath.empty()) {
            auto pid = getpid();
            int device_id = c10_npu::current_device();
            std::ostringstream oss;
            oss << "perf_pt_" << pid << "_" << device_id << ".log";
            std::string log_file_name = oss.str();
            auto perfDumpPath = c10_npu::option::OptionsManager::GetPerfDumpPath();
            char abs_path[PATH_MAX] = {'\0'};
            if (realpath(perfDumpPath.c_str(), abs_path) == nullptr) {
                TORCH_CHECK(0, "perfDumpPath is not realpath.", DIST_ERROR(ErrCode::NOT_FOUND));
            }
            auto path_temp = c10::str(perfDumpPath, "/", log_file_name);
            if (isFileExists(path_temp)) {
                perfdumppath = path_temp;
                std::ofstream outfile;
                try {
                    outfile.open(perfdumppath, std::ios::app);
                } catch (std::exception& e) {
                    throw std::runtime_error("Open shared directory failed. Please check whether perfdumppath is valid." + DIST_ERROR(ErrCode::NOT_FOUND));
                }

                const std::vector<uint32_t>& ranks = groupRanks();
                outfile << "[GLOBAL RANKID]:" << ranks[rank_] << "\n";
                
                outfile.close();
            }
        } else {
            recordComm(perfdumppath, opTypeToString(opType), rank_, hcclComms);
        }
    }

    for (const auto i : c10::irange(tensors.size())) {
        npuGuard.set_index(devices[i].index());
        c10_npu::NPUStream& hcclStream = hcclStreams_[key][i];

        // Both send tensor and recv tensor are created on a worker stream and used
        // in different hcclStreams.  Hence, both must record the hcclStream to
        // prevent being freed before the collective finishes.
        //
        // See [Sync Streams].
        if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::AVOID_RECORD_STREAM) {
            work->stashed_for_allocator_safety_.push_back(tensors[i]);
        } else {
            c10_npu::NPUCachingAllocator::recordStream(tensors[i].storage().data_ptr(), hcclStream);
            if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::ERASE_RECORD_STREAM) {
                work->recorded_inputs_.push_back(std::make_pair(tensors[i].storage().getWeakStorageImpl(), hcclStream));
            }
        }
    }
    {
        for (const auto i : c10::irange(tensors.size())) {
            npuGuard.set_index(devices[i].index());
            // to avoid to much task pushed to the stream, leading to stream overflow
            // insert sync point fluxLimit(key, i)
            c10_npu::NPUStream& hcclStream = hcclStreams_[key][i];
            hcclUs startut = TIME_NOW();
            HCCL_CHECK_ERROR(fn(tensors[i], hcclComms[i]->getHcclComm(), hcclStream, work->is_dispatched, p2pTargetRank), opTypeToString(opType).c_str());
        }
    }
    post(hcclStreams_[key], work);

    // End event should only be recorded after the hcclGroupEnd()
    for (const auto i : c10::irange(tensors.size())) {
        c10_npu::NPUStream& hcclStream = hcclStreams_[key][i];
        (*(work->hcclEndEvents_))[i].record(hcclStream);
        work->hcclComms_[i] = hcclComms[i];
        work->blockingWait_ = blockingWait_;
        work->opTimeout_ = options_->timeout;
        work->store_ = store_;
    }
    // Future only needs to be created and marked completed with outputs for
    // recv(), but still create future for use cases such as profiling even for
    // send().
    {
        c10_npu::NPUMultiStreamGuard guard(hcclStreams_[key]);
        work->future_ = c10::make_intrusive<at::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()),
            devices);
        work->future_->markCompleted(at::IValue(*work->outputs_));
    }

    if (asyncErrorHandling_ != NoHandling) {
        workEnqueue(work);
    }

    return work;
}

template <typename Fn>
c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    c10d::OpType opType)
{
    return collective(
        inputs,
        outputs,
        fn,
        [](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        [](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        opType);
}

template <typename Fn>
c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::pointToPoint(
    std::vector<at::Tensor>& tensors,
    Fn fn,
    int peer,
    c10d::OpType opType)
{
    return pointToPoint(
        tensors,
        fn,
        peer,
        opType,
        [](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        [](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {});
}

int g_allreduceID = 0;
c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceOptions& opts)
{
    check_npu_tensors_different_devices(tensors);

    if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {
        at_npu::native::OpHook::GetInstance().PreHook("allreduce", tensors);
    }

    std::vector<at::Tensor> tensors_cp = {tensors[0]};
    std::string functionName = __FUNCTION__;
    return collective(
        tensors_cp,
        tensors_cp,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            auto hcclType = getHcclDataType(input.scalar_type());
            checkSupportedDataType(hcclType, functionName);
            RECORD_FUNCTION("HcclAllreduce", std::vector<c10::IValue>({input}));

            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto hcclReduceOp = getHcclReduceOp(opts.reduceOp, input);
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream, is_dispatched]() -> int {
                torch_npu::profiler::MstxRange range(
                    getMstxHcclMsg("HcclAllreduce", numel, hcclType, comm), stream.stream(false));
                auto hccl_result = HcclAllReduce(
                    inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand::RunOpApi("HcclAllreduce", hccl_call);

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {
            if (tensors[0].scalar_type() == at::kBool || tensors[0].scalar_type() == at::kByte) {
                c10_npu::NPUStreamGuard guard(hcclStreams[0]);
                tensors_cp[0] = at_npu::native::custom_ops::npu_dtype_cast(tensors[0], at::kInt);
            }
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {
            if (tensors_cp[0].scalar_type() != tensors[0].scalar_type()) {
                c10_npu::NPUStreamGuard guard(hcclStreams[0]);
                c10_npu::NPUCachingAllocator::recordStream(tensors_cp[0].storage().data_ptr(), hcclStreams[0]);
                tensors[0].copy_(tensors_cp[0]);
            }
        },
        c10d::OpType::ALLREDUCE);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::batch_isend_irecv(
    std::vector<std::string>& op_type,
    std::vector<at::Tensor>& tensors,
    std::vector<uint32_t> remote_rank_list)
{
    if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {
        at_npu::native::OpHook::GetInstance().PreHook("batch_isend_irecv", tensors);
    }

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
                torch_npu::profiler::MstxRange range(
                    getMstxHcclMsg("HcclBatchSendRecv", sendRecvInfo[0].count, sendRecvInfo[0].dataType, comm),
                    stream.stream(false));
			    auto hccl_result = hcclBatchIsendIrecv(sendRecvInfo, itemNum, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
			};
            at_npu::native::OpCommand::RunOpApi("HcclBatchSendRecv", hccl_call);
            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {
            // No need to detect recv.
            if (c10_npu::model_state().get_model_mode() == c10_npu::ModelMode::L_TRAIN
                && c10_npu::option::OptionsManager::GetSilenceCheckFlag() != c10_npu::option::CHECK_CLOSE) {
                for (size_t i = 0; i < op_type.size(); ++i) {
                    if (op_type[i] != "irecv") {
                        c10_npu::NPUStreamGuard guard(hcclStreams[0]);
                        silenceCheck(tensors[i], c10d::OpType::SEND);
                    }
                }
            }
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        c10d::OpType::UNKNOWN);
}

int g_broadcastID = 100000;
c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const c10d::BroadcastOptions& opts)
{
    check_npu_tensors_different_devices(tensors);

    if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {
        at_npu::native::OpHook::GetInstance().PreHook("broadcast", tensors);
    }

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
                torch_npu::profiler::MstxRange range(
                    getMstxHcclMsg("HcclBroadcast", numel, hcclType, comm), stream.stream(false));
                auto hccl_result = HcclBroadcast(inputDataPtr, numel, hcclType, root, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand::RunOpApi("HcclBroadcast", hccl_call);

            return HCCL_SUCCESS;
        },
        c10d::OpType::BROADCAST);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceCoalescedOptions& opts)
{
    check_npu_tensors_same_device(tensors);
    std::vector<at::Tensor> tensors_cp = tensors;
    std::string functionName = __FUNCTION__;
    return collectiveCoalesced(
        tensors_cp,
        tensors_cp,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            auto hcclType = getHcclDataType(input.scalar_type());
            checkSupportedDataType(hcclType, functionName);
            RECORD_FUNCTION("HcclAllreduce", std::vector<c10::IValue>({input}));

            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto hcclReduceOp = getHcclReduceOp(opts.reduceOp, input);
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream, is_dispatched]() -> int {
                torch_npu::profiler::MstxRange range(
                    getMstxHcclMsg("HcclAllreduce", numel, hcclType, comm), stream.stream(false));
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
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {
            for (const auto i : c10::irange(tensors.size())) {
                if (tensors[i].scalar_type() == at::kBool || tensors[i].scalar_type() == at::kByte) {
                    c10_npu::NPUStreamGuard guard(hcclStreams[0]);
                    tensors_cp[i] = at_npu::native::custom_ops::npu_dtype_cast(tensors[i], at::kInt);
                }
            }
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {
            for (const auto i : c10::irange(tensors.size())) {
                if (tensors_cp[i].scalar_type() != tensors[i].scalar_type()) {
                    c10_npu::NPUStreamGuard guard(hcclStreams[0]);
                    c10_npu::NPUCachingAllocator::recordStream(tensors_cp[i].storage().data_ptr(), hcclStreams[0]);
                    tensors[i].copy_(tensors_cp[i]);
                }
            }
        },
        c10d::OpType::ALLREDUCE);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const c10d::ReduceOptions& opts)
{
    check_npu_tensors_different_devices(tensors);

    if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {
        at_npu::native::OpHook::GetInstance().PreHook("reduce", tensors);
    }

    std::string functionName = __FUNCTION__;
    uint64_t rank = opts.rootRank;
    std::vector<at::Tensor> tensors_cp = {tensors[0]};
    return collective(
        tensors_cp,
        tensors_cp,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            auto hcclType = getHcclDataType(input.scalar_type());
            checkSupportedDataType(hcclType, functionName);
            RECORD_FUNCTION("HcclReduce", std::vector<c10::IValue>({input}));

            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto reduceOp = getHcclReduceOp(opts.reduceOp, input);
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, reduceOp, rank, comm, stream, is_dispatched]() -> int {
                torch_npu::profiler::MstxRange range(
                    getMstxHcclMsg("HcclReduce", numel, hcclType, comm), stream.stream(false));
                auto hccl_result = hcclReduce(
                    inputDataPtr, outputDataPtr, numel, hcclType, reduceOp, rank, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand::RunOpApi("HcclReduce", hccl_call);

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {
            if (tensors[0].scalar_type() == at::kBool || tensors[0].scalar_type() == at::kByte) {
                c10_npu::NPUStreamGuard guard(hcclStreams[0]);
                tensors_cp[0] = at_npu::native::custom_ops::npu_dtype_cast(tensors[0], at::kInt);
            }
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {
            if (tensors_cp[0].scalar_type() != tensors[0].scalar_type()) {
                c10_npu::NPUStreamGuard guard(hcclStreams[0]);
                c10_npu::NPUCachingAllocator::recordStream(tensors_cp[0].storage().data_ptr(), hcclStreams[0]);
                tensors[0].copy_(tensors_cp[0]);
            }
        },
        c10d::OpType::REDUCE);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::_reduce_oop(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const c10d::ReduceOptions& opts)
{
    if (outputTensor.numel() != inputTensor.numel()) {
        TORCH_CHECK(false, "output tensor must have the same numel as input tensor", DIST_ERROR(ErrCode::PARAM));
    }
    uint64_t rank = opts.rootRank;
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    std::string functionName = __FUNCTION__;
    return collective(
        inputTensors,
        outputTensors,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            auto hcclType = getHcclDataType(input.scalar_type());
            checkSupportedDataType(hcclType, functionName);
            RECORD_FUNCTION("HcclReduce", std::vector<c10::IValue>({input}));

            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto reduceOp = getHcclReduceOp(opts.reduceOp, input);
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, reduceOp, rank, comm, stream, is_dispatched]() -> int {
                torch_npu::profiler::MstxRange range(
                    getMstxHcclMsg("HcclReduce", numel, hcclType, comm), stream.stream(false));
                auto hccl_result = hcclReduce(
                    inputDataPtr, outputDataPtr, numel, hcclType, reduceOp, rank, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand::RunOpApi("HcclReduce", hccl_call);

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {
            if (inputTensors[0].scalar_type() == at::kBool || inputTensors[0].scalar_type() == at::kByte) {
                c10_npu::NPUStreamGuard guard(hcclStreams[0]);
                inputTensors[0] = at_npu::native::custom_ops::npu_dtype_cast(inputTensors[0], at::kInt);
            }
            if (outputTensors[0].scalar_type() == at::kBool || outputTensors[0].scalar_type() == at::kByte) {
                c10_npu::NPUStreamGuard guard(hcclStreams[0]);
                outputTensors[0] = at_npu::native::custom_ops::npu_dtype_cast(outputTensors[0], at::kInt);
            }
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {
            if (outputTensors[0].scalar_type() != outputTensor.scalar_type()) {
                c10_npu::NPUStreamGuard guard(hcclStreams[0]);
                c10_npu::NPUCachingAllocator::recordStream(outputTensors[0].storage().data_ptr(), hcclStreams[0]);
                outputTensor.copy_(outputTensors[0]);
            }
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

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::_reduce_scatter_base_uneven(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& inputSplitSizes,
    const c10d::ReduceScatterOptions& opts)
{
    check_npu_single_tensor(outputTensor);
    check_npu_single_tensor(inputTensor);
    TORCH_CHECK(inputTensor.dtype() == outputTensor.dtype(), "output tensor must have the same type as input tensor", DIST_ERROR(ErrCode::PARAM));
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    check_npu_tensors_different_devices(inputTensors);
    check_npu_tensors_different_devices(outputTensors);

    fill_equal_split_sizes_when_empty(inputSplitSizes, inputTensor, size_);
    check_split_sizes(inputSplitSizes, inputTensor, size_);

    int inputSize = static_cast<int>(inputSplitSizes.size());
    int inputRowSize = static_cast<int>(inputTensor.size(0) ? inputTensor.numel() / inputTensor.size(0) : 1);
    std::vector<uint64_t> inputCounts;
    std::vector<uint64_t> inputSpl;
    inputSpl.push_back(0);
    for (int i = 0; i < inputSize; i++) {
        inputCounts.push_back(static_cast<uint64_t>(inputSplitSizes[i] * inputRowSize));
        if (i > 0) {
            inputSpl.push_back(inputSpl[i - 1] + inputCounts[i - 1]);
        }
    }

    auto inputTensors_ = cast_to_origin_format(inputTensors);
    auto outputTensors_ = cast_to_origin_format(outputTensors);

    return collective(
        inputTensors_,
        outputTensors_,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            RECORD_FUNCTION("HcclReduceScatterV", std::vector<c10::IValue>({input}));
            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            uint64_t outputCount = output.numel();
            auto numel = getNumelForHCCL(output);
            auto hcclReduceOp = getHcclReduceOp(opts.reduceOp, input);
            auto hcclType = getHcclDataType(input.scalar_type());
            auto hccl_call = [
                inputDataPtr,
                inputCounts,
                inputSpl,
                outputDataPtr,
                outputCount,
                hcclType,
                hcclReduceOp,
                numel,
                comm,
                stream,
                is_dispatched]() -> int {
                    torch_npu::profiler::MstxRange range(
                        getMstxHcclMsg("HcclReduceScatterV", numel, hcclType, comm),
                        stream.stream(false));
                    auto hccl_result = hcclReduceScatterV(
                        inputDataPtr,
                        inputCounts.data(),
                        inputSpl.data(),
                        outputDataPtr,
                        outputCount,
                        hcclType,
                        hcclReduceOp,
                        comm,
                        stream.stream(false));
                    *is_dispatched = true;
                    return hccl_result;
            };
            at_npu::native::OpCommand::RunOpApi("HcclReduceScatterV", hccl_call);

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        c10d::OpType::REDUCE_SCATTER);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::_allgather_base_uneven(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    const c10d::AllgatherOptions& opts)
{
    check_npu_single_tensor(outputTensor);
    check_npu_single_tensor(inputTensor);
    TORCH_CHECK(inputTensor.dtype() == outputTensor.dtype(), "output tensor must have the same type as input tensor", DIST_ERROR(ErrCode::PARAM));
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    check_npu_tensors_different_devices(inputTensors);
    check_npu_tensors_different_devices(outputTensors);

    fill_equal_split_sizes_when_empty(outputSplitSizes, outputTensor, size_);
    check_split_sizes(outputSplitSizes, outputTensor, size_);

    int outputSize = static_cast<int>(outputSplitSizes.size());
    int outputRowSize = static_cast<int>(outputTensor.size(0) ? outputTensor.numel() / outputTensor.size(0) : 1);
    std::vector<uint64_t> outputCounts;
    std::vector<uint64_t> outputSpl;
    outputSpl.push_back(0);
    for (int i = 0; i < outputSize; i++) {
        outputCounts.push_back(static_cast<uint64_t>(outputSplitSizes[i] * outputRowSize));
        if (i > 0) {
            outputSpl.push_back(outputSpl[i - 1] + outputCounts[i - 1]);
        }
    }

    auto inputTensors_ = cast_to_origin_format(inputTensors);
    auto outputTensors_ = cast_to_origin_format(outputTensors);

    return collective(
        inputTensors_,
        outputTensors_,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
            RECORD_FUNCTION("HcclAllGatherV", std::vector<c10::IValue>({input}));
            auto inputDataPtr = input.data_ptr();
            uint64_t inputCount = input.numel();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto hcclType = getHcclDataType(input.scalar_type());
            auto hccl_call = [
                inputDataPtr,
                inputCount,
                outputDataPtr,
                outputCounts,
                outputSpl,
                hcclType,
                numel,
                comm,
                stream,
                is_dispatched]() -> int {
                    torch_npu::profiler::MstxRange range(
                        getMstxHcclMsg("HcclAllGatherV", numel, hcclType, comm),
                        stream.stream(false));
                    auto hccl_result = hcclAllGatherV(
                        inputDataPtr,
                        inputCount,
                        outputDataPtr,
                        outputCounts.data(),
                        outputSpl.data(),
                        hcclType,
                        comm,
                        stream.stream(false));
                    *is_dispatched = true;
                    return hccl_result;
            };
            at_npu::native::OpCommand::RunOpApi("HcclAllGatherV", hccl_call);

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        c10d::OpType::ALLGATHER);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts)
{
    check_npu_tensors_different_devices(inputTensors);

    if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {
        at_npu::native::OpHook::GetInstance().PreHook("allgather", outputTensors, inputTensors);
    }

    auto inputTensors_ = cast_to_origin_format(inputTensors);
    bool same_size = check_same_size(outputTensors.back());
    if (same_size) {
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
                    torch_npu::profiler::MstxRange range(
                        getMstxHcclMsg("HcclAllGather", numel, hcclType, comm), stream.stream(false));
                    auto hccl_result = HcclAllGather(inputDataPtr, outputDataPtr, numel, hcclType, comm, stream.stream(false));
                    *is_dispatched = true;
                    return hccl_result;
                };
                at_npu::native::OpCommand::RunOpApi("HcclAllgather", hccl_call);

                return HCCL_SUCCESS;
            },
            [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
            [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {
                work->lazyDestroy(byte_alignment_inputTensors_);
                work->lazyDestroy(outputFlattened);
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
    } else {
        TORCH_NPU_WARN_ONCE("The current allgather operator has a defect in handling different tensor shape, \
        the work event forces a wait operation, and the allgather wait on the python side would be fake");
        const auto num_devices = outputTensors.size();
        const auto num_reduces = outputTensors[0].size();
        std::vector<c10::intrusive_ptr<c10d::Work>> works;
        // Need to add a method like startCoalescing();
        for (const auto i : c10::irange(num_reduces)) {
            std::vector<at::Tensor> inputs_multi_dev(num_devices);
            std::vector<at::Tensor> outputs_multi_dev(num_devices);
            for (const auto j : c10::irange(num_devices)) {
                // @lint-ignore CLANGTIDY
                outputs_multi_dev[j] = outputTensors[j][i];
                if (i == (rank_ * num_devices + j)) {
                    outputs_multi_dev[j].copy_(inputTensors[j]);
                }
            }

            auto broadcastOpts = c10d::BroadcastOptions{
                static_cast<int64_t>(i / num_devices),
                static_cast<int64_t>(i % num_devices),
                opts.timeout};
            
            auto work = collective(
                outputs_multi_dev, outputs_multi_dev, [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
                RECORD_FUNCTION("HcclBroadcast", std::vector<c10::IValue>({input}));
                const auto root = broadcastOpts.rootRank * inputs_multi_dev.size() + broadcastOpts.rootTensor;

                auto inputDataPtr = input.data_ptr();
                auto numel = getNumelForHCCL(input);
                auto hcclType = getHcclDataType(input.scalar_type());
                torch_npu::profiler::MstxRange range(
                    getMstxHcclMsg("HcclBroadcast", numel, hcclType, comm), stream.stream(false));
                auto hccl_result = HcclBroadcast(inputDataPtr, numel, hcclType, root, comm, stream.stream());
                *is_dispatched = true;
                return hccl_result;
                },
                c10d::OpType::BROADCAST);
            works.push_back(work);
        }
        // Need to add a method like endCoalescing();
        for (auto& work : works) {
            work->wait();
        }
        // Create a fake_work for python side;
        auto fake_work = initWork(getDeviceList(inputTensors), rank_, c10d::OpType::ALLGATHER);
        return fake_work;
    }
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const c10d::AllgatherOptions& opts)
{
    auto inputTensors_ = cast_to_origin_format(inputs);
    return collectiveCoalesced(
        inputTensors_,
        outputs,
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
                torch_npu::profiler::MstxRange range(
                    getMstxHcclMsg("HcclAllGather", numel, hcclType, comm), stream.stream(false));
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
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        c10d::OpType::ALLGATHER);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::allgather_togather(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts)
{
    check_npu_tensors_different_devices(inputTensors);
    check_npu_tensors_different_devices(outputTensors);

    if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {
        at_npu::native::OpHook::GetInstance().PreHook("allgather_togather", outputTensors, inputTensors);
    }

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
                torch_npu::profiler::MstxRange range(
                    getMstxHcclMsg("HcclAllGather", numel, hcclType, comm), stream.stream(false));
                auto hccl_result = HcclAllGather(inputDataPtr, outputDataPtr, numel, hcclType, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand::RunOpApi("HcclAllGather", hccl_call);

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        c10d::OpType::ALLGATHER);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::_allgather_base(
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

    if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {
        at_npu::native::OpHook::GetInstance().PreHook("_allgather_base", outputTensors, inputTensors);
    }

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
                torch_npu::profiler::MstxRange range(
                    getMstxHcclMsg("HcclAllGather", numel, hcclType, comm), stream.stream(false));
                auto hccl_result = HcclAllGather(inputDataPtr, outputDataPtr, numel, hcclType, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand::RunOpApi("HcclAllGather", hccl_call);

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        c10d::OpType::ALLGATHER);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ReduceScatterOptions& opts)
{
    check_npu_tensors_different_devices(outputTensors);

    if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {
        at_npu::native::OpHook::GetInstance().PreHook("reduce_scatter", outputTensors, inputTensors);
    }

    bool same_size = check_same_size(inputTensors.back());
    if (same_size) {
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
                torch_npu::profiler::MstxRange range(
                    getMstxHcclMsg("HcclReduceScatter", numel, hcclType, comm), stream.stream(false));
                auto hccl_result = HcclReduceScatter(
                    inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand::RunOpApi("HcclReduceScatter", hccl_call);

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {
            work->lazyDestroy(inputFlattened);
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
        c10d::OpType::REDUCE_SCATTER);
    } else {
        TORCH_NPU_WARN_ONCE("The current reduce_scatter operator has a defect in handling different tensor shape,",
            "the work event forces a wait operation in c++ side, and the reduce_scatter wait on the python side would be fake");
        auto outputTensor = outputTensors.back();
        auto inputTensors_ = inputTensors.back();
        const auto num_reduces = inputTensors_.size();
        std::vector<c10::intrusive_ptr<c10d::Work>> works;
        for (const auto i : c10::irange(num_reduces)) {
            auto& input = inputTensors_[i];
            auto& output = (i == rank_) ? outputTensor : input;
            auto reduceOpts = c10d::ReduceOptions{
                opts.reduceOp,
                static_cast<int64_t>(i),
                static_cast<int64_t>(0),
                opts.timeout};
            auto work = _reduce_oop(output, input, reduceOpts);
            works.push_back(work);
        }
        // Need to add a method like endCoalescing();
        for (auto& work : works) {
            work->wait();
        }
        // Create a fake_work for python side;
        auto fake_work = initWork(getDeviceList(outputTensors), rank_, c10d::OpType::REDUCE_SCATTER);
        return fake_work;
    }
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::_reduce_scatter_base(
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

    if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {
        at_npu::native::OpHook::GetInstance().PreHook("_reduce_scatter_base", outputs, inputs);
    }

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
                torch_npu::profiler::MstxRange range(
                    getMstxHcclMsg("HcclReduceScatter", numel, hcclType, comm), stream.stream(false));
                auto hccl_result = HcclReduceScatter(
                    inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand::RunOpApi("HcclReduceScatter", hccl_call);

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        c10d::OpType::REDUCE_SCATTER);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::ReduceScatterOptions& opts)
{
    std::string functionName = __FUNCTION__;
    return collectiveCoalesced(
        inputTensors,
        outputTensors,
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
                torch_npu::profiler::MstxRange range(
                    getMstxHcclMsg("HcclReduceScatter", numel, hcclType, comm), stream.stream(false));
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
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        c10d::OpType::REDUCE_SCATTER);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::barrier(const c10d::BarrierOptions& opts)
{
    std::vector<at::Device> devices;
    if (usedDeviceIdxs_.empty()) {
        auto numNPUs = c10_npu::device_count();
        int16_t deviceIdx = static_cast<int16_t>(rank_ % std::max(static_cast<int>(numNPUs), 1));
        devices.push_back(at::Device(c10::DeviceType::PrivateUse1));
    } else {
        for (auto usedDeviceIdx : usedDeviceIdxs_) {
            devices.push_back(at::Device(c10::DeviceType::PrivateUse1, usedDeviceIdx));
        }
    }

    std::vector<at::Tensor> barrierTensors;
    barrierTensors.reserve(devices.size());

    c10_npu::OptionalNPUGuard npuGuard;
    for (auto& device : devices) {
        npuGuard.set_index(device.index());
        barrierTensors.push_back(
            at::ones({1}, at::TensorOptions().device(c10::DeviceType::PrivateUse1).dtype(at::kFloat)));
    }

    auto work = allreduce(barrierTensors);

    // Work will take over barrierTensors
    auto hcclWork = dynamic_cast<ProcessGroupHCCL::WorkHCCL*>(work.get());
    TORCH_CHECK(hcclWork, DIST_ERROR(ErrCode::PARAM));
    hcclWork->barrierTensors_ = std::move(barrierTensors);

    return work;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const c10d::GatherOptions& /* unused */)
{
    throw std::runtime_error("ProcessGroupHCCL does not support gather" + DIST_ERROR(ErrCode::NOT_SUPPORT));
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::scatter(
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

    if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {
        at_npu::native::OpHook::GetInstance().PreHook("scatter", outputTensors, inputTensors);
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
                torch_npu::profiler::MstxRange range(
                    getMstxHcclMsg("HcclScatter", numel, hcclType, comm), stream.stream(false));
                auto hccl_result = hcclScatter(inputDataPtr, outputDataPtr, numel, hcclType, root, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand::RunOpApi("HcclScatter", hccl_call);

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {
            work->lazyDestroy(inputFlattened);
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

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::send(std::vector<at::Tensor>& tensors, int dstRank, int tag)
{
    check_npu_tensors_different_devices(tensors);

    if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {
        at_npu::native::OpHook::GetInstance().PreHook("send", tensors);
    }

    auto tensors_ = cast_to_origin_format(tensors);
    auto ret = pointToPoint(
        tensors_,
        [&](at::Tensor& input, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched, int dst_rank) {
            RECORD_FUNCTION("HcclSend", std::vector<c10::IValue>({input}));
            auto inputDataPtr = input.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto hcclType = getHcclDataType(input.scalar_type());
            auto hccl_call = [inputDataPtr, numel, hcclType, dst_rank, comm, stream, is_dispatched]() -> int {
                torch_npu::profiler::MstxRange range(
                    getMstxHcclMsg("HcclSend", numel, hcclType, comm), stream.stream(false));
                auto hccl_result = HcclSend(inputDataPtr, numel, hcclType, dst_rank, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand::RunOpApi("HcclSend", hccl_call);

            return HCCL_SUCCESS;
        },
        dstRank, c10d::OpType::SEND);
    return ret;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::recv(std::vector<at::Tensor>& tensors, int srcRank, int tag)
{
    check_npu_tensors_different_devices(tensors);

    if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {
        at_npu::native::OpHook::GetInstance().PreHook("recv", tensors);
    }

    auto tensors_ = create_base_format_tensors(tensors);
    auto ret = pointToPoint(
        tensors_,
        [&](at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched, int src_rank) {
            RECORD_FUNCTION("HcclRecv", std::vector<c10::IValue>({output}));
            if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() != c10_npu::option::AVOID_RECORD_STREAM) {
                c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
            }
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(output);
            auto hcclType = getHcclDataType(output.scalar_type());
            auto hccl_call = [outputDataPtr, numel, hcclType, src_rank, comm, stream, is_dispatched]() -> int {
                torch_npu::profiler::MstxRange range(
                    getMstxHcclMsg("HcclRecv", numel, hcclType, comm), stream.stream(false));
                auto hccl_result = HcclRecv(outputDataPtr, numel, hcclType, src_rank, comm, stream.stream(false));
                *is_dispatched = true;
                return hccl_result;
            };
            at_npu::native::OpCommand::RunOpApi("HcclRecv", hccl_call);

            return HCCL_SUCCESS;
        },
        srcRank, c10d::OpType::RECV,
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {
            for (size_t i = 0; i < tensors_.size(); ++i) {
                c10_npu::NPUStreamGuard guard(hcclStreams[i]);
                if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() ==
                    c10_npu::option::AVOID_RECORD_STREAM) {
                    work->stashed_for_allocator_safety_.push_back(tensors_[i]);
                } else {
                    c10_npu::NPUCachingAllocator::recordStream(tensors_[i].storage().data_ptr(), hcclStreams[i]);
                    if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() ==
                        c10_npu::option::ERASE_RECORD_STREAM) {
                        work->recorded_outputs_.push_back(
                            std::make_pair(tensors_[i].storage().getWeakStorageImpl(), hcclStreams[i]));
                    }
                }
                if (!at_npu::native::FormatHelper::IsBaseFormatType(tensors[i])) {
                    tensors[i].copy_(tensors_[i], true);
                }
            }
        });
    return ret;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::recvAnysource(std::vector<at::Tensor>& /* unused */, int /* unused */)
{
    TORCH_CHECK(false, "ProcessGroupHCCL does not support recv", DIST_ERROR(ErrCode::NOT_SUPPORT));
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const c10d::AllToAllOptions& opts)
{
    check_npu_single_tensor(outputTensor);
    check_npu_single_tensor(inputTensor);
    int ranks = getSize();
    TORCH_CHECK(ranks > 0, "Invalid rank count within current process group", ranks, DIST_ERROR(ErrCode::PARAM));
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};

    if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {
        at_npu::native::OpHook::GetInstance().PreHook("alltoall_base", outputTensors, inputTensors);
    }

    auto inputTensors_ = cast_to_origin_format(inputTensors);
    auto outputTensors_ = cast_to_origin_format(outputTensors);

    if (inputSplitSizes.empty() && outputSplitSizes.empty()) {
        TORCH_CHECK(
            outputTensor.numel() == inputTensor.numel() &&
            outputTensor.type() == inputTensor.type(),
            "Tensors are not equal in size or data type",
            DIST_ERROR(ErrCode::PARAM));
        TORCH_CHECK(
            outputTensor.size(0) % ranks == 0,
            "Tensor's dim 0 does not divide equally across group size",
            DIST_ERROR(ErrCode::PARAM));
        uint64_t output_counts = static_cast<uint64_t>(outputTensor.numel() / ranks);
        uint64_t input_counts = static_cast<uint64_t>(inputTensor.numel() / ranks);

        check_npu_tensors_different_devices(inputTensors);
        check_npu_tensors_different_devices(outputTensors);
        return collective(
            inputTensors_,
            outputTensors_,
            [&](at::Tensor& input,
                at::Tensor& output,
                HcclComm comm,
                c10_npu::NPUStream& stream, std::shared_ptr<bool> is_dispatched) {
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
                        torch_npu::profiler::MstxRange range(
                            getMstxHcclMsg("HcclAlltoAll", input_counts, inputhcclDataType, comm),
                            stream.stream(false));
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
                    at_npu::native::OpCommand::RunOpApi("HcclAlltoAll", hccl_call);
                    return HCCL_SUCCESS;
                },
            [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
            [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {
                for (size_t i = 0; i < outputTensors_.size(); ++i) {
                    c10_npu::NPUStreamGuard guard(hcclStreams[i]);
                    if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::AVOID_RECORD_STREAM) {
                        work->stashed_for_allocator_safety_.push_back(outputTensors_[i]);
                    } else {
                        c10_npu::NPUCachingAllocator::recordStream(outputTensors_[i].storage().data_ptr(), hcclStreams[i]);
                        if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::ERASE_RECORD_STREAM) {
                            work->recorded_outputs_.push_back(
                                std::make_pair(outputTensors_[i].storage().getWeakStorageImpl(), hcclStreams[i]));
                        }
                    }
                    if (!at_npu::native::FormatHelper::IsBaseFormatType(outputTensors[i])) {
                        outputTensors[i].copy_(outputTensors_[i], true);
                    }
                }
            },
            c10d::OpType::ALLTOALL_BASE);
    } else {
        uint64_t index = static_cast<uint64_t>(outputTensor.size(0) / ranks);
        if (outputSplitSizes.empty()) {
            for (int i = 0; i < ranks; i++) {
                outputSplitSizes.push_back(index);
            }
        }
        index = static_cast<uint64_t>(inputTensor.size(0) / ranks);
        if (inputSplitSizes.empty()) {
            for (int i = 0; i < ranks; i++) {
                inputSplitSizes.push_back(index);
            }
        }
        check_split_sizes(inputSplitSizes, inputTensor, size_);
        check_split_sizes(outputSplitSizes, outputTensor, size_);

        int inputSize = static_cast<int>(inputSplitSizes.size());
        int outSize = static_cast<int>(outputSplitSizes.size());
        int inputRowSize = static_cast<int>(inputTensor.size(0) ? inputTensor.numel() / inputTensor.size(0) : 1);
        int outputRowSize = static_cast<int>(outputTensor.size(0) ? outputTensor.numel() / outputTensor.size(0) : 1);
        std::vector<uint64_t> inputCounts;
        std::vector<uint64_t> inputSpl;
        std::vector<uint64_t> outputCounts;
        std::vector<uint64_t> outputSpl;
        inputSpl.push_back(0);
        outputSpl.push_back(0);
        for (int i = 0; i < outSize; i++) {
            outputCounts.push_back(static_cast<uint64_t>(outputSplitSizes[i] * outputRowSize));
            if (i > 0) {
                outputSpl.push_back(outputSpl[i - 1] + outputCounts[i - 1]);
            }
        }
        for (int i = 0; i < inputSize; i++) {
            inputCounts.push_back(static_cast<uint64_t>(inputSplitSizes[i] * inputRowSize));
            if (i > 0) {
                inputSpl.push_back(inputSpl[i - 1] + inputCounts[i - 1]);
            }
        }

        check_npu_tensors_different_devices(inputTensors);
        check_npu_tensors_different_devices(outputTensors);
        return collective(
            inputTensors_,
            outputTensors_,
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
                    torch_npu::profiler::MstxRange range(
                        getMstxHcclMsg("HcclAlltoAllV", static_cast<uint64_t>(inputCounts.size()),
                                       inputhcclDataType, comm),
                        stream.stream(false));
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
                at_npu::native::OpCommand::RunOpApi("HcclAlltoAllV", hccl_call);

                return HCCL_SUCCESS;
            },
            [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
            [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {
                for (size_t i = 0; i < outputTensors_.size(); ++i) {
                    c10_npu::NPUStreamGuard guard(hcclStreams[i]);
                    if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::AVOID_RECORD_STREAM) {
                        work->stashed_for_allocator_safety_.push_back(outputTensors_[i]);
                    } else {
                        c10_npu::NPUCachingAllocator::recordStream(outputTensors_[i].storage().data_ptr(), hcclStreams[i]);
                        if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() == c10_npu::option::ERASE_RECORD_STREAM) {
                            work->recorded_outputs_.push_back(
                                std::make_pair(outputTensors_[i].storage().getWeakStorageImpl(), hcclStreams[i]));
                        }
                    }
                    if (!at_npu::native::FormatHelper::IsBaseFormatType(outputTensors[i])) {
                        outputTensors[i].copy_(outputTensors_[i], true);
                    }
                }
            },
            c10d::OpType::ALLTOALL);
    }
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::alltoall(
    std::vector<at::Tensor>& output_tensors,
    std::vector<at::Tensor>& input_tensors,
    const c10d::AllToAllOptions& opts)
{
    auto device = output_tensors[0].device();
    for (const auto r : c10::irange(output_tensors.size())) {
        check_npu_single_tensor(output_tensors[r]);
        check_npu_single_tensor(input_tensors[r]);
        TORCH_CHECK(device == output_tensors[r].device() && device == input_tensors[r].device(),
            "tensors must be on the same device", DIST_ERROR(ErrCode::PARAM));
    }

    if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {
        at_npu::native::OpHook::GetInstance().PreHook("alltoall", output_tensors, input_tensors);
    }

    std::vector<int64_t> output_split_sizes;
    std::vector<int64_t> input_split_sizes;
    std::vector<at::Tensor> output_results;
    std::vector<at::Tensor> input_tensors_flattened;
    std::vector<at::Tensor> output_tensors_flattened;

    for (size_t i = 0; i < input_tensors.size(); i++) {
        input_split_sizes.push_back(input_tensors[i].numel());
        input_tensors_flattened.push_back(at::reshape(input_tensors[i], {input_tensors[i].numel(), 1}));
    }
    for (size_t i = 0; i < output_tensors.size(); i++) {
        output_split_sizes.push_back(output_tensors[i].numel());
        output_tensors_flattened.push_back(at::reshape(output_tensors[i], {output_tensors[i].numel(), 1}));
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

    std::vector<at::Tensor> in_tensors = {at::cat(input_tensors_flattened, 0)};
    std::vector<at::Tensor> out_tensors = {at::cat(output_tensors_flattened, 0)};

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
                torch_npu::profiler::MstxRange range(
                    getMstxHcclMsg("HcclAlltoAllV", static_cast<uint64_t>(input_counts.size()),
                                   inputhcclDataType, comm),
                    stream.stream(false));
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
            at_npu::native::OpCommand::RunOpApi("HcclAlltoAllV", hccl_call);

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
            output_results = at::split(out_tensors[0], output_split_sizes, 0);
            for (int i = 0; i < output_results.size(); i++) {
                output_tensors[i].copy_(at::reshape(output_results[i], output_tensors[i].sizes()), true);
            }
        },
        c10d::OpType::ALLTOALL);
}

} // namespace c10d_npu
