#include <ATen/record_function.h>
#include <algorithm>
#include <map>
#include <tuple>
#include <unordered_set>

#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <c10d/ParamCommsUtils.hpp>
#include <c10d/TraceUtils.h>
#include <c10d/Utils.hpp>

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

namespace c10d_npu {
namespace {
static constexpr uint32_t kOpWaitTimeoutOffset = 30U; // second
static uint32_t kOpWaitTimeout = 1868U; // second

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

int64_t physical_numel(at::Tensor& self)
{
    auto sizes = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.storage_sizes_;
    int64_t n = 1;
    for (auto s : sizes) {
        n *= s;
    }
    return n;
}

// use tensor numel when the format is ACL_FORMAT_ND or ACL_FORMAT_NCHW
uint64_t getNumelForHCCL(at::Tensor& self)
{
    aclFormat format = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.npu_format_;
    if (format != ACL_FORMAT_ND and format != ACL_FORMAT_NCHW) {
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

// Helper function that gets the data type and issues error if not supported
HcclDataType getHcclDataType(at::ScalarType type)
{
    try {
        return kScalarTypeToHcclDataType.at(type);
    } catch (std::out_of_range& e) {
        throw std::runtime_error("Unsupported data type for HCCL process group");
    }
}

std::string getHcclDataTypeSerialString(HcclDataType type)
{
    const auto& iter = kHcclDataTypeToStringMap.find(type);
    if (iter != kHcclDataTypeToStringMap.end()) {
        return iter->second;
    } else {
        TORCH_NPU_WARN_ONCE("Can not serialize undefined hccl data type.");
        return "";
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
    return hcclOp[reduceOp];
}

// AllGather & Broadcast support all data type, no need do more check.
void checkSupportedDataTypeOfAllReduce(HcclDataType type)
{
    static std::set<HcclDataType> allReduceSupportedDataTypes = {
        HCCL_DATA_TYPE_INT8,
        HCCL_DATA_TYPE_INT16,
        HCCL_DATA_TYPE_INT32,
        HCCL_DATA_TYPE_FP16,
        HCCL_DATA_TYPE_FP32,
        HCCL_DATA_TYPE_BFP16,
        HCCL_DATA_TYPE_INT64};
    TORCH_CHECK(
        allReduceSupportedDataTypes.count(type) != 0,
        "HCCL AllReduce & Reduce: Unsupported data type ",
        getHcclDataTypeSerialString(type));
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
    TORCH_CHECK(rank >= 0, "Invalid rank ", rank);
    auto numNPUs = c10_npu::device_count();
    TORCH_CHECK(numNPUs > 0, "Invalid device number", numNPUs);
    int16_t deviceIdx = static_cast<int16_t>(rank % numNPUs);
    return at::Device(c10::DeviceType::PrivateUse1, deviceIdx);
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
        ASCEND_LOGI("Event: record hccl group is successfully executed.");
        hcclEvent.block(hcclStream);
        ASCEND_LOGI("Event: block hccl group is successfully executed.");
    }
}

// Returns exception's what() given an exception_ptr instance.
std::string getExceptionMsgFromExceptionPtr(const std::exception_ptr& exceptionPtr)
{
    TORCH_CHECK(exceptionPtr != nullptr);
    try {
        std::rethrow_exception(exceptionPtr);
    } catch (const std::exception& e) {
        return e.what();
    } catch (...) {
        return "Unknown exception type";
    }
}

// exit call back for allreduce error
void exceptionCallback(aclrtExceptionInfo* exceptionInfo)
{
    // notice: Do not raise error, otherwise we will get call stacks of the rts callback function.
    fprintf(stdout, "Inner error, see details in Ascend logs.");
}
} // namespace

constexpr int64_t kSynchronizeBusyWaitMillis = 10;
constexpr int64_t maxOpNumPerSyncPoint = 2;
const int64_t ProcessGroupHCCL::kProcessGroupHCCLOpTimeoutMillis = 10 * 1000;
thread_local uint64_t ProcessGroupHCCL::hcclActiveGroupCounter_ = 0;
const int64_t ProcessGroupHCCL::kWatchdogThreadSleepMillis = 1000;
// const int64_t ProcessGroupHCCL::kProcessGroupHCCLOpTimeoutMillis = 10 * 1000;

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
    if (desyncDebug) {
        hcclStartEvents_ = std::make_shared<std::vector<c10_npu::NPUEvent>>(devices.size());
    }

    hcclEndEvents_ = std::make_shared<std::vector<c10_npu::NPUEvent>>(devices.size());
    hcclComms_.resize(devices.size());
}

ProcessGroupHCCL::WorkHCCL::WorkHCCL(const WorkHCCL& w)
    : Work(w.rank_, w.opType_),
    std::enable_shared_from_this<WorkHCCL>(w),
    devices_(w.devices_),
    hcclStartEvents_(w.hcclStartEvents_),
    hcclEndEvents_(w.hcclEndEvents_),
    hcclComms_(w.hcclComms_),
    blockingWait_(w.blockingWait_),
    opTimeout_(w.opTimeout_),
    workStartTime_(w.workStartTime_),
    seq_(w.seq_),
    startTraceUpdated_(w.startTraceUpdated_),
    store_(w.store_)
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
        if (std::string(e.what()).find("driver shutting down") == std::string::npos) {
            throw;
        }
        LOG(INFO) << "[Rank " << rank_ << "] Event query failed with exception: " << e.what();
    }

    return true;
}

// check if HCCL task is finished
bool ProcessGroupHCCL::WorkHCCL::finishedNPUExecutionInternal() const
{
  try {
    for (const auto i : c10::irange(devices_.size())) {
        // Checking the work's corresponding ASCEND events' status
        if (!(*hcclEndEvents_)[i].query()) {
            return false;
        }
    }
  } catch (const std::exception& e) {
        if (std::string(e.what()).find("driver shutting down") == std::string::npos) {
            throw;
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
        ASCEND_LOGI("Event: block hccl work is successfully executed.");
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

    if (c10_npu::option::OptionsManager::IsMultiStreamMemoryReuse()) {
        lazy_destory_tensors_.clear();
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
}

void ProcessGroupHCCL::WorkHCCL::lazyDestory(std::vector<at::Tensor> tensors) {
    if (tensors.empty() || !c10_npu::option::OptionsManager::IsMultiStreamMemoryReuse()) {
        return;
    }

    for (const auto i : c10::irange(tensors.size())) {
        lazy_destory_tensors_.push_back(tensors[i]);
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
        hcclComm->destropyHcclComm();
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
    options_(options),
    hcclCommCounter_(0),
    terminateProcessGroup_(false),
    traceKeyStart_("HCCL_" + std::to_string(rank) + "_trace_start"),
    traceKeyEnd_("HCCL_" + std::to_string(rank) + "_trace_end")
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
                throw std::runtime_error("Invalid value for environment variable: " + std::string(HCCL_BLOCKING_WAIT));
            }
        }
    } catch (std::exception& e) {
        throw std::runtime_error("Invalid value for environment variable: " + std::string(HCCL_BLOCKING_WAIT));
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
    hcclCommWatchdogThread_ = std::thread(&ProcessGroupHCCL::hcclCommWatchdog, this);
#endif
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
            hcclComm->destropyHcclComm();
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

        LOG(INFO) << "[Rank " << rank << "] Destroyed " << hcclComms.size()
                << "communicators on ASCEND device " << devName;
    }
}

// Abort all communicators on this rank
void ProcessGroupHCCL::abort(c10::optional<std::string> abortReason)
{
    std::lock_guard<std::mutex> lock(mutex_);
    abortCommsFromMap(devHCCLCommMap_, rank_, abortReason);
}

ProcessGroupHCCL::~ProcessGroupHCCL()
{
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
                hcclComm->destropyHcclComm();
            }
        }
    }
}

void ProcessGroupHCCL::hcclCommWatchdog()
{
    try {
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
        watchDogException_ = std::make_exception_ptr(std::runtime_error(exitMsg));
        std::rethrow_exception(watchDogException_);
    } catch (...) {
        const auto exitMsg = c10::str(
            "[Rank ",
            rank_,
            "] HCCL watchdog thread terminated with exception: unknown");
        LOG(ERROR) << exitMsg;
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

void ProcessGroupHCCL::workCleanupLoop()
{
    bool needSetDevice = true;
    bool done = false;
    std::list<ProcessGroupHCCL::WorkHCCL> completedWorkList;
    while (!done || !terminateProcessGroup_.load()) {
        std::unique_lock<std::mutex> lock(workMetaListMutex_);
        // We busy-poll the work vector every kWatchdogThreadSleepMillis
        // milliseconds as long as the atomic is True.
        workMetaListCV_.wait_for(lock, std::chrono::milliseconds(kWatchdogThreadSleepMillis),
                                 [&]() -> bool { return terminateProcessGroup_.load(); });

        for (auto it = workMetaList_.begin(); it != workMetaList_.end();
             /* no increment */) {
            auto& work = *it;
            if (needSetDevice) {
                NPU_CHECK_ERROR(aclrtSetDevice(static_cast<int>(work.devices_[0].index())));
                needSetDevice = false;
            }
            work.checkAndSetException();
            bool timedOut = work.checkTimeout();

            // If work hits an exception (either an error or timeout)
            if (work.exception()) {
                if (SHOULD_CLEAN_UP(asyncErrorHandling_)) {
                    // Abort work and corresponding communicators
                    work.abort();
                    // PG level abort, which would abort all other communicators on this
                    // rank
                    abort();
                }
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
                it = workMetaList_.erase(it);
            } else {
                // Increment the iterator if the current WorkHCCL object is not
                // completed.
                ++it;
            }
        }

        done = workMetaList_.empty();
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
            TORCH_CHECK(vec.size() == HCCL_ROOT_INFO_BYTES);
            std::memcpy(hcclID, vec.data(), vec.size());
        } catch (const std::exception& e) {
            throw std::runtime_error("store->get() got error: " + std::string(HCCL_BLOCKING_WAIT));
        } catch (...) {
            throw std::runtime_error("Unknown exception: " + std::string(HCCL_BLOCKING_WAIT));
        }
        auto main_list = store_->get(ver_key);
        if (main_list != ver_list) {
            TORCH_NPU_WARN("PTA version mismatch");
        }
    }
}

std::vector<std::shared_ptr<HCCLComm>>& ProcessGroupHCCL::getHCCLComm(
    const std::string& devicesKey,
    const std::vector<at::Device>& devices)
{
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

    // Move the HCCL resource to cache
    devHCCLCommMap_.emplace(devicesKey, std::move(hcclComms));
    return devHCCLCommMap_[devicesKey];
}

namespace {

// Check that all `tensors' have the same type and shape and are distributed
// across distinct NPUs.
void check_npu_tensors_different_devices(const std::vector<at::Tensor>& tensors)
{
    if (tensors.size() == 0) {
        TORCH_CHECK(false, "Tensor list must be nonempty");
    }
    // HCCL support one NPU per process only
    if (tensors.size() != 1) {
        TORCH_CHECK(false, "Tensor list mustn't be larger than the number of available NPUs");
    }
    const auto& first = tensors.front();
    // Set for ensuring that tensors are on separate devices.
    std::unordered_set<decltype(first.get_device())> usedDevices;
    usedDevices.reserve(tensors.size());

    for (const auto& t : tensors) {
        if (!torch_npu::utils::is_npu(t) || t.is_sparse()) {
            TORCH_CHECK(false, "Tensors must be NPU and dense");
        }
        if (t.scalar_type() != first.scalar_type()) {
            TORCH_CHECK(false, "Tensors must have identical type");
        }
        if (t.sizes() != first.sizes()) {
            TORCH_CHECK(false, "Tensors must have identical size");
        }
        if (t.strides() != first.strides()) {
            TORCH_CHECK(false, "Tensors must have identical strides");
        }
        if (!t.is_contiguous(t.suggest_memory_format())) {
            TORCH_CHECK(false, "Tensors must be contiguous");
        }
        const auto inserted = usedDevices.insert(t.get_device()).second;
        if (!inserted) {
            TORCH_CHECK(false, "Tensors must be on distinct NPU devices");
        }
    }
}

// check validity of single tensor
void check_npu_single_tensor(const at::Tensor& tensor)
{
    if (!torch_npu::utils::is_npu(tensor) || tensor.is_sparse()) {
        TORCH_CHECK(false, "Tensors must be NPU and dense");
    }
    if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
        TORCH_CHECK(false, "Tensors must be contiguous");
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
        TORCH_CHECK(false, "Tensor list operands to scatter/gather must have the same length");
    }
    const auto num_devices = tensor_lists.size();

    std::vector<at::Tensor> flattened;
    flattened.resize(num_devices);

    for (auto i = size_t{}; i < num_devices; ++i) {
        if (tensor_lists[i].size() != world_size * num_devices) {
            TORCH_CHECK(
                false,
                "Tensor list input to scatter/gather must match number of collective"
                " participants");
        }

        // Only check device match for the first tensor in the list; the call to
        // newLikeFlat() below will check the rest.
        if (tensor_lists[i].front().get_device() != other[i].get_device()) {
            TORCH_CHECK(
                false,
                "Corresponding input/output tensors to scatter/gather must all reside"
                " on the same device");
        }

        for (const auto& t : tensor_lists[i]) {
            if (t.numel() != other[i].numel()) {
                TORCH_CHECK(false, "All tensor operands to scatter/gather must have the same size");
            }
        }
        // Flatten the tensors (from all ranks) into a single big tensor.
        flattened[i] = c10d::newLikeFlat(tensor_lists, i);
    }
    return flattened;
}

} // namespace

c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL> ProcessGroupHCCL::initWork(
    std::vector<at::Device> devices,
    int rank,
    c10d::OpType opType)
{
    if (devices.size() != 1) {
        throw std::runtime_error("ProcessGroupHCCL support one device per process only");
    }
    return c10::make_intrusive<ProcessGroupHCCL::WorkHCCL>(devices, rank, opType, seq_, desyncDebug_);
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

ProcessGroupHCCL::Options::Options(bool is_high_priority_stream)
    : c10d::Backend::Options(HCCL_BACKEND_NAME),
      opTimeout(kProcessGroupHCCLOpTimeoutMillis),
      is_high_priority_stream(is_high_priority_stream)
{
}

int64_t ProcessGroupHCCL::getHcclComm(int rankid)
{
    at::Device device = getDeviceForRank(rankid);
    std::vector<at::Device> devices = {device};
    const auto key = getKeyFromDevices(devices);
    auto& hcclComms = getHCCLComm(key, devices);
    TORCH_CHECK(hcclComms.size() == 1, "expect hcclComms.size() = 1, but hcclComms.size() = ", hcclComms.size());
    auto ret_hcom = hcclComms[0]->getHcclComm();
    int64_t hccl_comm = static_cast<int64_t>(reinterpret_cast<intptr_t>(ret_hcom));
    return hccl_comm;
}

std::string ProcessGroupHCCL::getHcclCommName(int rankid) {
  at::Device device = getDeviceForRank(rankid);
  std::vector<at::Device> devices = {device};
  const auto key = getKeyFromDevices(devices);
  auto& hcclComms = getHCCLComm(key, devices);
  TORCH_CHECK(hcclComms.size() == 1, "expect hcclComms.size() = 1, but hcclComms.size() = ",
      hcclComms.size());
  HcclComm ret_hcom = hcclComms[0]->getHcclComm();
  char commName[MAX_GROUP_NAME_LEN];
  HCCL_CHECK_ERROR(at_npu::hccl::HcclGetCommNameFace(ret_hcom, commName));
  std::string name_str(commName);
  return name_str;
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

    const auto devices = getDeviceList(inputs);
    const auto key = getKeyFromDevices(devices);
    auto& hcclComms = getHCCLComm(key, devices);
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
        for (const auto i : c10::irange(devices.size())) {
            c10_npu::NPUStream& hcclStream = hcclStreams[i];
            (*(work->hcclStartEvents_))[i].record(hcclStream);
        }
    }

    pre(hcclStreams, work);

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
        c10_npu::NPUCachingAllocator::recordStream(inputs[i].storage().data_ptr(), hcclStream);
        if (c10_npu::option::OptionsManager::IsMultiStreamMemoryReuse()) {
            work->recorded_inputs_.push_back(std::make_pair(inputs[i].storage().getWeakStorageImpl(), hcclStream));
        }
    }
    {
        for (const auto i : c10::irange(inputs.size())) {
            npuGuard.set_index(devices[i].index());
            // to avoid to much task pushed to the stream, leading to stream overflow
            // insert sync point fluxLimit(key, i)
            c10_npu::NPUStream& hcclStream = hcclStreams[i];
            hcclUs startut = TIME_NOW();
            HCCL_CHECK_ERROR(fn(inputs[i], outputs[i], hcclComms[i]->getHcclComm(), hcclStream));
            if (c10_npu::option::OptionsManager::IsMultiStreamMemoryReuse()) {
                work->recorded_outputs_.push_back(
                    std::make_pair(outputs[i].storage().getWeakStorageImpl(), hcclStream));
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
        ASCEND_LOGI("Event: record hccl work is successfully executed.");
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

int g_allreduceID = 0;
c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceOptions& opts)
{
    check_npu_tensors_different_devices(tensors);
    std::vector<at::Tensor> tensors_cp = {tensors[0]};
    return collective(
        tensors_cp,
        tensors_cp,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream) {
            aclrtSetExceptionInfoCallback(exceptionCallback);

            auto hcclType = getHcclDataType(input.scalar_type());
            checkSupportedDataTypeOfAllReduce(hcclType);
            RECORD_FUNCTION("HcclAllreduce", std::vector<c10::IValue>({input}));

            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto hcclReduceOp = getHcclReduceOp(opts.reduceOp, input);
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream]() -> int {
                return HcclAllReduce(
                    inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream.stream(false));
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclAllreduce");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

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

int g_broadcastID = 100000;
c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const c10d::BroadcastOptions& opts)
{
    check_npu_tensors_different_devices(tensors);
    return collective(
        tensors,
        tensors,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream) {
            RECORD_FUNCTION("HcclBroadcast", std::vector<c10::IValue>({input}));
            const auto root = opts.rootRank * tensors.size() + opts.rootTensor;

            auto inputDataPtr = input.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto hcclType = getHcclDataType(input.scalar_type());
            auto hccl_call = [inputDataPtr, numel, hcclType, root, comm, stream]() -> int {
                return HcclBroadcast(inputDataPtr, numel, hcclType, root, comm, stream.stream(false));
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclBroadcast");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

            return HCCL_SUCCESS;
        },
        c10d::OpType::BROADCAST);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const c10d::AllreduceCoalescedOptions& /* unused */)
{
    throw std::runtime_error("ProcessGroupHCCL does not support allreduce_coalesced");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const c10d::ReduceOptions& opts)
{
    check_npu_tensors_different_devices(tensors);
    uint64_t rank = opts.rootRank;
    std::vector<at::Tensor> tensors_cp = {tensors[0]};
    return collective(
        tensors_cp,
        tensors_cp,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream) {
            auto hcclType = getHcclDataType(input.scalar_type());
            checkSupportedDataTypeOfAllReduce(hcclType);
            RECORD_FUNCTION("HcclReduce", std::vector<c10::IValue>({input}));

            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto reduceOp = getHcclReduceOp(opts.reduceOp, input);
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, reduceOp, rank, comm, stream]() -> int {
                return hcclReduce(
                    inputDataPtr, outputDataPtr, numel, hcclType, reduceOp, rank, comm, stream.stream(false));
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclReduce");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

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

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts)
{
    check_npu_tensors_different_devices(inputTensors);
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
            [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream) {
                RECORD_FUNCTION("HcclAllgather", std::vector<c10::IValue>({input}));

                c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
                auto inputDataPtr = input.data_ptr();
                auto outputDataPtr = output.data_ptr();
                auto numel = getNumelForHCCL(input);
                auto hcclType = getHcclDataType(input.scalar_type());
                auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, comm, stream]() -> int {
                    return HcclAllGather(inputDataPtr, outputDataPtr, numel, hcclType, comm, stream.stream(false));
                };
                at_npu::native::OpCommand cmd;
                cmd.Name("HcclAllgather");
                cmd.SetCustomHandler(hccl_call);
                cmd.Run();

                return HCCL_SUCCESS;
            },
            [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
            [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {
                work->lazyDestory(byte_alignment_inputTensors_);
                work->lazyDestory(outputFlattened);
                // Copy the flattened output tensors to the outputs.
                for (const auto i : c10::irange(outputTensors.size())) {
                    c10_npu::NPUStreamGuard guard(hcclStreams[i]);
                    for (const auto j : c10::irange(outputTensors[0].size())) {
                        // See [Sync Streams].
                        c10_npu::NPUCachingAllocator::recordStream(
                            outputTensors[i][j].storage().data_ptr(), hcclStreams[i]);

                        if (c10_npu::option::OptionsManager::IsMultiStreamMemoryReuse()) {
                            work->recorded_outputs_.push_back(
                                std::make_pair(outputTensors[i][j].storage().getWeakStorageImpl(), hcclStreams[i]));
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
                outputs_multi_dev,
                outputs_multi_dev,
                [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream) {
                RECORD_FUNCTION("HcclBroadcast", std::vector<c10::IValue>({input}));
                const auto root = broadcastOpts.rootRank * inputs_multi_dev.size() + broadcastOpts.rootTensor;

                auto inputDataPtr = input.data_ptr();
                auto numel = getNumelForHCCL(input);
                auto hcclType = getHcclDataType(input.scalar_type());
                return HcclBroadcast(inputDataPtr, numel, hcclType, root, comm, stream.stream());
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

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::allgather_togather(
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
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream) {
            RECORD_FUNCTION("HcclAllgatherTogather", std::vector<c10::IValue>({input}));
            c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto hcclType = getHcclDataType(input.scalar_type());
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, comm, stream]() -> int {
                return HcclAllGather(inputDataPtr, outputDataPtr, numel, hcclType, comm, stream.stream(false));
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

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::_allgather_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const c10d::AllgatherOptions& opts)
{
    if (inputTensor.dtype() != outputTensor.dtype()) {
        TORCH_CHECK(false, "output tensor must have the same type as input tensor");
    }
    if (inputTensor.numel() * size_ != outputTensor.numel()) {
        TORCH_CHECK(false, "output tensor size must be equal to world_size times input tensor size");
    }
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    check_npu_tensors_different_devices(inputTensors);
    check_npu_tensors_different_devices(outputTensors);
    auto inputTensors_ = cast_to_origin_format(inputTensors);

    return collective(
        inputTensors_,
        outputTensors,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream) {
            RECORD_FUNCTION("HcclAllgatherBase", std::vector<c10::IValue>({input}));
            c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto hcclType = getHcclDataType(input.scalar_type());
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, comm, stream]() -> int {
                return HcclAllGather(inputDataPtr, outputDataPtr, numel, hcclType, comm, stream.stream(false));
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

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ReduceScatterOptions& opts)
{
    check_npu_tensors_different_devices(outputTensors);

    auto inputFlattened = flatten_for_scatter_gather(inputTensors, outputTensors, size_);
    check_npu_tensors_different_devices(inputFlattened);

    return collective(
        inputFlattened,
        outputTensors,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream) {
            auto hcclType = getHcclDataType(input.scalar_type());
            checkSupportedDataTypeOfAllReduce(hcclType);
            RECORD_FUNCTION("HcclReduceScatter", std::vector<c10::IValue>({input}));
            c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(output);
            auto hcclReduceOp = getHcclReduceOp(opts.reduceOp, input);
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream]() -> int {
                return HcclReduceScatter(
                    inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream.stream(false));
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
                    c10_npu::NPUCachingAllocator::recordStream(inputTensors[i][j].storage().data_ptr(), hcclStreams[i]);

                    if (c10_npu::option::OptionsManager::IsMultiStreamMemoryReuse()) {
                        work->recorded_inputs_.push_back(
                            std::make_pair(inputTensors[i][j].storage().getWeakStorageImpl(), hcclStreams[i]));
                    }

                    inputFlattened[i][j].copy_(inputTensors[i][j], true);
                }
            }
        },
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        c10d::OpType::REDUCE_SCATTER);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const c10d::ReduceScatterOptions& opts)
{
    if (inputTensor.dtype() != outputTensor.dtype()) {
        TORCH_CHECK(false, "input tensor must be the same type as the output tensor.");
    }

    if (inputTensor.numel() != outputTensor.numel() * size_) {
        TORCH_CHECK(false, "input tensor must be the same size as output size times world size");
    }

    auto inputs = std::vector<at::Tensor>{inputTensor};
    auto outputs = std::vector<at::Tensor>{outputTensor};

    return collective(
        inputs,
        outputs,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream) {
            c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
            auto hcclType = getHcclDataType(input.scalar_type());
            checkSupportedDataTypeOfAllReduce(hcclType);
            RECORD_FUNCTION("HcclReduceScatterBase", std::vector<c10::IValue>({input}));
            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(output);
            auto hcclReduceOp = hcclOp[opts.reduceOp];
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream]() -> int {
                return HcclReduceScatter(
                    inputDataPtr, outputDataPtr, numel, hcclType, hcclReduceOp, comm, stream.stream(false));
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
    TORCH_CHECK(hcclWork);
    hcclWork->barrierTensors_ = std::move(barrierTensors);

    return work;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const c10d::GatherOptions& /* unused */)
{
    throw std::runtime_error("ProcessGroupHCCL does not support gather");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ScatterOptions& opts)
{
    check_npu_tensors_different_devices(outputTensors);
    std::vector<at::Tensor> inputFlattened;
    if (getRank() == opts.rootRank) {
        inputFlattened = flatten_for_scatter_gather(inputTensors, outputTensors, size_);
    } else {
        inputFlattened.push_back(at::empty(0).to(outputTensors[0]));
    }

    return collective(
        inputFlattened,
        outputTensors,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream) {
            RECORD_FUNCTION("HcclScatter", std::vector<c10::IValue>({input}));
            const auto root = opts.rootRank;
            c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(output);
            auto hcclType = getHcclDataType(input.scalar_type());
            auto hccl_call = [inputDataPtr, outputDataPtr, numel, hcclType, root, comm, stream]() -> int {
                return hcclScatter(inputDataPtr, outputDataPtr, numel, hcclType, root, comm, stream.stream(false));
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
                    c10_npu::NPUCachingAllocator::recordStream(inputTensors[i][j].storage().data_ptr(), hcclStreams[i]);

                    if (c10_npu::option::OptionsManager::IsMultiStreamMemoryReuse()) {
                        work->recorded_inputs_.push_back(
                            std::make_pair(inputTensors[i][j].storage().getWeakStorageImpl(), hcclStreams[i]));
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
    auto tensors_ = cast_to_origin_format(tensors);
    return collective(
        tensors_,
        tensors_,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream) {
            RECORD_FUNCTION("HcclSend", std::vector<c10::IValue>({input}));
            auto inputDataPtr = input.data_ptr();
            auto numel = getNumelForHCCL(input);
            auto hcclType = getHcclDataType(input.scalar_type());
            auto hccl_call = [inputDataPtr, numel, hcclType, dstRank, comm, stream]() -> int {
                return HcclSend(inputDataPtr, numel, hcclType, dstRank, comm, stream.stream(false));
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclSend");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

            return HCCL_SUCCESS;
        },
        c10d::OpType::SEND);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::recv(std::vector<at::Tensor>& tensors, int srcRank, int tag)
{
    check_npu_tensors_different_devices(tensors);
    auto tensors_ = create_base_format_tensors(tensors);
    return collective(
        tensors,
        tensors_,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream) {
            RECORD_FUNCTION("HcclRecv", std::vector<c10::IValue>({input}));
            c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForHCCL(output);
            auto hcclType = getHcclDataType(output.scalar_type());
            auto hccl_call = [outputDataPtr, numel, hcclType, srcRank, comm, stream]() -> int {
                return HcclRecv(outputDataPtr, numel, hcclType, srcRank, comm, stream.stream(false));
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclRecv");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {
            for (size_t i = 0; i < tensors_.size(); ++i) {
                c10_npu::NPUStreamGuard guard(hcclStreams[i]);
                c10_npu::NPUCachingAllocator::recordStream(tensors_[i].storage().data_ptr(), hcclStreams[i]);
                if (c10_npu::option::OptionsManager::IsMultiStreamMemoryReuse()) {
                    work->recorded_outputs_.push_back(
                        std::make_pair(tensors_[i].storage().getWeakStorageImpl(), hcclStreams[i]));
                }
                if (!at_npu::native::FormatHelper::IsBaseFormatType(tensors[i])) {
                    tensors[i].copy_(tensors_[i], true);
                }
            }
        },
        c10d::OpType::RECV);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupHCCL::recvAnysource(std::vector<at::Tensor>& /* unused */, int /* unused */)
{
    TORCH_CHECK(false, "ProcessGroupHCCL does not support recv");
}

void check_split_sizes(const std::vector<int64_t>& split_sizes, const at::Tensor& tensor, int group_size)
{
    if (split_sizes.empty()) {
        TORCH_CHECK(tensor.size(0) % group_size == 0, "Tensor's dim 0 does not divide equally across group size");
    } else {
        TORCH_CHECK(
            split_sizes.size() == static_cast<size_t>(group_size), "Number of tensor splits not equal to group size");
    }
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
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    auto inputTensors_ = cast_to_origin_format(inputTensors);
    auto outputTensors_ = cast_to_origin_format(outputTensors);
    int ranks = getSize();
    TORCH_CHECK(ranks > 0, "Invalid ranks", ranks);
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
    check_split_sizes(inputSplitSizes, inputTensor, size_);
    check_split_sizes(outputSplitSizes, outputTensor, size_);

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
        inputTensors_,
        outputTensors_,
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream) {
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
                              stream]() -> int {
                return hcclAlltoAllV(
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
            };
            at_npu::native::OpCommand cmd;
            cmd.Name("HcclAlltoAllV");
            cmd.SetCustomHandler(hccl_call);
            cmd.Run();

            return HCCL_SUCCESS;
        },
        [&](std::vector<c10_npu::NPUStream>&, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>&) {},
        [&](std::vector<c10_npu::NPUStream>& hcclStreams, c10::intrusive_ptr<ProcessGroupHCCL::WorkHCCL>& work) {
            for (size_t i = 0; i < outputTensors_.size(); ++i) {
                c10_npu::NPUStreamGuard guard(hcclStreams[i]);
                c10_npu::NPUCachingAllocator::recordStream(outputTensors_[i].storage().data_ptr(), hcclStreams[i]);
                if (!at_npu::native::FormatHelper::IsBaseFormatType(outputTensors[i])) {
                    outputTensors[i].copy_(outputTensors_[i], true);
                }
            }
        },
        c10d::OpType::ALLTOALL);
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
        TORCH_CHECK(
            device == output_tensors[r].device() && device == input_tensors[r].device(),
            "tensors must be on the same device");
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
        [&](at::Tensor& input, at::Tensor& output, HcclComm comm, c10_npu::NPUStream& stream) {
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
                              stream]() -> int {
                return hcclAlltoAllV(
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
            c10_npu::NPUCachingAllocator::recordStream(output_tensors_[0].storage().data_ptr(), hcclStreams[0]);
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
