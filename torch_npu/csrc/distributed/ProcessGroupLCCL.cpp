#include "ProcessGroupLCCL.hpp"

#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/OpCommand.h"


namespace c10d_npu {

namespace {
constexpr int64_t kSynchronizeBusyWaitMillis = 10;

void syncStreams(const std::vector<at::Device> &devices, std::vector<c10_npu::NPUEvent> &lcclEvents,
                 std::vector<c10_npu::NPUStream> &lcclStreams)
{
    for (size_t i = 0; i < devices.size(); ++i) {
        c10_npu::NPUStream &lcclStream = lcclStreams[i];
        c10_npu::NPUEvent &lcclEvent = lcclEvents[i];
        lcclEvent.record(c10_npu::getCurrentNPUStream(devices[i].index()));
        ASCEND_LOGI("Event: record lccl group is successfully executed, event=%p", lcclEvent.event());
        lcclEvent.block(lcclStream);
        ASCEND_LOGI("Event: block lccl group is successfully executed, event=%p", lcclEvent.event());
    }
}
} // namespace

ProcessGroupLCCL::WorkLCCL::WorkLCCL(const std::vector<at::Device> &devices)
    : devices_(devices), workStartTime_(std::chrono::steady_clock::now())
{
    // Creates the npu event wrappers
    // Note: The actual events are lazily created when first recorded to with
    // DEFAULT_FLAGS = npuEventDisableTiming.
    lcclEndEvents_ = std::make_shared<std::vector<c10_npu::NPUEvent>>(devices.size());
    lcclComms_.resize(devices.size());
}

ProcessGroupLCCL::WorkLCCL::~WorkLCCL()
{}

bool ProcessGroupLCCL::WorkLCCL::isCompleted()
{
    checkAndSetException();
    return exception() || finishedNPUExecutionInternal();
}

bool ProcessGroupLCCL::WorkLCCL::isSuccess() const
{
    if (exception()) {
        // Already detected an exception.
        return false;
    }
    return finishedNPUExecutionInternal();
}

void ProcessGroupLCCL::WorkLCCL::synchronizeInternal(std::chrono::milliseconds timeout)
{
    for (const auto i: c10::irange(devices_.size())) {
        auto currentStream = c10_npu::getCurrentNPUStream(devices_[i].index());
        // Block the current stream on the LCCL stream
        (*lcclEndEvents_)[i].block(currentStream);
        ASCEND_LOGI("Event: block lccl work is successfully executed, event=%p", (*lcclEndEvents_)[i].event());
    }

    // In case of blocking, wait for the operation to complete.
    if (blockingWait_) {
        // Wait for the operation to complete.
        while (!isCompleted()) {
            auto currentTimepoint = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(currentTimepoint - workStartTime_) > opTimeout_) {
                throw std::runtime_error("Operation has exceeded timeout limit!");
            }
            checkAndThrowException();
            std::this_thread::sleep_for(std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
        }
        checkAndThrowException();
    }
}

// Same as calling synchronize().
bool ProcessGroupLCCL::WorkLCCL::wait(std::chrono::milliseconds timeout)
{
    synchronizeInternal(timeout);
    // Always return true, because abort API is not implemented.
    return true;
}

void ProcessGroupLCCL::WorkLCCL::synchronize()
{
    // Call Synchronize without a timeout. We use this method to avoid adding a
    // timeout argument to the public synchronize API.
    synchronizeInternal(kNoTimeout);
}

bool ProcessGroupLCCL::WorkLCCL::finishedNPUExecution()
{
    checkAndSetException();
    return finishedNPUExecutionInternal();
}

std::vector<at::Tensor> ProcessGroupLCCL::WorkLCCL::result()
{
    return *outputs_;
}

void ProcessGroupLCCL::WorkLCCL::checkAndThrowException() const
{
    // Set the appropriate exception if found.
    checkAndSetException();

    // Throw an exception, only if we have a valid exception.
    if (exception()) {
        std::rethrow_exception(exception());
    }
}

void ProcessGroupLCCL::WorkLCCL::checkAndSetException() const
{
    if (exception()) {
        // We already have an exception.
        return;
    }
}

// check if LCCL task is finished
bool ProcessGroupLCCL::WorkLCCL::finishedNPUExecutionInternal() const
{
    // If in the Finalize, should not query event
    if (!c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
        return false;
    }
    try {
        for (const auto i: c10::irange(devices_.size())) {
            // Checking the work's corresponding ASCEND events' status
            if (!(*lcclEndEvents_)[i].query()) {
                return false;
            }
        }
    } catch (const std::exception &e) {
        if (std::string(e.what()).find("driver shutting down") == std::string::npos) {
            throw std::runtime_error(DIST_ERROR(ErrCode::INTERNAL));
        }
        LOG(INFO) << "[Rank " << rank_ << "] Event query failed with exception: " << e.what();
    }

    return true;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupLCCL::WorkLCCL::getFuture()
{
    return future_;
}

const int64_t ProcessGroupLCCL::kProcessGroupLCCLOpTimeoutMillis = 10 * 1000;

ProcessGroupLCCL::ProcessGroupLCCL(const c10::intrusive_ptr<c10d::Store> &store, int rank, int size)
    : c10d::Backend(rank, size), blockingWait_(false), store_(store),
      opTimeout_(ProcessGroupLCCL::kProcessGroupLCCLOpTimeoutMillis)
{}

std::vector<at_npu::lccl::LcclComm> &ProcessGroupLCCL::getLCCLComm(
    const std::string &devicesKey,
    const std::vector<at::Device> &devices)
{
    // Sanity check
    if (devicesKey.empty()) {
        throw std::runtime_error("Not able to create/get the lccll Communicator since "
                                 "the NPU devices are not known" +
                                 DIST_ERROR(ErrCode::PARAM));
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (devLCCLCommMap_.find(devicesKey) != devLCCLCommMap_.end()) {
            // Reuse the cached communicator if there is one.
            return devLCCLCommMap_[devicesKey];
        }
    }

    std::vector<at_npu::lccl::LcclComm> lcclComms;
    lcclComms.resize(devices.size());

    c10_npu::OptionalNPUGuard npuGuard;
    std::vector<c10_npu::NPUStream> streamVal;
    streamVal.reserve(devices.size());

    for (size_t i = 0; i < devices.size(); ++i) {
        npuGuard.set_index(devices[i].index());
        auto ret = at_npu::lccl::LcclCommInitRankLocal(size_, rank_, &lcclComms[i]);
        TORCH_CHECK(ret == 0, "init lccl comm failed, error code:", ret, PTA_ERROR(ErrCode::INTERNAL));

        // Creates the LCCL streams
        streamVal.push_back(c10_npu::getNPUStreamFromPool(devices[i].index()));
    }

    lcclStreams_.emplace(devicesKey, std::move(streamVal));

    // Note: these events are created with the (default) cudaEventDisableTiming
    // flag This flag provides the best performance when used with
    // StreamWaitEvent() and EventQuery(). Since we here don't measure the
    // performance using npuEvent, this should be set.
    lcclEvents_.emplace(std::piecewise_construct, std::make_tuple(devicesKey), std::make_tuple(devices.size()));

    // Hold the lock before modifying the cache.
    std::lock_guard<std::mutex> lock(mutex_);
    devLCCLCommMap_.emplace(devicesKey, std::move(lcclComms));
    return devLCCLCommMap_[devicesKey];
}

template<typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<c10d::Work> ProcessGroupLCCL::collective(std::vector<at::Tensor> &inputs,
                                                            std::vector<at::Tensor> &outputs, Fn fn, PreProcess pre,
                                                            PostProcess post, c10d::OpType opType)
{
    const auto devices = getDeviceList(inputs);
    auto key = getKeyFromDevices(devices);
    std::vector<at_npu::lccl::LcclComm> lcclComms;
    lcclComms = getLCCLComm(key, devices);
    // Used many times below, so we stash the unordered_map lookup
    auto &lcclStreams = lcclStreams_[key];
    // First let LCCL streams wait for input tensors allocation streams
    syncStreams(devices, lcclEvents_[key], lcclStreams);
    // Work itself will create the events on all NPUs of tensors
    auto work = c10::make_intrusive<ProcessGroupLCCL::WorkLCCL>(devices);
    // Store references to outputs to be used by WorkLCCL::result and operator<<.
    work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

    c10_npu::OptionalNPUGuard npuGuard;
    pre(lcclStreams, work);

    for (const auto i: c10::irange(inputs.size())) {
        npuGuard.set_index(devices[i].index());
        c10_npu::NPUStream &lcclStream = lcclStreams[i];

        // Both `inputs' and `outputs' are created on a worker stream and used in
        // different lcclStreams.  Hence, both must record the lcclStream to
        // prevent being freed before the collective finishes.
        //
        // We only record `inputs' here, and leave recording `outputs' to `fn' for
        // operations where `inputs' and `outputs' are not the same.
        //
        // See [Sync Streams].
        c10_npu::NPUCachingAllocator::recordStream(inputs[i].storage().data_ptr(), lcclStream);
    }
    {
        for (const auto i: c10::irange(inputs.size())) {
            npuGuard.set_index(devices[i].index());
            // to avoid to much task pushed to the stream, leading to stream overflow
            // insert sync point fluxLimit(key, i)

            c10_npu::NPUStream &lcclStream = lcclStreams[i];
            auto ret = fn(inputs[i], outputs[i], lcclComms[i], lcclStream);
            TORCH_CHECK(ret == 0, "LCCL function error:", opTypeToString(opType).c_str(), ", error code is", ret, "\n");
        }
    }
    post(lcclStreams, work);
    {
        c10_npu::NPUMultiStreamGuard guard(lcclStreams);
        work->future_ = c10::make_intrusive<at::ivalue::Future>(c10::ListType::create(c10::TensorType::get()), devices);
        work->future_->markCompleted(at::IValue(*work->outputs_));
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        c10_npu::NPUStream &lcclStream = lcclStreams_[key][i];
        (*(work->lcclEndEvents_))[i].record(lcclStream);
        ASCEND_LOGI("Event: record lccl work is successfully executed, event=%p", (*(work->lcclEndEvents_))[i].event());
        work->lcclComms_[i] = lcclComms[i];
    }
    work->blockingWait_ = blockingWait_;
    work->opTimeout_ = opTimeout_;
    return work;
}


template<typename Fn>
c10::intrusive_ptr<c10d::Work> ProcessGroupLCCL::collective(std::vector<at::Tensor> &inputs,
                                                            std::vector<at::Tensor> &outputs, Fn fn,
                                                            c10d::OpType opType)
{
    return collective(
        inputs, outputs, fn,
        [](std::vector<c10_npu::NPUStream> &, c10::intrusive_ptr<ProcessGroupLCCL::WorkLCCL> &) {
        },
        [](std::vector<c10_npu::NPUStream> &, c10::intrusive_ptr<ProcessGroupLCCL::WorkLCCL> &) {
        },
        opType);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupLCCL::allreduce(std::vector<at::Tensor> &tensors,
                                                           const c10d::AllreduceOptions &opts)
{
    checkTensors(tensors);
    std::vector<at::Tensor> tensors_cp = {tensors[0]};
    std::string functionName = __FUNCTION__;
    return collective(
        tensors_cp, tensors_cp,
        [&](at::Tensor &input, at::Tensor &output, at_npu::lccl::LcclComm comm, c10_npu::NPUStream &stream) {
            auto lcclType = getLcclDataType(input.scalar_type());
            checkSupportedDataType(lcclType, functionName);
            RECORD_FUNCTION("LcclAllreduce", std::vector<c10::IValue>({input}));

            auto inputDataPtr = input.data_ptr();
            auto outputDataPtr = output.data_ptr();
            auto numel = getNumelForLCCL(input);
            auto lcclReduceOp = getLcclReduceOp(opts.reduceOp, input);
            auto lccl_call = [inputDataPtr, outputDataPtr, numel, lcclType, lcclReduceOp, stream, comm]() -> int {
                auto lccl_result = at_npu::lccl::LcclAllReduce(inputDataPtr, outputDataPtr, numel, lcclType,
                                                               lcclReduceOp, comm, stream.stream(false));
                return lccl_result;
            };
            at_npu::native::OpCommand::RunOpApiV2("LcclAllreduce", lccl_call);
            return 0;
        },
        c10d::OpType::ALLREDUCE);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupLCCL::allgather(std::vector<std::vector<at::Tensor>> &outputTensors,
                                                           std::vector<at::Tensor> &inputTensors,
                                                           const c10d::AllgatherOptions &opts)
{
    checkTensors(inputTensors);

    auto inputTensors_ = castOriginFormat(inputTensors);
    bool same_size = CheckTensorsSameSize(outputTensors.back());
    if (same_size) {
        auto outputFlattened = FlattenForScatterGather(outputTensors, inputTensors, size_);
        checkTensors(outputFlattened);

        return collective(
            inputTensors_, outputFlattened,
            [&](at::Tensor &input, at::Tensor &output, at_npu::lccl::LcclComm comm, c10_npu::NPUStream &stream) {
                RECORD_FUNCTION("LcclAllgather", std::vector<c10::IValue>({input}));
                c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
                auto inputDataPtr = input.data_ptr();
                auto outputDataPtr = output.data_ptr();
                auto numel = getNumelForLCCL(input);
                auto lcclType = getLcclDataType(input.scalar_type());
                auto lccl_call = [inputDataPtr, outputDataPtr, numel, lcclType, comm, stream]() -> int {
                    auto lccl_result = at_npu::lccl::LcclAllGather(inputDataPtr, outputDataPtr, numel, lcclType, comm,
                                                                   stream.stream(false));
                    return lccl_result;
                };
                at_npu::native::OpCommand::RunOpApiV2("LcclAllgather", lccl_call);
                return 0;
            },
            [&](std::vector<c10_npu::NPUStream> &, c10::intrusive_ptr<ProcessGroupLCCL::WorkLCCL> &) {
            },
            [&](std::vector<c10_npu::NPUStream> &lcclStreams, c10::intrusive_ptr<ProcessGroupLCCL::WorkLCCL> &work) {
                // Copy the flattened output tensors to the outputs.

                for (const auto i: c10::irange(outputTensors.size())) {
                    c10_npu::NPUStreamGuard guard(lcclStreams[i]);
                    for (const auto j: c10::irange(outputTensors[0].size())) {
                        // See [Sync Streams].
                        c10_npu::NPUCachingAllocator::recordStream(outputTensors[i][j].storage().data_ptr(),
                                                                   lcclStreams[i]);

                        outputTensors[i][j].copy_(outputFlattened[i][j], true);
                    }
                }
            },
            c10d::OpType::ALLGATHER);
    } else {
        TORCH_CHECK(false, "lccl doesn't support to all_gather different shape");
    }
}

c10::intrusive_ptr<c10d::Work> ProcessGroupLCCL::broadcast(std::vector<at::Tensor> &tensors,
                                                           const c10d::BroadcastOptions &opts)
{
    checkTensors(tensors);

    return collective(
        tensors, tensors,
        [&](at::Tensor &input, at::Tensor &output, at_npu::lccl::LcclComm comm, c10_npu::NPUStream &stream) {
            RECORD_FUNCTION("LcclBroadcast", std::vector<c10::IValue>({input}));
            const auto root = opts.rootRank * tensors.size() + opts.rootTensor;

            auto inputDataPtr = input.data_ptr();
            auto numel = getNumelForLCCL(input);
            auto lcclType = getLcclDataType(input.scalar_type());
            auto lccl_call = [inputDataPtr, numel, lcclType, root, comm, stream]() -> int {
                auto lccl_result =
                    at_npu::lccl::LcclBroadcast(inputDataPtr, numel, lcclType, root, comm, stream.stream(false));
                return lccl_result;
            };
            at_npu::native::OpCommand::RunOpApiV2("LcclBroadcast", lccl_call);
            return 0;
        },
        c10d::OpType::BROADCAST);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupLCCL::reduce_scatter(std::vector<at::Tensor> &outputTensors,
                                                                std::vector<std::vector<at::Tensor>> &inputTensors,
                                                                const c10d::ReduceScatterOptions &opts)
{
    checkTensors(outputTensors);

    bool same_size = CheckTensorsSameSize(inputTensors.back());
    if (same_size) {
        auto inputFlattened = FlattenForScatterGather(inputTensors, outputTensors, size_);
        checkTensors(inputFlattened);
        std::string functionName = __FUNCTION__;
        return collective(
            inputFlattened, outputTensors,
            [&](at::Tensor &input, at::Tensor &output, at_npu::lccl::LcclComm comm, c10_npu::NPUStream &stream) {
                auto lcclType = getLcclDataType(input.scalar_type());
                checkSupportedDataType(lcclType, functionName);
                RECORD_FUNCTION("LcclReduceScatter", std::vector<c10::IValue>({input}));
                c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
                auto inputDataPtr = input.data_ptr();
                auto outputDataPtr = output.data_ptr();
                auto numel = getNumelForLCCL(output);
                auto lcclReduceOp = getLcclReduceOp(opts.reduceOp, input);
                auto lccl_call = [inputDataPtr, outputDataPtr, numel, lcclType, lcclReduceOp, stream, comm]() -> int {
                    auto lccl_result = at_npu::lccl::LcclReduceScatter(inputDataPtr, outputDataPtr, numel, lcclType,
                                                                       lcclReduceOp, comm, stream.stream(false));
                    return lccl_result;
                };
                at_npu::native::OpCommand::RunOpApiV2("LcclReduceScatter", lccl_call);
                return 0;
            },
            [&](std::vector<c10_npu::NPUStream> &lcclStreams, c10::intrusive_ptr<ProcessGroupLCCL::WorkLCCL> &work) {
                // Copy the input tensors to the flattened inputs.
                for (const auto i: c10::irange(inputTensors.size())) {
                    c10_npu::NPUStreamGuard guard(lcclStreams[i]);
                    for (const auto j: c10::irange(inputTensors[0].size())) {
                        // See [Sync Streams].
                        c10_npu::NPUCachingAllocator::recordStream(inputTensors[i][j].storage().data_ptr(),
                                                                   lcclStreams[i]);
                        inputFlattened[i][j].copy_(inputTensors[i][j], true);
                    }
                }
            },
            [&](std::vector<c10_npu::NPUStream> &, c10::intrusive_ptr<ProcessGroupLCCL::WorkLCCL> &) {
            },
            c10d::OpType::REDUCE_SCATTER);
    } else {
        TORCH_CHECK(false, "lccl doesn't support to reduce_scatter different shape");
    }
}

ProcessGroupLCCL::~ProcessGroupLCCL()
{
    {
        // Destroy all LCCL Communicators on Process Group Destruction
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &it: devLCCLCommMap_) {
            auto &lcclComms = it.second;

            for (const auto &lcclComm: lcclComms) {
                at_npu::lccl::LcclCommDestroy(lcclComm);
            }
        }
    }
}

} // namespace c10d_npu
