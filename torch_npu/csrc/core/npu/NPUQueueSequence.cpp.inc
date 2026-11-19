#include "torch_npu/csrc/core/npu/NPUQueue.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NPUAffinityController.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/framework/OpParamMaker.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"
#include "torch_npu/csrc/logging/LogContext.h"
#include "third_party/op-plugin/op_plugin/ops/dvm/lazy_fusion_kernel.h"

#ifndef BUILD_LIBTORCH
#include <Python.h>
#endif

#include <ATen/record_function.h>
#include <atomic>
#include <optional>
#include <sstream>
#include <acl/acl_rt.h>

namespace c10_npu {

void StartConsume(Repository* repo, c10::DeviceIndex device_id);

namespace {
class CallBackManager {
public:
    CallBackManager() {}
    ~CallBackManager() {}
    void SetExec(const ACL_EXEC_FUNC &func)
    {
        this->execFunc = func;
    }

    void SetCopy(const ACL_COPY_FUNC &func)
    {
        this->copyFunc = func;
    }

    void SetRelease(const ACL_RELEASE_FUNC &func)
    {
        this->releaseFunc = func;
    }

    void SetCopyReleaseParam(const ACL_COPY_RELEASE_PARM_FUNC &func)
    {
        this->copyReleaseParamFunc = func;
    }

    void SetReleaseParam(const ACL_RELEASE_PARAM_FUNC &func)
    {
        this->releaseParamFunc = func;
    }

    void SetNew(const ACL_NEW_FUNC &func)
    {
        this->newFunc = func;
    }

    void SetDelete(const ACL_DELETE_FUNC &func)
    {
        this->deleteFunc = func;
    }

    void *getCurrentParams(void *head, int offset)
    {
        return (uint8_t *)head + ParameterSize() * offset;
    }

    int Call(void *head, int offset)
    {
        TORCH_CHECK(this->execFunc, "Failed to find execution function.", PTA_ERROR(ErrCode::NOT_FOUND));
        auto dstPtr = (uint8_t *)head + ParameterSize() * offset;
        return this->execFunc(dstPtr);
    }

    void Copy(void *dstHead, int offset, void *src)
    {
        TORCH_CHECK(this->copyFunc, "Failed to find copy function.", PTA_ERROR(ErrCode::NOT_FOUND));
        auto dstPtr = (uint8_t *)dstHead + ParameterSize() * offset;
        return this->copyFunc(dstPtr, src);
    }

    void Release(void *head, int offset, ReleaseQueue &releaseQueue)
    {
        TORCH_CHECK(this->releaseFunc, "Failed to find release function.", PTA_ERROR(ErrCode::NOT_FOUND));
        auto ptr = (uint8_t *)head + ParameterSize() * offset;
        return this->releaseFunc(ptr, releaseQueue);
    }

    void CopyRealseParam(void *dstHead, int offset, void *src)
    {
        TORCH_CHECK(this->copyReleaseParamFunc, "Failed to find copy release params function.",
            PTA_ERROR(ErrCode::NOT_FOUND));
        auto dstPtr = (uint8_t *)dstHead + ParameterSize() * offset;
        return this->copyReleaseParamFunc(dstPtr, src);
    }

    void ReleaseParam(void *head, int offset)
    {
        TORCH_CHECK(this->releaseParamFunc, "Failed to find release params function.", PTA_ERROR(ErrCode::NOT_FOUND));
        auto ptr = (uint8_t *)head + ParameterSize() * offset;
        return this->releaseParamFunc(ptr);
    }

    void *Init(int capacity)
    {
        TORCH_CHECK(this->newFunc, "Failed to find new function.", PTA_ERROR(ErrCode::NOT_FOUND));
        int discovered_size = 0;
        void *ptr = this->newFunc(capacity, discovered_size); // not check as CUDA
        if (discovered_size <= 0) {
            if (ptr != nullptr && this->deleteFunc != nullptr) {
                this->deleteFunc(ptr);
            }
            TORCH_CHECK(
                false,
                "Queue callback returned an invalid parameter size.",
                PTA_ERROR(ErrCode::INTERNAL));
        }
        int expected_size = 0;
        if (!sizePerParams.compare_exchange_strong(
                expected_size,
                discovered_size,
                std::memory_order_acq_rel,
                std::memory_order_acquire) &&
            expected_size != discovered_size) {
            if (ptr != nullptr && this->deleteFunc != nullptr) {
                this->deleteFunc(ptr);
            }
            TORCH_CHECK(
                false,
                "Queue callback parameter size changed across repository initialization.",
                PTA_ERROR(ErrCode::INTERNAL));
        }
        return ptr;
    }

    void DeInit(void *ptr)
    {
        if (ptr != nullptr) {
            TORCH_CHECK(this->deleteFunc, "Failed to find delete function.", PTA_ERROR(ErrCode::NOT_FOUND));
            this->deleteFunc(ptr);
            ptr = nullptr;
        }
    }

private:
    int ParameterSize() const
    {
        const int size = sizePerParams.load(std::memory_order_acquire);
        TORCH_CHECK(size > 0, "Queue callback parameter size is not initialized.", PTA_ERROR(ErrCode::INTERNAL));
        return size;
    }

    std::atomic<int> sizePerParams{0};
    ACL_EXEC_FUNC execFunc = nullptr;
    ACL_COPY_FUNC copyFunc = nullptr;
    ACL_RELEASE_FUNC releaseFunc = nullptr;
    ACL_NEW_FUNC newFunc = nullptr;
    ACL_DELETE_FUNC deleteFunc = nullptr;
    ACL_COPY_RELEASE_PARM_FUNC copyReleaseParamFunc = nullptr;
    ACL_RELEASE_PARAM_FUNC releaseParamFunc = nullptr;
}; // class CallBackManager

CallBackManager &manager()
{
    static CallBackManager instance;
    return instance;
}

CallBackManager &releaseManager()
{
    static CallBackManager releaseinstance;
    return releaseinstance;
}

#ifndef BUILD_LIBTORCH
class QueuePythonThreadState final {
public:
    QueuePythonThreadState() noexcept
    {
        // Queue actors must stop and join while Python is still alive. Do not
        // acquire a new detached state from late global C++ teardown.
        if (!Py_IsInitialized()) {
            return;
        }
#if PY_VERSION_HEX >= 0x030D0000
        if (Py_IsFinalizing()) {
            return;
        }
#endif
        if (PyGILState_Check()) {
            state_ = PyEval_SaveThread();
        }
    }
    ~QueuePythonThreadState()
    {
        if (state_ != nullptr) {
            PyEval_RestoreThread(state_);
        }
    }
    QueuePythonThreadState(const QueuePythonThreadState&) = delete;
    QueuePythonThreadState& operator=(const QueuePythonThreadState&) = delete;
private:
    PyThreadState* state_ = nullptr;
};
#endif

class QueueWaitHooks {
public:
    explicit QueueWaitHooks(bool detach_python) : detach_python_(detach_python) {}
    void BeforeWait()
    {
#ifndef BUILD_LIBTORCH
        if (detach_python_ && !python_state_) {
            python_state_.emplace();
        }
#endif
    }
    void AfterWait()
    {
#ifndef BUILD_LIBTORCH
        python_state_.reset();
#endif
    }
private:
    bool detach_python_ = false;
#ifndef BUILD_LIBTORCH
    std::optional<QueuePythonThreadState> python_state_;
#endif
};

void JoinQueueActor(std::thread& actor, QueueWaitHooks& wait_hooks)
{
    if (!actor.joinable()) {
        return;
    }
    TORCH_CHECK(actor.get_id() != std::this_thread::get_id(),
        "Task queue actor cannot join itself.", PTA_ERROR(ErrCode::INTERNAL));
    wait_hooks.BeforeWait();
    try {
        actor.join();
    } catch (...) {
        wait_hooks.AfterWait();
        throw;
    }
    wait_hooks.AfterWait();
}

} // namespace

namespace register_queue_cb {
NPUCallBackRegisterBuilder::NPUCallBackRegisterBuilder(const ACL_EXEC_FUNC &execFunc, const ACL_COPY_FUNC &copyFunc,
    const ACL_RELEASE_FUNC &releaseFunc, const ACL_NEW_FUNC &newFunc, const ACL_DELETE_FUNC &deleteFunc,
    const ACL_COPY_RELEASE_PARM_FUNC &copyReleaseParamF, const ACL_RELEASE_PARAM_FUNC &releaseParamF)
{
    manager().SetExec(execFunc);
    manager().SetCopy(copyFunc);
    manager().SetRelease(releaseFunc);
    manager().SetNew(newFunc);
    manager().SetDelete(deleteFunc);
    releaseManager().SetCopyReleaseParam(copyReleaseParamF);
    releaseManager().SetReleaseParam(releaseParamF);
    releaseManager().SetNew(newFunc);
    releaseManager().SetDelete(deleteFunc);
}
} // namespace register_queue_cb


// If the capacity is too large, when the queue is full,
// a large amount of device memory is occupied at the same time;
// if the capacity is too small, and the main thread is fast enough,
// it does not make full use of concurrent design capabilities.
std::unordered_map<RepoStatus, std::string> deviceErrorMap = {
    {RepoStatus::UCE_EXIT, "UCE ERROR"},
    {RepoStatus::HBM_ECC_EXIT, "HBM MULTI BIT ECC ERROR"},
    {RepoStatus::STOP_EXIT, "FORCE STOP"},
    {RepoStatus::SUSPECT_MEM_EXIT, "SUSPECT MEM ERROR"},
    {RepoStatus::HCCS_LINK_EXIT, "HCCS LINK ERROR"},
    {RepoStatus::HCCL_OP_RETRY_EXIT, "HCCL OP RETRY FAILED"},
    {RepoStatus::SUSPECT_REMOTE_EXIT, "SUSPECT REMOTE ERROR"}
};

std::string get_func_error_msg(void *error_paras)
{
    auto queueParam = static_cast<c10_npu::queue::QueueParas *>(error_paras);
    auto type = queueParam->paramType;
    std::stringstream result;
    if (type == c10_npu::queue::EXECUTE_OPAPI) {
        auto cur_paras = static_cast<at_npu::native::ExecuteParasOpApi *>(queueParam->paramVal);
        auto op_name = cur_paras->opType;
        result << "the current working operator name is " << op_name;
    } else if (type == c10_npu::queue::EXECUTE_OPAPI_V2) {
        auto cur_paras = static_cast<at_npu::native::ExecuteParasOpApiV2 *>(queueParam->paramVal);
        auto op_name = cur_paras->opName;
        result << "the current working operator name is " << *op_name;
    } else if (type == c10_npu::queue::COMPILE_AND_EXECUTE) {
        auto cur_paras = static_cast<at_npu::native::ExecuteParas *>(queueParam->paramVal);
        auto op_name = cur_paras->opType;
        // Warning: key logs in the fault mode library!!! Don't make arbitrary modifications!!!
        result << "the current working operator name is " << op_name;
    } else if (type == c10_npu::queue::ASYNC_MEMCPY) {
        auto cur_paras = static_cast<c10_npu::queue::CopyParas *>(queueParam->paramVal);
        result << "the current copy params are srclen=" << cur_paras->srcLen << ", dstlen=" << cur_paras->dstLen <<
            ", kind=" << cur_paras->kind;
    } else {
        auto cur_paras = static_cast<c10_npu::queue::EventParas *>(queueParam->paramVal);
        result << "the current working event is " << cur_paras->event;
    }
    return result.str();
}

ErrorRecord Repository::SnapshotError() const
{
    std::lock_guard<std::mutex> lock(error_mu_);
    return error_;
}

void Repository::RecordRepositoryError(std::string message)
{
    std::lock_guard<std::mutex> lock(error_mu_);
    error_.repository = std::move(message);
}

void Repository::RecordAclError(std::string message)
{
    std::lock_guard<std::mutex> lock(error_mu_);
    error_.acl = std::move(message);
}

void Repository::RecordCurrentParameter(std::string message)
{
    std::lock_guard<std::mutex> lock(error_mu_);
    error_.current_parameter = std::move(message);
}

RepoStatus Repository::GetStatus() const
{
    const Lifecycle lifecycle = lifecycle_.load(std::memory_order_acquire);
    if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
        ASCEND_LOGE("Task queue is not initialized, shouldn't call GetStatus(). !!");
    }

    return repo_status.load();
}

void Repository::SetStatus(RepoStatus desired)
{
    if (desired != RepoStatus::RUN && desired != RepoStatus::INIT) {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
        const Lifecycle lifecycle = lifecycle_.load(std::memory_order_acquire);
        if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
            ASCEND_LOGE("Task queue is not initialized, shouldn't call SetStatus(). !!");
            return;
        }
        repo_status.store(desired, std::memory_order_release);
        ++status_epoch_;
        queue_core_.RequestStop();
        return;
    }
#ifndef BUILD_LIBTORCH
    QueueWaitHooks wait_hooks(g_used_aclop);
#else
    QueueWaitHooks wait_hooks(false);
#endif
    RepoStatus previous_status = RepoStatus::INIT;
    uint64_t previous_epoch = 0;
    {
        std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mu_);
        while (lifecycle_.load(std::memory_order_acquire) == Lifecycle::Initializing) {
            wait_hooks.BeforeWait();
            try {
                lifecycle_cv_.wait(lifecycle_lock);
            } catch (...) {
                if (lifecycle_lock.owns_lock()) {
                    lifecycle_lock.unlock();
                }
                wait_hooks.AfterWait();
                throw;
            }
            lifecycle_lock.unlock();
            wait_hooks.AfterWait();
            lifecycle_lock.lock();
        }
        TORCH_CHECK(lifecycle_.load(std::memory_order_acquire) == Lifecycle::Running,
            "Task queue cannot activate from its current lifecycle state.", PTA_ERROR(ErrCode::INTERNAL));
        if (queue_core_.Snapshot().mode == queue::QueueMode::Running) {
            repo_status.store(desired, std::memory_order_release);
            ++status_epoch_;
            return;
        }
        TORCH_CHECK(!consumer.joinable() || consumer.get_id() != std::this_thread::get_id(),
            "Task queue cannot restart from its consumer callback.", PTA_ERROR(ErrCode::INTERNAL));
        previous_status = repo_status.load(std::memory_order_acquire);
        previous_epoch = status_epoch_;
        lifecycle_.store(Lifecycle::Initializing, std::memory_order_release);
    }
    try {
        // Stop the old actor before resetting sequence positions. Public status
        // remains unchanged so a newer STOP/error can still preempt this restart.
        consumer_stop_requested_.store(true, std::memory_order_release);
        queue_core_.RequestStop();
        JoinQueueActor(consumer, wait_hooks);
        if (queue_core_.Snapshot().failed) {
            queue_core_.StopAndDiscardMetadata(wait_hooks);
        } else {
            queue_core_.StopAndDiscard(
                [this](uint32_t slot) { manager().Release(data, slot, releaseQueue); }, wait_hooks);
        }
        // No main consumer can now publish another release parameter.
        releaseQueue.Shutdown();
        releaseQueue.InitReleaseQueue(device_idx);
        {
            std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
            if (status_epoch_ == previous_epoch &&
                repo_status.load(std::memory_order_acquire) == previous_status) {
                queue_core_.Start();
                repo_status.store(desired, std::memory_order_release);
                ++status_epoch_;
                consumer_stop_requested_.store(false, std::memory_order_release);
                std::thread next_consumer(StartConsume, this, device_idx);
                consumer = std::move(next_consumer);
            }
            lifecycle_.store(Lifecycle::Running, std::memory_order_release);
        }
        lifecycle_cv_.notify_all();
    } catch (...) {
        const std::exception_ptr recovery_error = std::current_exception();
        consumer_stop_requested_.store(true, std::memory_order_release);
        queue_core_.RequestStop();
        try {
            queue_core_.StopAndDiscardMetadata(wait_hooks);
        } catch (...) {
            ASCEND_LOGE("Task queue recovery metadata cleanup failed.");
        }
        try {
            std::rethrow_exception(recovery_error);
        } catch (const std::exception& error) {
            try { RecordRepositoryError(std::string("Task queue recovery failed: ") + error.what()); }
            catch (...) { ASCEND_LOGE("Task queue recovery error recording failed."); }
        } catch (...) {
            try { RecordRepositoryError("Task queue recovery failed with an unknown exception."); }
            catch (...) { ASCEND_LOGE("Task queue recovery error recording failed."); }
        }
        {
            std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
            repo_status.store(RepoStatus::ERROR_EXIT, std::memory_order_release);
            ++status_epoch_;
            lifecycle_.store(Lifecycle::Running, std::memory_order_release);
        }
        lifecycle_cv_.notify_all();
        std::rethrow_exception(recovery_error);
    }
}

void Repository::ChangeStatus(RepoStatus expected, RepoStatus desired)
{
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
    const Lifecycle lifecycle = lifecycle_.load(std::memory_order_acquire);
    if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
        ASCEND_LOGE("Task queue is not initialized, shouldn't call ChangeStatus(). !!");
        return;
    }
    if (repo_status.load(std::memory_order_acquire) != expected) {
        return;
    }
    if (desired == RepoStatus::RUN || desired == RepoStatus::INIT) {
        TORCH_CHECK(
            lifecycle == Lifecycle::Running &&
                queue_core_.Snapshot().mode == queue::QueueMode::Running,
            "Task queue cannot publish an active status while its core is stopped.",
            PTA_ERROR(ErrCode::INTERNAL));
        repo_status.store(desired, std::memory_order_release);
        ++status_epoch_;
    } else {
        repo_status.store(desired, std::memory_order_release);
        ++status_epoch_;
        queue_core_.RequestStop();
    }
}

NPUStatus Repository::MakeSureQueueEmpty(bool check_error)
{
    TORCH_NPU_QUEUE_LOGD("MakeSureQueueEmpty: start, device = %d, queued = %zu, status = %d",
        device_idx, queue_core_.Snapshot().size, GetStatus());
    std::string error_msg;
    std::string runtime_error;
    if (!CheckInit()) {
        ASCEND_LOGE("Task queue is not initialized, shouldn't call MakeSureQueueEmpty(). !!");
        return NPU_STATUS_FAILED;
    }
    ASCEND_LOGI("Begin to makesure taskqueue empty.");
    if (lazy_fusion::IsEnabled()) {
        lazy_fusion::LazyFusionFlush();
    }
    // While waiting for ACL thread to launch tasks,
    // the current thread should not hold GIL.
    // When the operator compilation is triggered in the ACL thread,
    // the TE module attempts to obtain the GIL.
    // If the current thread does not release the GIL, a deadlock will
    // occur.
    if (consumer.joinable()) {
#ifndef BUILD_LIBTORCH
        QueueWaitHooks wait_hooks(g_used_aclop);
#else
        QueueWaitHooks wait_hooks(false);
#endif
        queue_core_.WaitDrained(wait_hooks);
    }

    const ErrorRecord error = SnapshotError();

    const RepoStatus current_status = GetStatus();
    auto iter = deviceErrorMap.find(current_status);
    if (iter != deviceErrorMap.end()) {
        std::string throwError = iter->second;
        std::string device_detail;
        if (current_status != RepoStatus::STOP_EXIT && current_status != RepoStatus::UCE_EXIT) {
            device_detail = c10_npu::c10_npu_get_error_message();
        }
        runtime_error = throwError + ", " + device_detail + PTA_ERROR(ErrCode::ACL);
        error_msg = throwError + " happened.";
    }

    if (current_status == RepoStatus::CAN_EXIT) {
        error_msg = "Inner error happened with CAN_EXIT status, detail: " + error.repository;
    }

    auto errmsg = GetQueueErrMsg();
    bool cannOOM = isCannOOM(errmsg);
    if (current_status == RepoStatus::ERROR_EXIT) {
        // Avoid repeatedly throwing exceptions
        SetStatus(CAN_EXIT);

        if (c10_npu::option::OptionsManager::IsOomSnapshotEnable() && cannOOM) {
            c10_npu::option::oom_observer();
        }

        runtime_error = "The Inner error is reported as above. "
            "The process exits for this inner error, and " +
            error.repository + ".\n" +
            "Since the operator is called asynchronously, the stacktrace may be inaccurate. "
            "If you want to get the accurate stacktrace, "
            "please set the environment variable ASCEND_LAUNCH_BLOCKING=1.\n" +
            "Note: ASCEND_LAUNCH_BLOCKING=1 will force ops to run in synchronous mode, "
            "resulting in performance degradation. "
            "Please unset ASCEND_LAUNCH_BLOCKING in time after debugging." +
            PTA_ERROR(ErrCode::ACL) + ".\n" + error.acl;
        error_msg = "Inner error happened, detail: " + error.repository;
    }

    if (!error_msg.empty()) {
        ASCEND_LOGE("%s", error_msg.c_str());
    }
    if (check_error && !runtime_error.empty()) {
        ASCEND_LOGE("runtime_error: %s", runtime_error.c_str());
        if (cannOOM) {
            TORCH_CHECK_WITH(OutOfMemoryError, false, runtime_error.c_str());
        } else {
            throw std::runtime_error(runtime_error);
        }
    }
    TORCH_NPU_QUEUE_LOGD("MakeSureQueueEmpty: clearing successful, device = %d, queued = %zu, status = %d",
        device_idx, queue_core_.Snapshot().size, GetStatus());

    return NPU_STATUS_SUCCESS;
}

void Repository::CheckDeviceError(int ret, std::string& err_msg)
{
    std::string acl_error;
    if (ret != ACL_ERROR_RT_DEVICE_TASK_ABORT && ret != ACL_ERROR_RT_DEVICE_MEM_ERROR) {
        acl_error = c10_npu::c10_npu_get_error_message();
        RecordAclError(acl_error);
    }
    if (ret == ACL_ERROR_RT_DEVICE_MEM_ERROR ||
        acl_error.find(DEVICE_MEM_ERROR) != std::string::npos ||
        acl_error.find(DEVICE_MEM_ERROR_V2) != std::string::npos) {
        if (checkUceErrAndRepair(false, err_msg)) {
            ASCEND_LOGE("UCE ERROR happened, set task queue status to UCE_EXIT");
            SetStatus(UCE_EXIT);
        }
    } else if (ret == ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR ||
               acl_error.find(DEVICE_HBM_ECC_ERROR) != std::string::npos ||
               acl_error.find(DEVICE_HBM_ECC_ERROR_V2) != std::string::npos) {
        record_mem_hbm_ecc_error();
        SetStatus(HBM_ECC_EXIT);
    } else if (ret == ACL_ERROR_RT_SUSPECT_DEVICE_MEM_ERROR ||
               acl_error.find(SUSPECT_DEVICE_MEM_ERROR) != std::string::npos ||
               acl_error.find(SUSPECT_DEVICE_MEM_ERROR_V2) != std::string::npos) {
        ASCEND_LOGE("SUSPECT MEM ERROR happened, set task queue status to SUSPECT_MEM_EXIT");
        SetStatus(SUSPECT_MEM_EXIT);
    } else if (ret == ACL_ERROR_RT_LINK_ERROR ||
               acl_error.find(HCCS_LINK_ERROR) != std::string::npos ||
               acl_error.find(HCCS_LINK_ERROR_V2) != std::string::npos ||
               acl_error.find(HCCS_LINK_ERROR_V3) != std::string::npos) {
        ASCEND_LOGE("HCCS LINK ERROR happened, set task queue status to HCCS_LINK_EXIT");
        SetStatus(HCCS_LINK_EXIT);
    } else if (ret == ACL_ERROR_RT_COMM_OP_RETRY_FAIL ||
               acl_error.find(HCCL_OP_RETRY_FAILED) != std::string::npos ||
               acl_error.find(HCCL_OP_RETRY_FAILED_V2) != std::string::npos) {
        ASCEND_LOGE("HCCL OP RETRY FAILED happened, set task queue status to HCCL_OP_RETRY_EXIT");
        SetStatus(HCCL_OP_RETRY_EXIT);
    } else if (ret == ACL_ERROR_RT_SUSPECT_REMOTE_ERROR ||
               acl_error.find(SUSPECT_REMOTE_ERROR) != std::string::npos ||
               acl_error.find(SUSPECT_REMOTE_ERROR_V2) != std::string::npos) {
        ASCEND_LOGE("SUSPECT REMOTE ERROR happened, set task queue status to SUSPECT_REMOTE_EXIT");
        SetStatus(SUSPECT_REMOTE_EXIT);
    } else if (GetStatus() != STOP_EXIT) {
        SetStatus(ERROR_EXIT);
    }
}

bool Repository::ReadQueue()
{
    bool execution_failed = false;
    const bool consumed = queue_core_.ConsumeOne([this, &execution_failed](uint32_t slot) {
        const std::string current_para = get_func_error_msg(manager().getCurrentParams(data, slot));
        RecordCurrentParameter(current_para);
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(2, data, slot);
        const auto ret = manager().Call(data, slot);
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(3, data, slot);
#else
        const auto ret = manager().Call(data, slot);
#endif
        if (ret != 0) {
            execution_failed = true;
            std::string repository_error = current_para;
            RecordRepositoryError(repository_error);
            ASCEND_LOGE("---Thread---%llu: device = %d, slot = %u, status = %d, ret = %d",
                std::this_thread::get_id(), device_idx, slot, GetStatus(), ret);
            TORCH_NPU_QUEUE_LOGI("ReadQueue: read failed, %s, device = %d, slot = %u, status = %d, ret = %d",
                repository_error.c_str(), device_idx, slot, GetStatus(), ret);
            manager().Release(data, slot, releaseQueue);
            std::string err_msg;
            CheckDeviceError(ret, err_msg);
            if (!err_msg.empty()) {
                repository_error += ". Other error information exists:" + err_msg;
                RecordRepositoryError(repository_error);
            }
            if (GetStatus() == RepoStatus::STOP_EXIT) {
                const auto acl_ret = c10_npu::acl::AclrtSynchronizeDeviceWithTimeout();
                ASCEND_LOGI("ReadQueue: SynchronizeDevice with FORCE STOP, device = %d, slot = %u, "
                            "status = %d, ret = %d, acl_ret = %d",
                    device_idx, slot, GetStatus(), ret, acl_ret);
            }
            return;
        }

        manager().Release(data, slot, releaseQueue);
        RecordCurrentParameter({});
        TORCH_NPU_QUEUE_LOGD("ReadQueue: read success, %s, device = %d, slot = %u, status = %d",
            current_para.c_str(), device_idx, slot, GetStatus());
        if (GetStatus() == RepoStatus::STOP_EXIT) {
            const auto acl_ret = c10_npu::acl::AclrtSynchronizeDeviceWithTimeout();
            ASCEND_LOGI("ReadQueue: SynchronizeDevice with FORCE STOP, device = %d, slot = %u, "
                        "status = %d, ret = %d, acl_ret = %d",
                device_idx, slot, GetStatus(), ret, acl_ret);
        }
    });
    if (execution_failed) {
        ClearQueue();
        c10_npu::NPUEventManager::GetInstance().ClearUnrecordedCount();
        return false;
    }
    return consumed;
}

void Repository::ThrowDeviceError(RepoStatus current_status, void* cur_paras)
{
    auto iter = deviceErrorMap.find(current_status);
    if (iter == deviceErrorMap.end()) {
        return;
    }
    std::string throwError = iter->second;
    auto queueParam = static_cast<c10_npu::queue::QueueParas *>(cur_paras);
    auto type = queueParam->paramType;
    // The RECORD_EVENT in the destructor process should not throw an exception.
    if (type == c10_npu::queue::LAZY_DESTROY_EVENT || type == c10_npu::queue::RECORD_EVENT) {
        return;
    }
    ASCEND_LOGE("getUceErrorFlag in Enqueue, throw %s.", throwError.c_str());
    std::string device_error_msg;
    if (current_status != RepoStatus::STOP_EXIT && current_status != RepoStatus::UCE_EXIT) {
        device_error_msg = c10_npu::c10_npu_get_error_message();
    }
    throw std::runtime_error(throwError + ", " + device_error_msg + PTA_ERROR(ErrCode::ACL));
}

void Repository::Enqueue(void *cur_paras)
{
    TORCH_CHECK(
        lifecycle_.load(std::memory_order_acquire) == Lifecycle::Running,
        "Task queue is not ready to accept a producer.",
        PTA_ERROR(ErrCode::INTERNAL));

    TORCH_NPU_QUEUE_LOGD("Enqueue: start, %s, device = %d, queued = %zu, status = %d",
        get_func_error_msg(cur_paras).c_str(), device_idx, queue_core_.Snapshot().size, GetStatus());

    const RepoStatus current_status = GetStatus();
    ThrowDeviceError(current_status, cur_paras);
    const ErrorRecord error = SnapshotError();

    if (current_status == RepoStatus::CAN_EXIT) {
        ASCEND_LOGE("Inner error happened with CAN_EXIT status, detail: %s", error.repository.c_str());
    }

    if (current_status == RepoStatus::ERROR_EXIT) {
        // Avoid repeatedly throwing exceptions
        SetStatus(CAN_EXIT);

        auto retmsg = std::string("The Inner error is reported as above. "
            "The process exits for this inner error, and " +
            error.repository + ".\n" +
            "Since the operator is called asynchronously, the stacktrace may be inaccurate. "
            "If you want to get the accurate stacktrace, "
            "please set the environment variable ASCEND_LAUNCH_BLOCKING=1.\n" +
            "Note: ASCEND_LAUNCH_BLOCKING=1 will force ops to run in synchronous mode, "
            "resulting in performance degradation. "
            "Please unset ASCEND_LAUNCH_BLOCKING in time after debugging." +
            PTA_ERROR(ErrCode::ACL) + ".\n" + error.acl);

        auto errmsg = GetQueueErrMsg();
        if (isCannOOM(errmsg)) {
            if (c10_npu::option::OptionsManager::IsOomSnapshotEnable()) {
                c10_npu::option::oom_observer();
            }
            ASCEND_LOGE("%s", retmsg.c_str());
            TORCH_CHECK_WITH(OutOfMemoryError, false, retmsg.c_str());
        }
        throw std::runtime_error(retmsg);
    }

    if (current_status != RUN && current_status != INIT) {
        auto queueParam = static_cast<c10_npu::queue::QueueParas *>(cur_paras);
        auto type = queueParam->paramType;
        if (type == c10_npu::queue::EXECUTE_OPAPI) {
            auto cur_paras = static_cast<at_npu::native::ExecuteParasOpApi *>(queueParam->paramVal);
            auto op_name = cur_paras->opType;
            ASCEND_LOGE("Task queue thread is exit, can't call Enqueue() for executing and op name is=%s.", op_name);
        } else if (type == c10_npu::queue::EXECUTE_OPAPI_V2) {
            auto cur_paras = static_cast<at_npu::native::ExecuteParasOpApiV2 *>(queueParam->paramVal);
            auto op_name = cur_paras->opName;
            ASCEND_LOGE("Task queue thread is exit, can't call Enqueue() for executing and op name is=%s.", op_name->c_str());
        } else if (type == c10_npu::queue::COMPILE_AND_EXECUTE) {
            auto cur_paras = static_cast<at_npu::native::ExecuteParas *>(queueParam->paramVal);
            auto op_name = cur_paras->opType;
            ASCEND_LOGW("Task queue thread is exit, can't call Enqueue() for executing and op name is=%s.", op_name);
        } else if (type == c10_npu::queue::ASYNC_MEMCPY) {
            auto cur_paras = static_cast<c10_npu::queue::CopyParas *>(queueParam->paramVal);
            ASCEND_LOGW("Task queue thread is exit, can't call Enqueue() for copy, srclen=%zu, dstlen is %zu, kind=%d",
                cur_paras->srcLen, cur_paras->dstLen, cur_paras->kind);
        } else {
            auto cur_paras = static_cast<c10_npu::queue::EventParas *>(queueParam->paramVal);
            ASCEND_LOGW("Task queue thread is exit, can't call Enqueue() for event, event is=%p", cur_paras->event);
        }
        return;
    }

    if (lazy_fusion::IsEnabled()) {
        auto type = static_cast<c10_npu::queue::QueueParas *>(cur_paras)->paramType;
        // LAZY_DESTROY_EVENT may execute in a separate thread, parallel with forward and backward threads.
        if (type != c10_npu::queue::LAZY_DESTROY_EVENT) {
            lazy_fusion::LazyFusionFlush();
        }
    }

#ifndef BUILD_LIBTORCH
    QueueWaitHooks wait_hooks(g_used_aclop);
#else
    QueueWaitHooks wait_hooks(false);
#endif
    const bool accepted = queue_core_.Enqueue(
        [this, cur_paras](uint32_t slot) {
            ThrowDeviceError(GetStatus(), cur_paras);
            // The deep copy is deliberately inside the core's short producer
            // commit section. Publishing tail/size before this returns would
            // expose a partially copied parameter record to the ACL thread.
            manager().Copy(data, slot, cur_paras);
        },
        wait_hooks);
    if (!accepted) {
        const RepoStatus rejected_status = GetStatus();
        ThrowDeviceError(rejected_status, cur_paras);
        TORCH_CHECK(
            rejected_status != RepoStatus::RUN && rejected_status != RepoStatus::INIT,
            "active task queue rejected a producer",
            PTA_ERROR(ErrCode::INTERNAL));
        TORCH_NPU_QUEUE_LOGI("Enqueue: producer admission closed, device = %d, status = %d",
            device_idx, rejected_status);
    }
}

void Repository::Dequeue()
{
    const Lifecycle lifecycle = lifecycle_.load(std::memory_order_acquire);
    if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
        ASCEND_LOGE("Task queue is not initialized, shouldn't call Dequeue(). !!");
        return;
    }

    while (!consumer_stop_requested_.load(std::memory_order_acquire) &&
           GetStatus() != RepoStatus::CAN_EXIT) {
        if (deviceErrorMap.find(GetStatus()) != deviceErrorMap.end()) {
            ClearQueue();
            c10_npu::NPUEventManager::GetInstance().ClearUnrecordedCount();
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
            continue;
        }
        bool consumed = false;
        try {
            consumed = ReadQueue();
        } catch (const std::exception& error) {
            RecordRepositoryError(error.what());
            SetStatus(RepoStatus::ERROR_EXIT);
            ClearQueue();
            c10_npu::NPUEventManager::GetInstance().ClearUnrecordedCount();
            break;
        } catch (...) {
            RecordRepositoryError("unknown exception in task queue consumer");
            SetStatus(RepoStatus::ERROR_EXIT);
            ClearQueue();
            c10_npu::NPUEventManager::GetInstance().ClearUnrecordedCount();
            break;
        }
        if (!consumed) {
            if (GetStatus() == RepoStatus::NEED_EXIT) {
                ChangeStatus(NEED_EXIT, CAN_EXIT);
                break;
            }

            if (GetStatus() == RepoStatus::ERROR_EXIT) {
                break;
            }

            if (GetStatus() == RepoStatus::STOP_EXIT) {
                std::this_thread::sleep_for(std::chrono::microseconds(1000));
                continue;
            }
        }
    }
}

void Repository::ReleaseResource()
{
    manager().DeInit(data);
    data = nullptr;
}

void Repository::ClearQueue()
{
    RepoStatus status = RepoStatus::INIT;
    uint64_t epoch = 0;
    queue::QueueSnapshot snapshot;
    {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
        if (lifecycle_.load(std::memory_order_acquire) != Lifecycle::Running) {
            // Initialization, restart, another cleanup, or destruction owns
            // the transition. It will leave the core in a coherent state.
            return;
        }
        status = repo_status.load(std::memory_order_acquire);
        epoch = status_epoch_;
        snapshot = queue_core_.Snapshot();

        // A consumer may have decided to clear under STOP just before another
        // thread completed restart. Once the new generation is Running, stale
        // cleanup must not touch it.
        if ((status == RepoStatus::RUN || status == RepoStatus::INIT) &&
            snapshot.mode == queue::QueueMode::Running) {
            return;
        }
        lifecycle_.store(Lifecycle::Initializing, std::memory_order_release);
    }

#ifndef BUILD_LIBTORCH
    QueueWaitHooks wait_hooks(g_used_aclop);
#else
    QueueWaitHooks wait_hooks(false);
#endif
    bool cleanup_failed = false;
    try {
        if (snapshot.failed) {
            // The callback that owns the head may have partially released it.
            // Metadata-only discard prevents a non-idempotent retry.
            queue_core_.StopAndDiscardMetadata(wait_hooks);
        } else {
            queue_core_.StopAndDiscard(
                [this](uint32_t slot) { manager().Release(data, slot, releaseQueue); },
                wait_hooks);
        }
    } catch (const std::exception& error) {
        cleanup_failed = true;
        RecordRepositoryError(std::string("Task queue cleanup failed: ") + error.what());
        ASCEND_LOGE("Task queue cleanup failed, discarding metadata: %s", error.what());
        try {
            queue_core_.StopAndDiscardMetadata(wait_hooks);
        } catch (...) {
            ASCEND_LOGE("Task queue metadata cleanup failed.");
        }
    } catch (...) {
        cleanup_failed = true;
        RecordRepositoryError("Task queue cleanup failed with an unknown exception.");
        ASCEND_LOGE("Task queue cleanup failed, discarding metadata.");
        try {
            queue_core_.StopAndDiscardMetadata(wait_hooks);
        } catch (...) {
            ASCEND_LOGE("Task queue metadata cleanup failed.");
        }
    }

    {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
        if (cleanup_failed) {
            repo_status.store(RepoStatus::ERROR_EXIT, std::memory_order_release);
            ++status_epoch_;
        } else if (
            status_epoch_ == epoch &&
            repo_status.load(std::memory_order_acquire) == status &&
            (status == RepoStatus::RUN || status == RepoStatus::INIT) &&
            queue_core_.Snapshot().mode == queue::QueueMode::Stopped) {
            // An explicit SetStatus recovery owns stop/join/start. Do not
            // reset sequence positions from a consumer's cleanup callback.
            repo_status.store(RepoStatus::ERROR_EXIT, std::memory_order_release);
            ++status_epoch_;
        }
        lifecycle_.store(Lifecycle::Running, std::memory_order_release);
    }
    lifecycle_cv_.notify_all();
}

void Repository::SetQueueErrMsg(const char *errmsg)
{
    std::lock_guard<std::mutex> lock(error_mu_);
    error_.queue = errmsg == nullptr ? std::string{} : std::string(errmsg);
}

std::string Repository::GetQueueErrMsg()
{
    std::lock_guard<std::mutex> lock(error_mu_);
    return error_.queue;
}

Repository::~Repository()
{
    {
        std::lock_guard<std::mutex> lock(lifecycle_mu_);
        const Lifecycle lifecycle = lifecycle_.load(std::memory_order_relaxed);
        if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
            return;
        }
        lifecycle_.store(Lifecycle::Stopping, std::memory_order_release);
    }

    repo_status.store(RepoStatus::NEED_EXIT, std::memory_order_release);
    queue_core_.RequestStop();
#ifndef BUILD_LIBTORCH
    QueueWaitHooks wait_hooks(g_used_aclop);
#else
    QueueWaitHooks wait_hooks(false);
#endif
    try {
        // Drain with runtime-aware wait hooks before joining. Once this
        // returns, the consumer has no leased task and join only observes its
        // loop exit; an attached Python thread never blocks the task callback.
        queue_core_.StopAndDrain(wait_hooks);
    } catch (const std::exception& error) {
        ASCEND_LOGE("Task queue drain failed, discarding pending tasks: %s", error.what());
        try {
            // StopAndDrain waits for active producers before surfacing a
            // shared callback error, so this fallback has no producer wait.
            if (queue_core_.Snapshot().failed) {
                queue_core_.StopAndDiscardMetadata(wait_hooks);
            } else {
                queue_core_.StopAndDiscard(
                    [this](uint32_t slot) { manager().Release(data, slot, releaseQueue); },
                    wait_hooks);
            }
        } catch (const std::exception& discard_error) {
            ASCEND_LOGE("Task queue cleanup failed: %s", discard_error.what());
            try {
                queue_core_.StopAndDiscardMetadata(wait_hooks);
            } catch (...) {
                ASCEND_LOGE("Task queue metadata cleanup failed.");
            }
        } catch (...) {
            ASCEND_LOGE("Task queue cleanup failed with an unknown exception.");
            try {
                queue_core_.StopAndDiscardMetadata(wait_hooks);
            } catch (...) {
                ASCEND_LOGE("Task queue metadata cleanup failed.");
            }
        }
    } catch (...) {
        ASCEND_LOGE("Task queue drain failed with an unknown exception.");
        try {
            if (queue_core_.Snapshot().failed) {
                queue_core_.StopAndDiscardMetadata(wait_hooks);
            } else {
                queue_core_.StopAndDiscard(
                    [this](uint32_t slot) { manager().Release(data, slot, releaseQueue); },
                    wait_hooks);
            }
        } catch (...) {
            ASCEND_LOGE("Task queue cleanup failed with an unknown exception.");
            try {
                queue_core_.StopAndDiscardMetadata(wait_hooks);
            } catch (...) {
                ASCEND_LOGE("Task queue metadata cleanup failed.");
            }
        }
    }
    JoinQueueActor(consumer, wait_hooks);
    try {
        releaseQueue.Shutdown();
    } catch (const std::exception& error) {
        ASCEND_LOGE("Release queue shutdown failed: %s", error.what());
    } catch (...) {
        ASCEND_LOGE("Release queue shutdown failed with an unknown exception.");
    }
    try {
        ReleaseResource();
    } catch (const std::exception& error) {
        ASCEND_LOGE("Task queue storage cleanup failed: %s", error.what());
    } catch (...) {
        ASCEND_LOGE("Task queue storage cleanup failed with an unknown exception.");
    }
    lifecycle_.store(Lifecycle::Stopped, std::memory_order_release);
    lifecycle_cv_.notify_all();
}

bool Repository::CheckInit() const
{
    return lifecycle_.load(std::memory_order_acquire) == Lifecycle::Running;
}

void StartConsume(Repository *repo, c10::DeviceIndex device_id)
{
    try {
        SetThreadType(ThreadType::ACL_THREAD);
        SetThreadAffinity(device_id);

        aclError ret = c10_npu::SetDevice(device_id);
        if (ret != 0) {
            C10_NPU_SHOW_ERR_MSG();
            ASCEND_LOGE("***Thread*%d: set device (%d): ret = %d", std::this_thread::get_id(), device_id, ret);
        }

        while (!repo->consumer_stop_requested_.load(std::memory_order_acquire) &&
               repo->GetStatus() != RepoStatus::CAN_EXIT && repo->GetStatus() != RepoStatus::ERROR_EXIT) {
            repo->Dequeue();
        }
    } catch (const std::exception& error) {
        ASCEND_LOGE("Task queue consumer escaped its dequeue loop: %s", error.what());
        try {
            repo->SetStatus(RepoStatus::ERROR_EXIT);
        } catch (...) {
            ASCEND_LOGE("Task queue consumer could not publish ERROR_EXIT.");
        }
    } catch (...) {
        ASCEND_LOGE("Task queue consumer escaped its dequeue loop with an unknown exception.");
        try {
            repo->SetStatus(RepoStatus::ERROR_EXIT);
        } catch (...) {
            ASCEND_LOGE("Task queue consumer could not publish ERROR_EXIT.");
        }
    }
}

void Repository::InitRepo(c10::DeviceIndex device_id)
{
    std::lock_guard<std::mutex> lock(lifecycle_mu_);
    const Lifecycle current = lifecycle_.load(std::memory_order_acquire);
    if (current == Lifecycle::Running) {
        return;
    }
    TORCH_CHECK(current == Lifecycle::Uninitialized || current == Lifecycle::Stopped,
        "Task queue cannot initialize from its current lifecycle state.", PTA_ERROR(ErrCode::INTERNAL));
    TORCH_CHECK(!consumer.joinable(), "Previous task queue consumer must be joined before initialization.",
        PTA_ERROR(ErrCode::INTERNAL));
    lifecycle_.store(Lifecycle::Initializing, std::memory_order_release);

    try {
        device_idx = device_id;
        repo_status.store(RepoStatus::INIT, std::memory_order_release);
        ++status_epoch_;
        {
            std::lock_guard<std::mutex> error_lock(error_mu_);
            error_ = ErrorRecord{};
        }
        data = manager().Init(static_cast<int>(queue_core_.Capacity()));
        ASCEND_LOGI("TaskQueue is enable");
        queue_core_.Start();

        consumer_stop_requested_.store(false, std::memory_order_release);
        // Release storage must be live before the main consumer can transfer
        // ownership to it.
        releaseQueue.InitReleaseQueue(device_id);
        std::thread cur_consumer(StartConsume, this, device_id);
        consumer = std::move(cur_consumer);
        lifecycle_.store(Lifecycle::Running, std::memory_order_release);
        lifecycle_cv_.notify_all();
    } catch (...) {
        const std::exception_ptr initialization_error = std::current_exception();
        queue_core_.RequestStop();
        if (consumer.joinable()) {
            repo_status.store(RepoStatus::NEED_EXIT, std::memory_order_release);
#ifndef BUILD_LIBTORCH
            QueueWaitHooks wait_hooks(g_used_aclop);
#else
            QueueWaitHooks wait_hooks(false);
#endif
            try {
                JoinQueueActor(consumer, wait_hooks);
            } catch (...) {
                ASCEND_LOGE("Task queue initialization consumer cleanup failed.");
            }
        }
        try {
            queue_core_.StopAndDiscard();
        } catch (...) {
            ASCEND_LOGE("Task queue initialization metadata cleanup failed.");
        }
        try {
            releaseQueue.Shutdown();
        } catch (...) {
            ASCEND_LOGE("Task queue initialization release cleanup failed.");
        }
        try {
            ReleaseResource();
        } catch (...) {
            ASCEND_LOGE("Task queue initialization storage cleanup failed.");
        }
        data = nullptr;
        lifecycle_.store(Lifecycle::Stopped, std::memory_order_release);
        lifecycle_cv_.notify_all();
        std::rethrow_exception(initialization_error);
    }
}

std::string Repository::GetPara()
{
    const ErrorRecord error = SnapshotError();
    if (!error.current_parameter.empty()) {
        return error.current_parameter;
    }
    std::string repo_para = "EmptyQueue";
    queue_core_.InspectHead([this, &repo_para](uint32_t slot) {
        repo_para = get_func_error_msg(manager().getCurrentParams(data, slot));
    });
    return repo_para;
}
void ReleaseQueue::PushToReleaseQueue(void *cur_paras)
{
    TORCH_CHECK(
        lifecycle_.load(std::memory_order_acquire) == Lifecycle::Running,
        "Release queue is not ready to accept a parameter.",
        PTA_ERROR(ErrCode::INTERNAL));

#ifndef BUILD_LIBTORCH
    QueueWaitHooks wait_hooks(g_used_aclop);
#else
    QueueWaitHooks wait_hooks(false);
#endif
    const bool accepted = queue_core_.Enqueue(
        [this, cur_paras](uint32_t slot) { releaseManager().CopyRealseParam(data, slot, cur_paras); }, wait_hooks);
    TORCH_CHECK(accepted, "Release queue stopped before accepting a parameter.", PTA_ERROR(ErrCode::INTERNAL));
}

void ReleaseQueue::PopFromReleaseQueue()
{
    const Lifecycle lifecycle = lifecycle_.load(std::memory_order_acquire);
    if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
        ASCEND_LOGE("Release queue is not initialized, shouldn't call PopFromReleaseQueue(). !!");
        return;
    }

    const bool consumed = queue_core_.ConsumeOne(
        [this](uint32_t slot) { releaseManager().ReleaseParam(data, slot); });
    if (!consumed && GetStatus() == RepoStatus::NEED_EXIT) {
        ChangeStatus(NEED_EXIT, CAN_EXIT);
    }
}

void StartRelease(ReleaseQueue *releaseQue)
{
    try {
        SetThreadType(ThreadType::RELEASE_THREAD);
        SetThreadAffinity(releaseQue->GetDeviceID());

        while (releaseQue->GetStatus() != RepoStatus::CAN_EXIT) {
            releaseQue->PopFromReleaseQueue();
        }
    } catch (const std::exception& error) {
        ASCEND_LOGE("Release queue consumer failed: %s", error.what());
        try {
            releaseQue->SetStatus(RepoStatus::ERROR_EXIT);
        } catch (...) {
            ASCEND_LOGE("Release queue consumer could not publish ERROR_EXIT.");
        }
    } catch (...) {
        ASCEND_LOGE("Release queue consumer failed with an unknown exception.");
        try {
            releaseQue->SetStatus(RepoStatus::ERROR_EXIT);
        } catch (...) {
            ASCEND_LOGE("Release queue consumer could not publish ERROR_EXIT.");
        }
    }
}

void ReleaseQueue::InitReleaseQueue(c10::DeviceIndex device_id)
{
    std::lock_guard<std::mutex> lock(lifecycle_mu_);
    const Lifecycle current = lifecycle_.load(std::memory_order_acquire);
    if (current == Lifecycle::Running) {
        return;
    }
    TORCH_CHECK(current == Lifecycle::Uninitialized || current == Lifecycle::Stopped,
        "Release queue cannot initialize from its current lifecycle state.", PTA_ERROR(ErrCode::INTERNAL));
    TORCH_CHECK(!releaser.joinable(), "Previous release consumer must be joined before initialization.",
        PTA_ERROR(ErrCode::INTERNAL));
    lifecycle_.store(Lifecycle::Initializing, std::memory_order_release);
    try {
        device_idx = device_id;
        repo_status.store(RepoStatus::INIT, std::memory_order_release);
        data = releaseManager().Init(static_cast<int>(queue_core_.Capacity()));
        queue_core_.Start();
        std::thread cur_releaser(StartRelease, this);
        releaser = std::move(cur_releaser);
        lifecycle_.store(Lifecycle::Running, std::memory_order_release);
    } catch (...) {
        const std::exception_ptr initialization_error = std::current_exception();
        try {
            queue_core_.StopAndDiscard();
        } catch (...) {
            ASCEND_LOGE("Release queue initialization metadata cleanup failed.");
        }
        try {
            releaseManager().DeInit(data);
        } catch (...) {
            ASCEND_LOGE("Release queue initialization storage cleanup failed.");
        }
        data = nullptr;
        lifecycle_.store(Lifecycle::Stopped, std::memory_order_release);
        std::rethrow_exception(initialization_error);
    }
}

ReleaseQueue::~ReleaseQueue()
{
    try {
        Shutdown();
    } catch (const std::exception& error) {
        ASCEND_LOGE("Release queue cleanup failed: %s", error.what());
    } catch (...) {
        ASCEND_LOGE("Release queue cleanup failed with an unknown exception.");
    }
}

void ReleaseQueue::Shutdown()
{
    {
        std::lock_guard<std::mutex> lock(lifecycle_mu_);
        const Lifecycle current = lifecycle_.load(std::memory_order_relaxed);
        if (current == Lifecycle::Uninitialized || current == Lifecycle::Stopped ||
            current == Lifecycle::Stopping) {
            return;
        }
        lifecycle_.store(Lifecycle::Stopping, std::memory_order_release);
    }

    if (repo_status.load(std::memory_order_acquire) != RepoStatus::ERROR_EXIT) {
        repo_status.store(RepoStatus::NEED_EXIT, std::memory_order_release);
    }
    queue_core_.RequestStop();
#ifndef BUILD_LIBTORCH
    QueueWaitHooks wait_hooks(g_used_aclop);
#else
    QueueWaitHooks wait_hooks(false);
#endif
    JoinQueueActor(releaser, wait_hooks);

    const bool release_failed = queue_core_.Snapshot().failed;
    if (release_failed) {
        // The failing callback may have partially destroyed its payload even
        // though the core retired its cursor. Never retry ReleaseParam.
        try {
            queue_core_.StopAndDiscard();
        } catch (...) {
            ASCEND_LOGE("Release queue metadata cleanup failed.");
        }
    } else {
        try {
            queue_core_.StopAndDiscard(
                [this](uint32_t slot) { releaseManager().ReleaseParam(data, slot); });
        } catch (const std::exception& error) {
            ASCEND_LOGE("Release queue cleanup failed, discarding metadata: %s", error.what());
            try {
                queue_core_.StopAndDiscard();
            } catch (...) {
                ASCEND_LOGE("Release queue metadata cleanup failed.");
            }
        } catch (...) {
            ASCEND_LOGE("Release queue cleanup failed, discarding metadata.");
            try {
                queue_core_.StopAndDiscard();
            } catch (...) {
                ASCEND_LOGE("Release queue metadata cleanup failed.");
            }
        }
    }
    try {
        releaseManager().DeInit(data);
    } catch (const std::exception& error) {
        ASCEND_LOGE("Release queue storage cleanup failed: %s", error.what());
    } catch (...) {
        ASCEND_LOGE("Release queue storage cleanup failed with an unknown exception.");
    }
    data = nullptr;
    lifecycle_.store(Lifecycle::Stopped, std::memory_order_release);
}

RepoStatus ReleaseQueue::GetStatus() const
{
    const Lifecycle lifecycle = lifecycle_.load(std::memory_order_acquire);
    if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
        ASCEND_LOGE("Release queue is not initialized, shouldn't call GetStatus(). !!");
    }

    return repo_status.load();
}

c10::DeviceIndex ReleaseQueue::GetDeviceID() const
{
    return device_idx;
}


void ReleaseQueue::SetStatus(RepoStatus desired)
{
    const Lifecycle lifecycle = lifecycle_.load(std::memory_order_acquire);
    if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
        ASCEND_LOGE("Release queue is not initialized, shouldn't call SetStatus(). !!");
        return;
    }

    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
    if (desired == RepoStatus::RUN || desired == RepoStatus::INIT) {
        TORCH_CHECK(queue_core_.Snapshot().mode == queue::QueueMode::Running,
            "Release queue activation requires initialization after its actor joined.", PTA_ERROR(ErrCode::INTERNAL));
    }
    repo_status = desired;
    if (desired != RepoStatus::RUN && desired != RepoStatus::INIT) {
        queue_core_.RequestStop();
    }
}

void ReleaseQueue::ChangeStatus(RepoStatus expected, RepoStatus desired)
{
    const Lifecycle lifecycle = lifecycle_.load(std::memory_order_acquire);
    if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
        ASCEND_LOGE("Release queue is not initialized, shouldn't call ChangeStatus(). !!");
        return;
    }

    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
    if (desired == RepoStatus::RUN || desired == RepoStatus::INIT) {
        TORCH_CHECK(queue_core_.Snapshot().mode == queue::QueueMode::Running,
            "Release queue activation requires initialization after its actor joined.", PTA_ERROR(ErrCode::INTERNAL));
    }
    if (repo_status.compare_exchange_strong(expected, desired)) {
        if (desired != RepoStatus::RUN && desired != RepoStatus::INIT) {
            queue_core_.RequestStop();
        }
    }
}
} // namespace c10_npu
