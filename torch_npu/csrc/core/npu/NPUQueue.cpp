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

#ifndef BUILD_LIBTORCH
#include <Python.h>
#endif

#include <ATen/record_function.h>
#include <unistd.h>
#include <sstream>
#include <sys/time.h>
#include <sys/eventfd.h>
#include <third_party/acl/inc/acl/acl_rt.h>

namespace c10_npu {
struct timeval delay = { 0, 1 };

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
        return (uint8_t *)head + sizePerParams * offset;
    }

    int Call(void *head, int offset)
    {
        TORCH_CHECK(this->execFunc, "Failed to find execution function.", PTA_ERROR(ErrCode::NOT_FOUND));
        auto dstPtr = (uint8_t *)head + sizePerParams * offset;
        return this->execFunc(dstPtr);
    }

    void Copy(void *dstHead, int offset, void *src)
    {
        TORCH_CHECK(this->copyFunc, "Failed to find copy function.", PTA_ERROR(ErrCode::NOT_FOUND));
        auto dstPtr = (uint8_t *)dstHead + sizePerParams * offset;
        return this->copyFunc(dstPtr, src);
    }

    void Release(void *head, int offset, ReleaseQueue &releaseQueue)
    {
        TORCH_CHECK(this->releaseFunc, "Failed to find release function.", PTA_ERROR(ErrCode::NOT_FOUND));
        auto ptr = (uint8_t *)head + sizePerParams * offset;
        return this->releaseFunc(ptr, releaseQueue);
    }

    void CopyRealseParam(void *dstHead, int offset, void *src)
    {
        TORCH_CHECK(this->copyReleaseParamFunc, "Failed to find copy release params function.",
            PTA_ERROR(ErrCode::NOT_FOUND));
        auto dstPtr = (uint8_t *)dstHead + sizePerParams * offset;
        return this->copyReleaseParamFunc(dstPtr, src);
    }

    void ReleaseParam(void *head, int offset)
    {
        TORCH_CHECK(this->releaseParamFunc, "Failed to find release params function.", PTA_ERROR(ErrCode::NOT_FOUND));
        auto ptr = (uint8_t *)head + sizePerParams * offset;
        return this->releaseParamFunc(ptr);
    }

    void *Init(int capacity)
    {
        TORCH_CHECK(this->newFunc, "Failed to find new function.", PTA_ERROR(ErrCode::NOT_FOUND));
        void *ptr = this->newFunc(capacity, sizePerParams); // not check as CUDA
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
    int sizePerParams = 0;
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
static constexpr size_t kQueueCapacity = 4096;
static std::string repo_error;
static std::string acl_error;

std::unordered_map<RepoStatus, std::string> deviceErrorMap = {
    {RepoStatus::UCE_EXIT, "UCE ERROR"},
    {RepoStatus::HBM_ECC_EXIT, "HBM MULTI BIT ECC ERROR"},
    {RepoStatus::STOP_EXIT, "FORCE STOP"},
    {RepoStatus::SUSPECT_MEM_EXIT, "SUSPECT MEM ERROR"},
    {RepoStatus::HCCS_LINK_EXIT, "HCCS LINK ERROR"},
    {RepoStatus::HCCL_OP_RETRY_EXIT, "HCCL OP RETRY FAILED"}
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

RepoStatus Repository::GetStatus() const
{
    if (initialized == false) {
        ASCEND_LOGE("Task queue is not initialized, shouldn't call GetStatus(). !!");
    }

    return repo_status.load();
}

void Repository::SetStatus(RepoStatus desired)
{
    if (initialized == false) {
        ASCEND_LOGE("Task queue is not initialized, shouldn't call SetStatus(). !!");
        return;
    }

    repo_status = desired;
}

void Repository::ChangeStatus(RepoStatus expected, RepoStatus desired)
{
    if (initialized == false) {
        ASCEND_LOGE("Task queue is not initialized, shouldn't call ChangeStatus(). !!");
        return;
    }

    repo_status.compare_exchange_strong(expected, desired);
}

NPUStatus Repository::MakeSureQueueEmpty(bool check_error)
{
    std::string error_msg;
    std::string runtime_error;
    if (initialized == false) {
        ASCEND_LOGE("Task queue is not initialized, shouldn't call MakeSureQueueEmpty(). !!");
        return FAILED;
    }
    ASCEND_LOGI("Begin to makesure taskqueue empty.");
    // While waiting for ACL thread to launch tasks,
    // the current thread should not hold GIL.
    // When the operator compilation is triggered in the ACL thread,
    // the TE module attempts to obtain the GIL.
    // If the current thread does not release the GIL, a deadlock will
    // occur.
#ifndef BUILD_LIBTORCH
    PyThreadState *gilState = nullptr;
    if (PyGILState_Check() != 0 && g_used_aclop) {
        gilState = PyEval_SaveThread();
    }
#endif

    if (consumer.joinable()) {
        ssize_t s;
        uint64_t u = 1;
        while (!IsEmptyQueue()) {
            std::lock_guard<std::mutex> lock(mu_empty);
            need_empty = true;
            __sync_synchronize();
            if (!IsEmptyQueue()) { // double-check, very important idea
                s = eventfd_read(efd_empty, &u);
                if (s != 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    ASCEND_LOGE("eventfd_read failed. s=%zd, errno=%s.", s, strerror(errno));
#ifndef BUILD_LIBTORCH
                    // Get the GIL
                    if (gilState) {
                        PyEval_RestoreThread(gilState);
                    }
#endif
                    return INTERNEL_ERROR;
                }
            }
            need_empty = false;
        }
    }

    const RepoStatus current_status = GetStatus();
    auto iter = deviceErrorMap.find(current_status);
    if (iter != deviceErrorMap.end()) {
        std::string throwError = iter->second;
        std::string error_msg;
        if (current_status != RepoStatus::STOP_EXIT && current_status != RepoStatus::UCE_EXIT) {
            error_msg = c10_npu::c10_npu_get_error_message();
        }
        runtime_error = throwError + ", " + error_msg + PTA_ERROR(ErrCode::ACL);
        error_msg = throwError + " happened.";
    }

    if (current_status == RepoStatus::CAN_EXIT) {
        error_msg = "Inner error happened with CAN_EXIT status, detail: " + repo_error;
    }

    if (current_status == RepoStatus::ERROR_EXIT) {
        // Avoid repeatedly throwing exceptions
        SetStatus(CAN_EXIT);

        if (c10_npu::option::OptionsManager::IsOomSnapshotEnable()) {
            auto errmsg = GetQueueErrMsg();
            const char *memerror = "Failed to allocate memory";
            if (strstr(errmsg, memerror) != nullptr) {
                c10_npu::option::oom_observer();
            }
        }

        runtime_error = "The Inner error is reported as above. "
            "The process exits for this inner error, and " +
            repo_error + ".\n" +
            "Since the operator is called asynchronously, the stacktrace may be inaccurate. "
            "If you want to get the accurate stacktrace, "
            "please set the environment variable ASCEND_LAUNCH_BLOCKING=1.\n" +
            "Note: ASCEND_LAUNCH_BLOCKING=1 will force ops to run in synchronous mode, "
            "resulting in performance degradation. "
            "Please unset ASCEND_LAUNCH_BLOCKING in time after debugging." +
            PTA_ERROR(ErrCode::ACL) + ".\n" + acl_error;
        error_msg = "Inner error happened, detail: " + repo_error;
    }

#ifndef BUILD_LIBTORCH
    // Get the GIL
    if (gilState) {
        PyEval_RestoreThread(gilState);
    }
#endif

    if (!error_msg.empty()) {
        ASCEND_LOGE("%s", error_msg.c_str());
    }
    if (check_error && !runtime_error.empty()) {
        throw std::runtime_error(runtime_error);
    }

    return SUCCESS;
}

bool Repository::WriteQueue(void *cur_paras)
{
    std::lock_guard<std::mutex> lock(mu_enqueue);

    const RepoStatus current_status = GetStatus();
    ThrowDeviceError(current_status, cur_paras);

    if (IsFullQueue()) {
        return false;
    }

    __sync_synchronize();
    manager().Copy(datas, write_idx.idx, cur_paras);
    __sync_synchronize();

    write_idx.idx = (write_idx.idx + 1) & (kQueueCapacity - 1);
    return true;
}

void Repository::CheckDeviceError(int ret, std::string& err_msg)
{
    if (ret != ACL_ERROR_RT_DEVICE_TASK_ABORT && ret != ACL_ERROR_RT_DEVICE_MEM_ERROR) {
        acl_error = c10_npu::c10_npu_get_error_message();
    }
    if (ret == ACL_ERROR_RT_DEVICE_MEM_ERROR || acl_error.find(DEVICE_HBM_ECC_ERROR) != std::string::npos) {
        if (checkUceErrAndRepair(false, err_msg)) {
            ASCEND_LOGE("UCE ERROR happened, set task queue status to UCE_EXIT");
            SetStatus(UCE_EXIT);
        }
    } else if (ret == ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR || acl_error.find(DEVICE_HBM_ECC_ERROR) != std::string::npos) {
        record_mem_hbm_ecc_error();
        SetStatus(HBM_ECC_EXIT);
    } else if (ret == ACL_ERROR_RT_SUSPECT_DEVICE_MEM_ERROR || acl_error.find(SUSPECT_DEVICE_MEM_ERROR) != std::string::npos) {
        ASCEND_LOGE("SUSPECT MEM ERROR happened, set task queue status to SUSPECT_MEM_EXIT");
        SetStatus(SUSPECT_MEM_EXIT);
    } else if (ret == ACL_ERROR_RT_LINK_ERROR || acl_error.find(HCCS_LINK_ERROR) != std::string::npos) {
        ASCEND_LOGE("HCCS LINK ERROR happened, set task queue status to HCCS_LINK_EXIT");
        SetStatus(HCCS_LINK_EXIT);
    } else if (ret == ACL_ERROR_RT_COMM_OP_RETRY_FAIL || acl_error.find(HCCL_OP_RETRY_FAILED) != std::string::npos) {
        ASCEND_LOGE("HCCL OP RETRY FAILED happened, set task queue status to HCCL_OP_RETRY_EXIT");
        SetStatus(HCCL_OP_RETRY_EXIT);
    } else if (GetStatus() != STOP_EXIT) {
        SetStatus(ERROR_EXIT);
    }
}

bool Repository::ReadQueue()
{
    if (IsEmptyQueue()) {
        static const auto task_queue_enable = c10_npu::option::OptionsManager::GetTaskQueueEnable();
        if (task_queue_enable == 2) {
            // read queue polls for at most 1 ms when queue is empty.
            for (int i = 0; i < READ_QUEUE_POLL_MAX_LOOP; ++i) {
                if (!IsEmptyQueue()) {
                    break;
                }
            }
            if (IsEmptyQueue()) {
                return false;
            }
        } else {
            return false;
        }
    }

    __sync_synchronize();
#ifndef BUILD_LIBTORCH
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(2, datas, read_idx.idx);
    auto ret = manager().Call(datas, read_idx.idx);
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(3, datas, read_idx.idx);
#else
    auto ret = manager().Call(datas, read_idx.idx);
#endif
    if (ret != 0) {
        repo_error = get_func_error_msg(manager().getCurrentParams(datas, read_idx.idx));
        ASCEND_LOGE("---Thread---%llu: device = %d, write_idx = %u, read_idx = %u, status = %d, ret = %d",
            std::this_thread::get_id(), device_idx, write_idx.idx, read_idx.idx, GetStatus(), ret);
        while (!IsEmptyQueue()) { // ignore other tasks
            manager().Release(datas, read_idx.idx, releaseQueue);
            read_idx.idx = (read_idx.idx + 1) & (kQueueCapacity - 1);
        }
        std::string err_msg;
        CheckDeviceError(ret, err_msg);
        if (!err_msg.empty()) {
            repo_error = repo_error + ". Other error information exists:" + err_msg;
        }
        ClearQueue();
        c10_npu::NPUEventManager::GetInstance().ClearUnrecordedCount();
        return false;
    }

    manager().Release(datas, read_idx.idx, releaseQueue);
    __sync_synchronize();

    read_idx.idx = (read_idx.idx + 1) & (kQueueCapacity - 1);

    return true;
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
    std::string error_msg;
    if (current_status != RepoStatus::STOP_EXIT && current_status != RepoStatus::UCE_EXIT) {
        error_msg = c10_npu::c10_npu_get_error_message();
    }
    throw std::runtime_error(throwError + ", " + error_msg + PTA_ERROR(ErrCode::ACL));
}

void Repository::Enqueue(void *cur_paras)
{
    if (initialized == false) {
        ASCEND_LOGE("Task queue is not initialized, shouldn't call Enqueue(). !!");
        return;
    }

    const RepoStatus current_status = GetStatus();
    ThrowDeviceError(current_status, cur_paras);

    if (current_status == RepoStatus::CAN_EXIT) {
        ASCEND_LOGE("Inner error happened with CAN_EXIT status, detail: %s", repo_error.c_str());
    }

    if (current_status == RepoStatus::ERROR_EXIT) {
        // Avoid repeatedly throwing exceptions
        SetStatus(CAN_EXIT);

        if (c10_npu::option::OptionsManager::IsOomSnapshotEnable()) {
            auto errmsg = GetQueueErrMsg();
            const char *memerror = "Failed to allocate memory";
            if (strstr(errmsg, memerror) != nullptr) {
                c10_npu::option::oom_observer();
            }
        }

        throw std::runtime_error("The Inner error is reported as above. "
            "The process exits for this inner error, and " +
            repo_error + ".\n" +
            "Since the operator is called asynchronously, the stacktrace may be inaccurate. "
            "If you want to get the accurate stacktrace, "
            "please set the environment variable ASCEND_LAUNCH_BLOCKING=1.\n" +
            "Note: ASCEND_LAUNCH_BLOCKING=1 will force ops to run in synchronous mode, "
            "resulting in performance degradation. "
            "Please unset ASCEND_LAUNCH_BLOCKING in time after debugging." +
            PTA_ERROR(ErrCode::ACL) + ".\n" + acl_error);
    }

    if (current_status != RUN && current_status != INIT) {
        auto queueParam = static_cast<c10_npu::queue::QueueParas *>(cur_paras);
        auto type = queueParam->paramType;
        if (type == c10_npu::queue::EXECUTE_OPAPI) {
            auto cur_paras = static_cast<at_npu::native::ExecuteParasOpApi *>(queueParam->paramVal);
            auto op_name = cur_paras->opType;
            ASCEND_LOGE("Task queue thread is exit, can't call Enqueue() for executing and op name is=%s.", op_name);
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
    bool ret = false;
    ssize_t s;
    uint64_t u = 1;

    SetWriteWorking(true);
    while (!ret && (GetStatus() == RUN || GetStatus() == INIT)) {
        ret = WriteQueue(cur_paras);
        if (ret == false) {
            SetWriteWorking(false);
            __sync_synchronize();
            if (IsFullQueue()) {
#ifndef BUILD_LIBTORCH
                // double check the current thread hold a Gil lock
                // and release the GIL to TE op compiler in case the acl thread deadlock.
                // However, this operator could produce another form of deadlock. 
                // When thread A deconstract a tensor, it will hold the mutex of deviceCachingAllocator and insert an event into the taskqueue.
                // If the taskqueue is full, thead A will run into here and release the GIL.
                // Once another thread B get GIL and trigger GC, it may deconstract another tensor
                // and try to get deviceCachingAllocator's mutex, which would cause another form of deadlock.
                // Since the aclop will be deprecated soon, we just add a using-aclop check here to aviod the second case of deadlock.
                if (PyGILState_Check() != 0 && g_used_aclop) {
                    Py_BEGIN_ALLOW_THREADS s = eventfd_read(efd_write, &u);
                    Py_END_ALLOW_THREADS
                } else {
                    s = eventfd_read(efd_write, &u);
                }
#else
                s = eventfd_read(efd_write, &u);
#endif
                if (s != 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    ASCEND_LOGE("waiting dequeue failed. s=%zd, errno=%s.", s, strerror(errno));
                    return;
                }
                SetWriteWorking(true);
            }
            continue;
        }
        __sync_synchronize();
        while (!IsReadWorking()) {
            s = eventfd_write(efd_read, u);
            if (s != 0) {
                if (errno == EINTR) {
                    continue;
                }
                ASCEND_LOGE("notify consumer failed!! s=%zd, errno=%s", s, strerror(errno));
                return;
            }
            break;
        }
    }
    SetWriteWorking(false);
}

void Repository::Dequeue()
{
    if (initialized == false) {
        ASCEND_LOGE("Task queue is not initialized, shouldn't call Dequeue(). !!");
        return;
    }

    bool ret = false;
    bool notify_empty = false;
    ssize_t s;
    uint64_t u = 1;

    SetReadWorking(true);
    while (ret == false && GetStatus() != RepoStatus::CAN_EXIT) {
        if (deviceErrorMap.find(GetStatus()) != deviceErrorMap.end()) {
            ClearQueue();
            c10_npu::NPUEventManager::GetInstance().ClearUnrecordedCount();
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
            continue;
        }
        ret = ReadQueue();
        if (ret == false) {
            if (GetStatus() == RepoStatus::NEED_EXIT) {
                ChangeStatus(NEED_EXIT, CAN_EXIT);
                break;
            }

            if (GetStatus() == RepoStatus::ERROR_EXIT) {
                break;
            }

            if (GetStatus() == RepoStatus::STOP_EXIT) {
                continue;
            }

            SetReadWorking(false);
            __sync_synchronize();
            if (IsEmptyQueue()) {
                s = eventfd_read(efd_read, &u);
                if (s != 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    ASCEND_LOGE("waiting enqueue failed. s=%zd, errno=%s.", s, strerror(errno));
                    return;
                }
                SetReadWorking(true);
            }
            continue;
        }
        __sync_synchronize();
        notify_empty = need_empty && IsEmptyQueue(); // need_empty && (ret == false || IsEmptyQueue());
        while (notify_empty) {
            s = eventfd_write(efd_empty, u);
            if (s != 0) {
                if (errno == EINTR) {
                    continue;
                }
                ASCEND_LOGE("notify make_sure failed. s=%zd, errno=%s.", s, strerror(errno));
                return;
            }
            break;
        }
        __sync_synchronize();
        while (!IsWriteWorking()) {
            s = eventfd_write(efd_write, u);
            if (s != 0) {
                if (errno == EINTR) {
                    continue;
                }
                ASCEND_LOGE("notify producer failed. s=%zd, errno=%s.", s, strerror(errno));
                return;
            }
            break;
        }
    }
    SetReadWorking(false);
}

void Repository::ReleaseResource()
{
    manager().DeInit(datas);
    if (efd_read > 0) {
        close(efd_read);
        efd_read = -1;
    }
    if (efd_write > 0) {
        close(efd_write);
        efd_write = -1;
    }
    if (efd_empty > 0) {
        close(efd_empty);
        efd_empty = -1;
    }
}

void Repository::ClearQueue()
{
    read_idx.idx = write_idx.idx;
    __sync_synchronize();
    eventfd_write(efd_empty, 1);
    eventfd_write(efd_write, 1);
}

void Repository::SetQueueErrMsg(const char *errmsg)
{
    error_msg = errmsg;
}

const char *Repository::GetQueueErrMsg()
{
    return error_msg;
}

Repository::~Repository()
{
    if (initialized) {
        if (consumer.joinable()) {
            SetStatus(NEED_EXIT);
            (void)eventfd_write(efd_read, 1); // escape wait
            consumer.join();
        }
        eventfd_write(efd_empty, 1);
        ReleaseResource();
    }
}

bool Repository::IsFullQueue() const
{
    return ((write_idx.idx + 1) & (kQueueCapacity - 1)) == read_idx.idx;
}

bool Repository::CheckInit() const
{
    return initialized;
}

void StartConsume(Repository *repo, c10::DeviceIndex device_id)
{
    SetThreadType(ThreadType::ACL_THREAD);
    SetThreadAffinity(device_id);

    aclError ret = c10_npu::SetDevice(device_id);
    if (ret != 0) {
        C10_NPU_SHOW_ERR_MSG();
        ASCEND_LOGE("***Thread*%d: set device (%d): ret = %d", std::this_thread::get_id(), device_id, ret);
    }

    while (repo->GetStatus() != RepoStatus::CAN_EXIT and repo->GetStatus() != RepoStatus::ERROR_EXIT) {
        repo->Dequeue();
    }
    return;
}

void Repository::InitRepo(c10::DeviceIndex device_id)
{
    if (datas == nullptr) {
        datas = manager().Init(kQueueCapacity);
        ASCEND_LOGI("TaskQueue is enable");
    }

    efd_read = eventfd(0, 0);
    efd_write = eventfd(0, 0);
    efd_empty = eventfd(0, 0);

    initialized = true;
    SetStatus(INIT);
    device_idx = device_id;
    std::thread cur_consumer(StartConsume, this, device_id);
    consumer = std::move(cur_consumer);

    releaseQueue.InitReleaseQueue(device_id);
}

std::string Repository::GetPara()
{
    if (IsEmptyQueue()) {
        return "EmptyQueue";
    }
    __sync_synchronize();
    std::string repo_para = get_func_error_msg(manager().getCurrentParams(datas, read_idx.idx));
    __sync_synchronize();
    return repo_para;
}

static constexpr size_t kReleaseQueueCapacity = 8192;
bool ReleaseQueue::WriteToReleaseQueue(void *cur_paras)
{
    if (IsFullQueue()) {
        return false;
    }
    __sync_synchronize();
    releaseManager().CopyRealseParam(datas, write_idx.idx, cur_paras);

    __sync_synchronize();
    write_idx.idx = (write_idx.idx + 1) & (kReleaseQueueCapacity - 1);
    return true;
}

void ReleaseQueue::PushToReleaseQueue(void *cur_paras)
{
    if (initialized == false) {
        ASCEND_LOGE("Release queue is not initialized, shouldn't call PushToReleaseQueue(). !!");
        return;
    }

    bool ret = false;
    while (ret == false) {
        ret = WriteToReleaseQueue(cur_paras);
        if (ret == true) {
            break;
        }
    }
}

bool ReleaseQueue::ReadFromReleaseQueue()
{
    if (IsEmptyQueue()) {
        return false;
    }

    __sync_synchronize();
    releaseManager().ReleaseParam(datas, read_idx.idx);

    __sync_synchronize();
    read_idx.idx = (read_idx.idx + 1) & (kReleaseQueueCapacity - 1);

    return true;
}

void ReleaseQueue::PopFromReleaseQueue()
{
    if (initialized == false) {
        ASCEND_LOGE("Release queue is not initialized, shouldn't call PopFromReleaseQueue(). !!");
        return;
    }

    bool ret = false;
    while ((ret == false) && (GetStatus() != RepoStatus::CAN_EXIT)) {
        ret = ReadFromReleaseQueue();
        if (ret == false) {
            if (GetStatus() == RepoStatus::NEED_EXIT) {
                ChangeStatus(NEED_EXIT, CAN_EXIT);
                break;
            }
            delay.tv_usec = 1;
            select(0, nullptr, nullptr, nullptr, &delay);
        }
    }
}

void StartRelease(ReleaseQueue *releaseQue)
{
    SetThreadType(ThreadType::RELEASE_THREAD);
    SetThreadAffinity(releaseQue->GetDeviceID());

    while (releaseQue->GetStatus() != RepoStatus::CAN_EXIT) {
        releaseQue->PopFromReleaseQueue();
    }
    return;
}

void ReleaseQueue::InitReleaseQueue(c10::DeviceIndex device_id)
{
    if (datas == nullptr) {
        datas = releaseManager().Init(kReleaseQueueCapacity);
    }

    initialized = true;
    SetStatus(INIT);
    std::thread cur_releaser(StartRelease, this);
    releaser = std::move(cur_releaser);
    device_idx = device_id;
}

ReleaseQueue::~ReleaseQueue()
{
    if (initialized) {
        if (releaser.joinable()) {
            SetStatus(NEED_EXIT);
            releaser.join();
        }
    }
    releaseManager().DeInit(datas);
}

bool ReleaseQueue::IsFullQueue() const
{
    return ((write_idx.idx + 1) % kReleaseQueueCapacity) == read_idx.idx;
}

RepoStatus ReleaseQueue::GetStatus() const
{
    if (initialized == false) {
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
    if (initialized == false) {
        ASCEND_LOGE("Release queue is not initialized, shouldn't call SetStatus(). !!");
        return;
    }

    repo_status = desired;
}

void ReleaseQueue::ChangeStatus(RepoStatus expected, RepoStatus desired)
{
    if (initialized == false) {
        ASCEND_LOGE("Release queue is not initialized, shouldn't call ChangeStatus(). !!");
        return;
    }

    repo_status.compare_exchange_strong(expected, desired);
}
} // namespace c10_npu
