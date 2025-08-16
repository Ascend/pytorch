#include "Stress_detect.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/interface/MlInterface.h"

std::atomic<bool> StressDetector::task_in_progress(false);
std::atomic<bool> StressDetector::stop_thread(false);
std::atomic<bool> StressDetector::new_task_submitted(false);
std::atomic<bool> StressDetector::thread_initialized(false);
std::promise<int> StressDetector::promise;
std::future<int> StressDetector::current_task_future;
std::thread StressDetector::stress_detect_thread;
std::condition_variable StressDetector::cv;
std::mutex StressDetector::mtx;

int StressDetector::device_id;
void* StressDetector::workspaceAddr = nullptr;
size_t StressDetector::workspaceSize = 0;
int StressDetector::stressMode = 0;
void* StressDetector::localHcclComm = nullptr;

constexpr int kDetectSucceeded = 0;
constexpr int kDetectFailed = 1;
constexpr int kDetectFailedWithHardwareFailure = 2;

// Persistent worker thread implementation
void StressDetector::worker_thread()
{
    if (prctl(PR_SET_NAME, ("StressDetect_thread")) != 0) {
        ASCEND_LOGE("set thread name failed!");
    }

    while (!stop_thread.load()) {
        std::unique_lock<std::mutex> lock(mtx);

        // Wait for new task submission or thread stop signal
        cv.wait(lock, [] { return new_task_submitted.load() || stop_thread.load(); });

        if (stop_thread.load()) {
            return; // Exit thread
        }

        // Execute the task
        int ret = -1;
        try {
            if (stressMode == 0) {
                if (c10_npu::amlapi::IsExistAmlAicoreDetectOnline()) {
                    AmlAicoreDetectAttr attr;
                    attr.mode = AML_DETECT_RUN_MODE_ONLINE;
                    attr.workspace = workspaceAddr;
                    attr.workspaceSize = workspaceSize;
                    ret = c10_npu::amlapi::AmlAicoreDetectOnlineFace(device_id, &attr);
                    ASCEND_LOGI("Stress detect with AmlAicoreDetectOnline, device id is %d, result is %d.", device_id, ret);
                } else {
                    ret = c10_npu::acl::AclStressDetect(device_id, workspaceAddr, workspaceSize);
                    ASCEND_LOGI("Stress detect with StressDetect, device id is %d, result is %d.", device_id, ret);
                }
            } else {
                if (c10_npu::amlapi::IsExistAmlP2PDetectOnline()) {
                    AmlP2PDetectAttr attr;
                    attr.workspace = workspaceAddr;
                    attr.workspaceSize = workspaceSize;
                    ret = c10_npu::amlapi::AmlP2PDetectOnlineFace(device_id, localHcclComm, &attr);
                    ASCEND_LOGI("Stress detect with AmlP2PDetectOnline, device id is %d, result is %d.", device_id, ret);
                } else {
                    ASCEND_LOGW("Stress detect with AmlP2PDetectOnline failed, CANN version lower than 8.3.RC1 and currently does not support AmlP2PDetectOnline.");
                    TORCH_NPU_WARN("Stress detect with AmlP2PDetectOnline failed, CANN version lower than 8.3.RC1 and currently does not support AmlP2PDetectOnline.");
                }
            }
        } catch (std::exception &e) {
            ret = -1;
            ASCEND_LOGW("Stress detect failed, device id is %d, type is %d, error:%s", device_id, stressMode, e.what());
            TORCH_NPU_WARN("Stress detect failed, type is ", stressMode, ", error: ", e.what());
        }

        // Task complete, free memory
        aclrtFree(workspaceAddr);

        // Set task result and reset flags
        task_in_progress.store(false);
        promise.set_value(ret);  // Pass the task execution result

        // Reset task submission flag
        new_task_submitted.store(false);
    }
}

int StressDetector::transfer_result(int detectResult)
{
    int ret = kDetectFailed;
    switch (detectResult) {
        case 0:
            ret = kDetectSucceeded;
            ASCEND_LOGI("Stress detect test case execution succeeded, device id is %d.", device_id);
            break;
        case ACLNN_STRESS_BIT_FAIL:
        case ACLNN_STRESS_LOW_BIT_FAIL:
        case ACLNN_STRESS_HIGH_BIT_FAIL:
            ret = kDetectFailedWithHardwareFailure;
            ASCEND_LOGW("Stress detect failed due to hardware malfunction, device id is %d, error code is %d.", device_id, detectResult);
            TORCH_NPU_WARN("Stress detect failed due to hardware malfunction, device id is ", device_id, ", error code is ", detectResult);
            break;
        case ACLNN_CLEAR_DEVICE_STATE_FAIL:
            ASCEND_LOGW("Stress detect error. Error code is 574007. Error message is Voltage recovery failed, device id is %d.", device_id);
            TORCH_CHECK(false, "Stress detect error. Error code is 574007. Error message is Voltage recovery failed.", PTA_ERROR(ErrCode::ACL));
            break;
        default:
            ret = kDetectFailed;
            ASCEND_LOGW("Stress detect test case execution failed, device id is %d, error code is %d.", device_id, detectResult);
            TORCH_NPU_WARN("Stress detect test case execution failed, device id is ", device_id, ", error code is ", detectResult);
            break;
    }
    return ret;
}

// Synchronous stress detection task execution
int StressDetector::perform_stress_detect(int deviceid, int mode, int64_t comm)
{
    // If it's the first call, start the persistent thread
    if (!thread_initialized.load()) {
        std::lock_guard<std::mutex> lock(mtx);  // Ensure thread safety
        if (!thread_initialized.load()) {  // Double check
            stress_detect_thread = std::thread(worker_thread);
            thread_initialized.store(true);  // Mark thread as started
        }
    }

    // Set task parameters
    task_in_progress.store(true);
    
    // Allocate workspace memory
    workspaceAddr = nullptr;
    uint64_t size = 10;
    workspaceSize = size << 10 << 10 << 10;  // Assume memory size
    if (workspaceSize > 0) {
        auto ret = c10_npu::acl::AclrtMallocAlign32(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_ERROR_NONE) {
            c10_npu::NPUCachingAllocator::emptyCache();
            ret = c10_npu::acl::AclrtMallocAlign32(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_ERROR_NONE) {
                ASCEND_LOGW("Stress detect failed, call AclrtMallocAlign32 failed, device id is %d, ERROR : %d. Skip StressDetect.", deviceid, ret);
                TORCH_NPU_WARN("Stress detect failed, call AclrtMallocAlign32 failed, skip StressDetect, device id is ", deviceid, ", error is ", ret);
                task_in_progress.store(false); // Task ends
                return kDetectFailed;
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(mtx);

        // Prepare promise and future
        promise = std::promise<int>();
        current_task_future = promise.get_future();

        // Update task-related information
        StressDetector::device_id = deviceid;
        StressDetector::workspaceAddr = workspaceAddr;
        StressDetector::workspaceSize = workspaceSize;
        StressDetector::stressMode = mode;
        StressDetector::localHcclComm = reinterpret_cast<void*>(static_cast<intptr_t>(comm));

        // Mark new task submitted
        new_task_submitted.store(true);
    }

    // Notify the persistent thread to start the task
    cv.notify_one();

    // Synchronously wait for the task to complete and get the result
    int ret = current_task_future.get();

    return transfer_result(ret);
}

// Stop the thread
void StressDetector::stop_worker_thread()
{
    {
        std::lock_guard<std::mutex> lock(mtx);
        stop_thread.store(true);
    }
    cv.notify_one(); // Notify the thread to exit
    if (stress_detect_thread.joinable()) {
        stress_detect_thread.join(); // Wait for the thread to exit
    }
}