#include "Stress_detect.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

std::unordered_map<int, std::chrono::time_point<std::chrono::steady_clock>> StressDetector::last_call_times;
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
const int StressDetector::interval_time = 3600;

// Persistent worker thread implementation
void StressDetector::worker_thread()
{
    while (!stop_thread.load()) {
        std::unique_lock<std::mutex> lock(mtx);

        // Wait for new task submission or thread stop signal
        cv.wait(lock, [] { return new_task_submitted.load() || stop_thread.load(); });

        if (stop_thread.load()) {
            return; // Exit thread
        }

        // Execute the task
        int ret = c10_npu::acl::AclStressDetect(device_id, workspaceAddr, workspaceSize);

        // Task complete, free memory
        aclrtFree(workspaceAddr);

        // Set task result and reset flags
        task_in_progress.store(false);
        promise.set_value(ret);  // Pass the task execution result

        // Reset task submission flag
        new_task_submitted.store(false);
    }
}

// Synchronous stress detection task execution
int StressDetector::perform_stress_detect(int deviceid)
{
    auto current_time = std::chrono::steady_clock::now();
    // Check the calling interval
    if (last_call_times.find(deviceid) != last_call_times.end() &&
        std::chrono::duration_cast<std::chrono::seconds>(current_time - last_call_times[deviceid]).count() < interval_time) {
        ASCEND_LOGW("StressDetect can only be called once every hour for the given deviceid:{%d}, Return 1.", deviceid);
        return 1;
    }
    last_call_times[deviceid] = current_time;

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
    uint64_t size = 2;
    workspaceSize = size << 10 << 10 << 10;  // Assume memory size
    if (workspaceSize > 0) {
        auto ret = c10_npu::acl::AclrtMallocAlign32(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_ERROR_NONE) {
            c10_npu::NPUCachingAllocator::emptyCache();
            ret = c10_npu::acl::AclrtMallocAlign32(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_ERROR_NONE) {
                ASCEND_LOGW("call AclrtMallocAlign32 failed, ERROR : %d. Skip StressDetect.", ret);
                task_in_progress.store(false); // Task ends
                return ACL_ERROR_NONE;
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

        // Mark new task submitted
        new_task_submitted.store(true);
    }

    // Notify the persistent thread to start the task
    cv.notify_one();

    // Synchronously wait for the task to complete and get the result
    int ret = current_task_future.get();

    return ret;
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