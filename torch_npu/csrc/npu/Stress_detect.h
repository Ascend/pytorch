#ifndef STRESS_DETECT_H
#define STRESS_DETECT_H

#include <unordered_map>
#include <chrono>
#include <thread>
#include <atomic>
#include <future>
#include <condition_variable>
#include <sys/prctl.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

class StressDetector {
public:
    TORCH_NPU_API static int perform_stress_detect(int deviceid);
    TORCH_NPU_API static void stop_worker_thread();

private:
    static void worker_thread();

    // Records the last call time for each device
    static std::unordered_map<int, std::chrono::time_point<std::chrono::steady_clock>> last_call_times;
    
    // Thread for handling the stress detection task
    static std::thread stress_detect_thread;

    // Condition variable and mutex to control the thread
    static std::condition_variable cv;
    static std::mutex mtx;
    
    // Flag to indicate if a task is in progress
    static std::atomic<bool> task_in_progress;
    
    // Flag to signal the thread to stop
    static std::atomic<bool> stop_thread;

    // Flag to indicate if a new task has been submitted
    static std::atomic<bool> new_task_submitted;

    // Promise and future for the task, used for synchronizing task results
    static std::promise<int> promise;
    static std::future<int> current_task_future;

    // Stores parameters related to the task
    static int device_id;
    static void* workspaceAddr;
    static size_t workspaceSize;

    // Interval between tasks
    static const int interval_time;

    // Flag to indicate if the thread has been initialized
    static std::atomic<bool> thread_initialized;
};

#endif // STRESS_DETECT_H