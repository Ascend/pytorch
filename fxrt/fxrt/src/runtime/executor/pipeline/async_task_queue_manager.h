#ifndef __RUNTIME_EXECUTOR_PIPELINE_ASYNC_TASK_QUEUE_MANAGER_H__
#define __RUNTIME_EXECUTOR_PIPELINE_ASYNC_TASK_QUEUE_MANAGER_H__

#include <set>
#include "runtime/executor/pipeline/async_task_queue.h"
#include "hardware/hardware_abstract/device_context.h"
#include "common/common.h"

namespace fxrt {
namespace runtime {
// Singleton AsyncTaskQueueManager manages multi-stage asynchronous processing tasks
// Uses lock-free queues for thread-safe operations between stages: infer -> launch.
class AsyncTaskQueueManager {
 public:
  static AsyncTaskQueueManager& GetInstance();

  AsyncTaskQueue* GetInferQueue() {
    return inferQueue_.get();
  }
  AsyncTaskQueue* GetLaunchQueue() {
    return launchQueue_.get();
  }

  void InitializeAll();

  // Suspends all pipeline queue, can not push element to a queue which is in pause status.
  void PauseAll();
  // Continue all pipeline queue which is in pause status.
  void ContinueAll();

  // Blocks until all queued tasks complete processing.
  void WaitAll();
  // Waits for worker threads to terminate (shutdown sequence).
  void WorkerJoin();

  void AddDeviceContext(const device::DeviceContext* deviceContext);

  const std::set<const device::DeviceContext*>& GetAllDeviceContexts() const;

  // Bind device and set device context for async pipeline threads.
  void BindDevice();

 private:
  AsyncTaskQueueManager();
  ~AsyncTaskQueueManager() = default;
  DISABLE_COPY_AND_ASSIGN(AsyncTaskQueueManager);

  AsyncTaskQueuePtr inferQueue_;
  AsyncTaskQueuePtr launchQueue_;
  std::set<const device::DeviceContext*> deviceContexts_;
};
} // namespace runtime
} // namespace fxrt
#endif // __RUNTIME_EXECUTOR_PIPELINE_ASYNC_TASK_QUEUE_MANAGER_H__
