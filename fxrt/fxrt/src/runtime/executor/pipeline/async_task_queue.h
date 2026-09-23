#ifndef __RUNTIME_EXECUTOR_PIPELINE_ASYNC_TASK_QUEUE_H__
#define __RUNTIME_EXECUTOR_PIPELINE_ASYNC_TASK_QUEUE_H__

#include <atomic>
#include <memory>
#include <thread>
#include <string>
#include <set>
#include <utility>
#include <functional>
#include <cstddef>

#include "hardware/hardware_abstract/device_context.h"
#include "runtime/executor/pipeline/lock_free_ring_queue.h"
#include "common/common.h"

namespace fxrt {
namespace runtime {
constexpr uint64_t kLFQueueCapacity = 8192;

enum class TaskType : uint8_t { Wait = 0, Infer = 1, Launch = 2, BindDevice = 3, Other = 4 };

struct AsyncTask {
 public:
  template <typename F>
  AsyncTask(F&& func, TaskType type) : func_(std::forward<F>(func)), type_(type) {}
  ~AsyncTask() = default;
  AsyncTask(AsyncTask&& other) noexcept : func_(std::move(other.func_)), type_(other.type_) {}
  AsyncTask& operator=(AsyncTask&& other) noexcept {
    if (this != &other) {
      func_ = std::move(other.func_);
      type_ = other.type_;
    }
    return *this;
  }
  DISABLE_COPY_AND_ASSIGN(AsyncTask)

  std::function<void()> func_;
  TaskType type_;
};

// AsyncTaskQueue is a lock-free asynchronous queue that supports multiple producers and a single consumer. It
// internally starts a thread to act as the consumer, allowing concurrent pushing of elements. The elements must be of a
// type that can be constructed into an std::function<void()> object.
class AsyncTaskQueue {
 public:
  explicit AsyncTaskQueue(std::string name);
  ~AsyncTaskQueue();

  void Initialize();

  // Bind device and set device context for async pipeline thread.
  void BindDevice(const std::set<const device::DeviceContext*>& deviceContexts);

  // Push element to lock free queue, the args parameter must be of type std::function<void()> or convertible to this
  // type. Push is multi thread safety.
  template <typename... Args>
  void Push(Args&&... args) {
    if (!init_ || worker_ == nullptr) {
      RT_GLOG(EXCEPTION) << "The queue is not initialized before.";
    }
    if (!alive_.load(std::memory_order_relaxed)) {
      return;
    }
    if (!tasksQueue_.Push(std::forward<Args>(args)...)) {
      RT_GLOG(EXCEPTION) << "Failed to push task to queue: " << name_;
    }
  }

  // Wait for all async task finish executing.
  void Wait();

  // Check the queue is empty or not.
  bool Empty() const;

  // Pause the queue, can not push element to a queue which is in pause status, AsyncTaskQueue is in pause status after
  // creation.
  void Pause();

  // Continue the queue which is in pause status.
  void Continue();

  // Thread join before the process exit.
  void WorkerJoin();

  std::thread::id GetThreadID() const;
  const std::unique_ptr<std::thread>& GetWorker() const {
    return worker_;
  }

 private:
  void WorkerLoop();
  void SetThreadName() const;
  std::unique_ptr<std::thread> worker_;
  std::string name_;

  LockFreeRingQueue<AsyncTask, kLFQueueCapacity> tasksQueue_;
  bool init_{false};
  // Read by the threads that Wait on the queue, written by WorkerJoin.
  std::atomic<bool> alive_{true};
};
using AsyncTaskQueuePtr = std::unique_ptr<AsyncTaskQueue>;
} // namespace runtime
} // namespace fxrt
#endif // __RUNTIME_EXECUTOR_PIPELINE_ASYNC_TASK_QUEUE_H__
