#include "runtime/executor/pipeline/async_task_queue.h"
#include "runtime/utils/exception.h"
#include "common/common.h"

namespace fxrt {
namespace runtime {
constexpr size_t kThreadNameThreshold = 15;

void AsyncTaskQueue::SetThreadName() const {
  // Set thread name to monitor thread status or gdb debug.
  (void)pthread_setname_np(pthread_self(), name_.substr(0, kThreadNameThreshold).c_str());
}

AsyncTaskQueue::AsyncTaskQueue(std::string name) : name_(std::move(name)) {
  worker_ = std::make_unique<std::thread>(&AsyncTaskQueue::WorkerLoop, this);
}

AsyncTaskQueue::~AsyncTaskQueue() {
  try {
    WorkerJoin();
  } catch (const std::exception& e) {
    RT_GLOG(ERROR) << "WorkerJoin failed, error msg:" << e.what();
  }
}

void AsyncTaskQueue::WorkerLoop() {
  SetThreadName();

  while (alive_.load(std::memory_order_relaxed)) {
    auto* task = tasksQueue_.Front();
    if (task == nullptr) {
      return;
    }

    try {
      task->func_();
      tasksQueue_.Pop();
    } catch (const std::exception& e) {
      FxrtException::GetInstance().SetException();
      RT_GLOG(ERROR) << "Run task failed and catch exception: " << e.what();
      while (!tasksQueue_.Empty()) {
        auto* remainingTask = tasksQueue_.Front();
        if (remainingTask != nullptr && remainingTask->type_ == TaskType::Wait) {
          remainingTask->func_();
        }
        tasksQueue_.Pop();
      }
    }
  }
}

void AsyncTaskQueue::Initialize() {
  if (init_) {
    return;
  }

  if (worker_ == nullptr) {
    worker_ = std::make_unique<std::thread>(&AsyncTaskQueue::WorkerLoop, this);
  }
  init_ = true;
}

void AsyncTaskQueue::BindDevice(const std::set<const device::DeviceContext*>& deviceContexts) {
  auto bind_device_task = [&deviceContexts]() {
    std::for_each(deviceContexts.begin(), deviceContexts.end(), [](const device::DeviceContext* item) {
      item->deviceResManager_->BindDeviceToCurrentThread(false);
    });
  };
  Push(std::move(bind_device_task), TaskType::BindDevice);
  Wait();
  FxrtException::GetInstance().CheckException();
}

void AsyncTaskQueue::Wait() {
  if (!init_ || worker_ == nullptr) {
    return;
  }
  if (worker_->get_id() == std::this_thread::get_id()) {
    return;
  }

  std::atomic<bool> atomicWaitFlag = false;
  auto waitTask = [&atomicWaitFlag]() { atomicWaitFlag.store(true, std::memory_order_release); };
  Push(std::move(waitTask), TaskType::Wait);

  // waitTask is run by the worker thread. WorkerJoin is the only path on which
  // the worker stops without running the pushed tasks, and it clears alive_, so
  // this wait always has an exit.
  while (!atomicWaitFlag.load(std::memory_order_acquire) && alive_.load(std::memory_order_relaxed)) {
    std::this_thread::yield();
  }
}

bool AsyncTaskQueue::Empty() const {
  return tasksQueue_.Empty();
}

void AsyncTaskQueue::Pause() {
  if (!init_) {
    return;
  }

  if (tasksQueue_.IsPaused()) {
    // Has been paused already.
    return;
  }

  Wait();
  tasksQueue_.Pause();
}

void AsyncTaskQueue::Continue() {
  if (!init_) {
    return;
  }
  tasksQueue_.Continue();
}

std::thread::id AsyncTaskQueue::GetThreadID() const {
  CHECK_IF_NULL(worker_);
  return worker_->get_id();
}

void AsyncTaskQueue::WorkerJoin() {
  if (worker_ == nullptr) {
    return;
  }
  if (init_) {
    while (!Empty()) {
    }
  }

  alive_.store(false, std::memory_order_release);
  tasksQueue_.Finalize();

  if (worker_->joinable()) {
    worker_->join();
  }
  worker_ = nullptr;
}
} // namespace runtime
} // namespace fxrt
