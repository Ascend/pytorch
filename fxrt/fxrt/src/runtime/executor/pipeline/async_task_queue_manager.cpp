#include "runtime/executor/pipeline/async_task_queue_manager.h"
#include <memory>
#include "runtime/utils/exception.h"
#include "runtime/utils/gil_scoped.h"

namespace fxrt {
namespace runtime {
AsyncTaskQueueManager& AsyncTaskQueueManager::GetInstance() {
  static AsyncTaskQueueManager instance;
  return instance;
}

AsyncTaskQueueManager::AsyncTaskQueueManager()
    : inferQueue_(std::make_unique<AsyncTaskQueue>("infer_queue")),
      launchQueue_(std::make_unique<AsyncTaskQueue>("launch_queue")) {}

void AsyncTaskQueueManager::InitializeAll() {
  inferQueue_->Initialize();
  launchQueue_->Initialize();
}

void AsyncTaskQueueManager::PauseAll() {
  GilReleaseWithCheck gil_release;
  inferQueue_->Pause();
  launchQueue_->Pause();
}

void AsyncTaskQueueManager::ContinueAll() {
  inferQueue_->Continue();
  launchQueue_->Continue();
}

void AsyncTaskQueueManager::WaitAll() {
  GilReleaseWithCheck gil_release;
  inferQueue_->Wait();
  launchQueue_->Wait();
  FxrtException::GetInstance().CheckException();
}

void AsyncTaskQueueManager::WorkerJoin() {
  GilReleaseWithCheck gil_release;
  inferQueue_->WorkerJoin();
  launchQueue_->WorkerJoin();
}

void AsyncTaskQueueManager::AddDeviceContext(const device::DeviceContext* deviceContext) {
  CHECK_IF_NULL(deviceContext);
  (void)deviceContexts_.insert(deviceContext);
}

const std::set<const device::DeviceContext*>& AsyncTaskQueueManager::GetAllDeviceContexts() const {
  return deviceContexts_;
}

void AsyncTaskQueueManager::BindDevice() {
  inferQueue_->BindDevice(deviceContexts_);
  launchQueue_->BindDevice(deviceContexts_);
}

} // namespace runtime
} // namespace fxrt
