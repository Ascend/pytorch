#include "runtime/executor/pipeline/pipeline_executor.h"
#include <chrono>
#include <memory>
#include <map>
#include <utility>
#include "runtime/executor/pipeline/async_task_queue_manager.h"
#include "ops/utils/async.h"

namespace fxrt {
namespace runtime {

#if defined(__aarch64__)
#define CPU_RELAX() asm volatile("yield" ::: "memory")
#elif defined(__x86_64__)
#include <emmintrin.h>
#define CPU_RELAX() _mm_pause()
#else
#define CPU_RELAX()
#endif

PipelineExecutor::PipelineExecutor(
    const std::shared_ptr<std::vector<OpRunner>>& opRunners,
    const std::map<hardware::DeviceType, device::DeviceContext*>& deviceContexts,
    const ir::ValuePtr& output)
    : Executor(opRunners, deviceContexts, output), initialized_(false) {}

void PipelineExecutor::Initialize() {
  if (initialized_) {
    return;
  }
  auto& asyncTaskQueueManager = AsyncTaskQueueManager::GetInstance();
  asyncTaskQueueManager.InitializeAll();
  asyncTaskQueueManager.ContinueAll();
  for (const auto& item : deviceContexts_) {
    asyncTaskQueueManager.AddDeviceContext(item.second);
  }
  asyncTaskQueueManager.BindDevice();
  asyncTaskQueueManager.PauseAll();

  initialized_ = true;
}

void PipelineExecutor::TryWaitLastRunFinish() {
  constexpr size_t spinCount = 1000;
  constexpr size_t waitLastRunMaxTime = 1000;

  if (!waitLastRunFinish_.load(std::memory_order_acquire)) {
    return;
  }

  for (size_t i = 0; i < spinCount && waitLastRunFinish_.load(std::memory_order_acquire); ++i) {
    CPU_RELAX();
  }
  if (!waitLastRunFinish_.load(std::memory_order_acquire)) {
    return;
  }

  std::unique_lock<std::mutex> lock(mtx_);
  if (cv_.wait_for(lock, std::chrono::milliseconds(waitLastRunMaxTime), [this] {
        return !waitLastRunFinish_.load(std::memory_order_acquire);
      })) {
    return;
  }

  try {
    WaitLaunchTaskFinish();
  } catch (...) {
    waitLastRunFinish_.store(false, std::memory_order_release);
    throw;
  }
  waitLastRunFinish_.store(false, std::memory_order_release);
}

void PipelineExecutor::Run(bool isDynamic) {
  RT_VLOG(VL_RUNTIME) << "Begin pipeline executor run.";
  TryWaitLastRunFinish();

  auto& launchOpFunc = ops::OpAsync::GetLaunchOpFunc();
  if (launchOpFunc == nullptr) {
    std::ostringstream oss;
    for (auto iter = deviceContexts_.begin(); iter != deviceContexts_.end(); ++iter) {
      oss << " " << hardware::GetDeviceNameByType(iter->first);
    }
    RT_GLOG(EXCEPTION) << "Device" << oss.str() << " does not suppprt pipeline executor.";
  }

  auto bindTask = [this]() -> int {
    for (auto iter = deviceContexts_.begin(); iter != deviceContexts_.end(); ++iter) {
      iter->second->deviceResManager_->BindDeviceToCurrentThread(false);
    }
    return ops::SUCCESS;
  };
  launchOpFunc("bind_device", bindTask, false);

  OpRunner* opRunners = opRunners_->data();
  size_t opNum = opRunners_->size();

  waitLastRunFinish_.store(true, std::memory_order_release);

  for (size_t i = 0; i < opNum; ++i) {
    OpRunner& opRunner = opRunners[i];

    opRunner.UpdateTensors();
    if (auto errNo = opRunner.InferShape() != ops::SUCCESS) {
      RT_GLOG(EXCEPTION) << "Infer shape failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
    }
    opRunner.AllocateMemory();
    if (auto errNo = opRunner.CalcWorkspace() != ops::SUCCESS) {
      RT_GLOG(EXCEPTION) << "CalcWorkspace shape failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
    }
    opRunner.AllocateWorkspaceMemory();
    opRunner.FreeMemory();

    if (!opRunner.NeedLaunch()) {
      continue;
    }

    auto launchTask = [&opRunner]() -> int {
      if (auto errNo = opRunner.Launch() != ops::SUCCESS) {
        RT_GLOG(EXCEPTION) << "Launch failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
      }
      return 0;
    };
    launchOpFunc(opRunner.GetOpName(), launchTask, false);
  }

  auto launchKernelsFinish = [this]() -> int {
    waitLastRunFinish_.store(false, std::memory_order_release);
    std::lock_guard<std::mutex> lock(mtx_);
    cv_.notify_one();
    return 0;
  };
  launchOpFunc("launch_kernels_finish", launchKernelsFinish, false);

  RT_VLOG(VL_RUNTIME) << "End pipeline executor run.";
}
} // namespace runtime
} // namespace fxrt
