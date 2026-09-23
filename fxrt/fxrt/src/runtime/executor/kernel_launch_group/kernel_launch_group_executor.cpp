#include "runtime/executor/kernel_launch_group/kernel_launch_group_executor.h"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <map>
#include <memory>
#include <vector>
#include "runtime/executor/pipeline/async_task_queue_manager.h"
#include "runtime/utils/exception.h"
#include "hardware/hardware_abstract/device_context_manager.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"
#include "ops/utils/async.h"

namespace fxrt {
namespace runtime {
std::vector<std::pair<size_t, void*>> KernelLaunchGroupExecutor::streams_{};
std::vector<DeviceEventPtr> KernelLaunchGroupExecutor::events_{};
std::vector<DeviceEventPtr> KernelLaunchGroupExecutor::eventsToDefaultStream_{};
std::vector<AsyncTaskQueuePtr> KernelLaunchGroupExecutor::queues_{};

KernelLaunchGroupExecutor::KernelLaunchGroupExecutor(
    const std::shared_ptr<std::vector<OpRunner>>& opRunners,
    const std::map<hardware::DeviceType, device::DeviceContext*>& deviceContexts,
    const std::shared_ptr<std::vector<std::pair<OpRunner*, size_t>>>& opRunnerGroups,
    const std::shared_ptr<std::vector<OpRunner*>>& serialLaunchOps,
    const std::shared_ptr<std::vector<ir::TensorPtr>>& graphInputTensors,
    const std::shared_ptr<std::vector<std::pair<ir::TensorPtr, std::vector<int64_t>>>>&
        graphInputTensorsWithDynamicShape,
    const std::shared_ptr<std::unordered_set<ir::Tensor*>>& graphOutputs,
    uint64_t parallelDispatchNum,
    uint64_t parallelSliceNum,
    const ir::ValuePtr& output)
    : PipelineExecutor(opRunners, deviceContexts, output),
      opRunnerGroups_(opRunnerGroups),
      serialLaunchOps_(serialLaunchOps),
      graphInputTensors_(graphInputTensors),
      graphInputTensorsWithDynamicShape_(graphInputTensorsWithDynamicShape),
      graphOutputs_(graphOutputs),
      parallelDispatchNum_(parallelDispatchNum),
      parallelSliceNum_(parallelSliceNum),
      memoryCache_(),
      initialized_(false) {
  device::DeviceContextKey deviceContextKey = device::DeviceToDeviceContextKey(
      {hardware::DeviceType::NPU, Uint32ToInt8(fxrt::collective::CollectiveManager::Instance().local_rank_id())});
  deviceContext_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(deviceContextKey);
}

KernelLaunchGroupExecutor::~KernelLaunchGroupExecutor() {
  memoryCache_.ClearAllCache();
}

void KernelLaunchGroupExecutor::Run(bool isDynamic) {
  WaitLaunchTaskFinish();

  UpdateInputTensors();
  bool shapeChange = CheckInputShapeChange();
  if (shapeChange) {
    ResetTensorCacheMemory();
    memoryCache_.ClearExpiredCache();
    static const size_t memoryBlockSize = 3000;
    memoryCache_.ReserveKernelMemoryBlocks(memoryBlockSize, deviceContext_);

    RunWithRecordCacheMemory();
  } else {
    AllocateGraphCacheMemory();
    // Real parallel dispatch with use memory trace.
    AllocateGraphOutputMemory();
    ParallelDispatchKernels();
    FreeGraphCacheMemory();
  }
}

void KernelLaunchGroupExecutor::Initialize() {
  RT_VLOG(VL_RUNTIME) << "Begin initialize";
  if (initialized_) {
    RT_GLOG(EXCEPTION) << "KernelLaunchGroupExecutor has been initialized, can not support multi graph.";
  }
  PipelineExecutor::Initialize();

  if (streams_.empty()) {
    streams_.resize(parallelDispatchNum_);
    for (size_t i = 0; i < parallelDispatchNum_; i++) {
      if (!deviceContext_->deviceResManager_->CreateStream(&(streams_[i].first))) {
        RT_GLOG(EXCEPTION) << "Create stream failed.";
      }
      streams_[i].second = deviceContext_->deviceResManager_->GetStream(streams_[i].first);
      CHECK_IF_NULL(streams_[i].second);
    }
  }

  if (events_.empty()) {
    // New one more for sync between default stream and last launch stream;
    for (size_t i = 0; i < parallelDispatchNum_ * parallelSliceNum_ + 1; i++) {
      auto event = deviceContext_->deviceResManager_->CreateEventWithFlag(false, false, false);
      CHECK_IF_NULL(event);
      events_.push_back(event);
    }
  }

  if (eventsToDefaultStream_.empty()) {
    for (size_t i = 0; i < streams_.size(); i++) {
      auto event = deviceContext_->deviceResManager_->CreateEventWithFlag(false, false, true);
      CHECK_IF_NULL(event);
      eventsToDefaultStream_.push_back(event);
    }
  }

  if (queues_.empty()) {
    for (size_t i = 0; i < parallelDispatchNum_; i++) {
      auto queue = std::make_unique<AsyncTaskQueue>(std::string("batch_launch_") + std::to_string(i));
      CHECK_IF_NULL(queue);
      queue->Initialize();
      queues_.push_back(std::move(queue));
    }
  }

  const size_t kEventNum = 2;
  CHECK_IF_NULL(serialLaunchOps_);
  for (auto* opRunner : *serialLaunchOps_) {
    serialLaunchKernelsToEvents_[opRunner] = std::vector<DeviceEventPtr>(kEventNum, nullptr);
  }

  for (auto& item : serialLaunchKernelsToEvents_) {
    auto& eventArray = item.second;
    for (size_t i = 0; i < eventArray.size(); i++) {
      auto event = deviceContext_->deviceResManager_->CreateEventWithFlag(false, false, false);
      CHECK_IF_NULL(event);
      eventArray[i] = event;
    }
  }

  initialized_ = true;
  RT_VLOG(VL_RUNTIME) << "End initialize";
}

void KernelLaunchGroupExecutor::UpdateInputTensors() {
  CHECK_IF_NULL(graphInputTensors_);
  for (auto& tensor : *graphInputTensors_) {
    CHECK_IF_NULL(tensor);
    tensor->Update();
  }
}

bool KernelLaunchGroupExecutor::CheckInputShapeChange() {
  bool shapeChange = false;
  CHECK_IF_NULL(graphInputTensorsWithDynamicShape_);
  auto& graphInputTensorsWithDynamicShape = *graphInputTensorsWithDynamicShape_;
  RT_VLOG(VL_RUNTIME) << "Dynamic input tensor number: " << graphInputTensorsWithDynamicShape.size();

  // Disable parallel dispatch for static shape case.
  if (graphInputTensorsWithDynamicShape.empty()) {
    return true;
  }

  for (auto& [tensor, shape] : graphInputTensorsWithDynamicShape) {
    // tensor->Shape() is current input shape, shape is the recorded shape from last execution.
    if (tensor->Shape() != shape || tensor->Shape().empty()) {
      shapeChange = true;
      // Update recorded shape only when actually different, avoiding redundant vector deep-copy.
      shape = tensor->Shape();
    }
  }
  RT_VLOG(VL_RUNTIME) << "Input tensor shape changed: " << shapeChange;
  return shapeChange;
}

void KernelLaunchGroupExecutor::ResetTensorCacheMemory() {
  std::unordered_map<OpRunner*, std::vector<KernelMemoryTraceBlockPtr>>& allKernelBlockInfo =
      *(memoryCache_.GetAllKernelBlocksInfo());
  for (auto& iter : allKernelBlockInfo) {
    if (iter.first->GetWorkspace() != nullptr) {
      iter.first->SetWorkspace(nullptr);
    }
    const auto& kernelMemBlock = iter.second;
    for (auto& block : kernelMemBlock) {
      CHECK_IF_NULL(block);
      if (block->memType_ == kOutputMem) {
        const auto* tensor = block->tensor_;
        CHECK_IF_NULL(tensor);
        tensor->GetStorage()->SetData(nullptr);
      }
    }
  }
}

void KernelLaunchGroupExecutor::RunWithRecordCacheMemory() {
  RT_VLOG(VL_RUNTIME) << "Begin pipeline executor run.";
  auto& asyncTaskQueueManager = AsyncTaskQueueManager::GetInstance();
  asyncTaskQueueManager.ContinueAll();

  AsyncTaskQueue* inferQueue = asyncTaskQueueManager.GetInferQueue();
  AsyncTaskQueue* launchQueue = asyncTaskQueueManager.GetLaunchQueue();
  OpRunner* opRunners = opRunners_->data();
  size_t opNum = opRunners_->size();
  for (size_t i = 0; i < opNum; ++i) {
    OpRunner& opRunner = opRunners[i];

    auto inferTask = [&opRunner, launchQueue, this]() {
      opRunner.UpdateTensors();
      // Do infer shape and calculate workspace size in infer queue.
      if (auto errNo = opRunner.InferShape() != ops::SUCCESS) {
        RT_GLOG(EXCEPTION) << "Infer shape failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
      }
      if (auto errNo = opRunner.CalcWorkspace() != ops::SUCCESS) {
        RT_GLOG(EXCEPTION) << "CalcWorkspace shape failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
      }

      // Push async launch task into launch queue.
      auto launchTask = [&opRunner, this]() {
        opRunner.AllocateMemory();
        RecordMemory(&opRunner, opRunner.GetOutput());
        if (auto errNo = opRunner.Launch() != ops::SUCCESS) {
          RT_GLOG(EXCEPTION) << "Launch failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
        }
        opRunner.FreeMemory();
      };
      launchQueue->Push(std::move(launchTask), TaskType::Launch);
    };

    inferQueue->Push(std::move(inferTask), TaskType::Infer);
  }

  asyncTaskQueueManager.WaitAll();
  asyncTaskQueueManager.PauseAll();

  memoryCache_.MergeBlocks();
  RT_VLOG(VL_RUNTIME) << "End pipeline executor run.";
}

void KernelLaunchGroupExecutor::RecordMemory(OpRunner* opRunner, const ir::Value* value) {
  CHECK_IF_NULL(value);
  CHECK_IF_NULL(graphOutputs_);

  if (value->IsTensor()) {
    auto& tensor = value->ToTensor();
    CHECK_IF_NULL(tensor);
    if ((graphOutputs_->find(tensor.get()) == graphOutputs_->end()) && tensor->GetStorage()->CheckOwnsData()) {
      memoryCache_.AddKernelMemoryTraceBlock(
          std::make_shared<KernelMemoryTraceBlock>(
              opRunner,
              tensor->GetStorage()->Data(),
              tensor->GetStorage()->SizeBytes(),
              kOutputMem,
              0,
              tensor.get(),
              deviceContext_),
          deviceContext_);
    }
  } else if (value->IsTuple()) {
    auto& tuple = value->ToTuple();
    CHECK_IF_NULL(tuple);
    size_t size = tuple->Size();
    for (size_t i = 0; i < size; i++) {
      auto& value = (*tuple)[i];
      CHECK_IF_NULL(value);
      CHECK_IF_FAIL(!value->IsTensor());
      auto& tensor = value->ToTensor();
      // If enable kernel launch capture, the kernel output as graph output will be captured and can not changed, so
      // need trace the graph output kernel tensor device address, which device memory will be allocated and released
      // with the whole graph.
      if ((graphOutputs_->find(tensor.get()) != graphOutputs_->end()) || !tensor->GetStorage()->CheckOwnsData()) {
        continue;
      }
      memoryCache_.AddKernelMemoryTraceBlock(
          std::make_shared<KernelMemoryTraceBlock>(
              opRunner,
              tensor->GetStorage()->Data(),
              tensor->GetStorage()->SizeBytes(),
              kOutputMem,
              i,
              tensor.get(),
              deviceContext_),
          deviceContext_);
    }
  }

  if (opRunner->GetWorkspaceSize() > 0) {
    memoryCache_.AddKernelMemoryTraceBlock(
        std::make_shared<KernelMemoryTraceBlock>(
            opRunner,
            opRunner->GetWorkspace(),
            opRunner->GetWorkspaceSize(),
            kWorkspaceMem,
            0,
            nullptr,
            deviceContext_),
        deviceContext_);
  }
}

void KernelLaunchGroupExecutor::DispatchParallelLaunchKernels(size_t index) {
  if (index >= parallelDispatchNum_) {
    RT_GLOG(EXCEPTION) << "Invalid index: " << index << ", expected less than: " << parallelDispatchNum_;
  }
  deviceContext_->deviceResManager_->BindDeviceToCurrentThread(false);
  size_t realStreamId = streams_[index].first;
  void* realStream = streams_[index].second;

  for (size_t innerIndex = 0; innerIndex < parallelSliceNum_; innerIndex++) {
    events_[index + innerIndex * parallelDispatchNum_]->WaitEventWithoutReset(realStreamId);

    const std::pair<OpRunner*, size_t>& opRunners = (*opRunnerGroups_)[index + innerIndex * parallelDispatchNum_];

    auto* firstOpRunner = opRunners.first;
    size_t opNum = opRunners.second;
    for (size_t i = 0; i < opNum; ++i) {
      auto* opRunner = firstOpRunner + i;

      auto commuIter = serialLaunchKernelsToEvents_.find(opRunner);
      if (commuIter != serialLaunchKernelsToEvents_.end()) {
        const auto& eventArray = commuIter->second;
        auto& recordEvent = eventArray[0];
        auto& waitEvent = eventArray[1];
        recordEvent->RecordEvent(realStreamId);
        waitEvent->WaitEventWithoutReset(realStreamId);
        continue;
      }

      opRunner->UpdateTensors();
      SetOutputAndWsCacheMemory(opRunner);
      SetInputCacheMemory(opRunner);
      if (auto errNo = opRunner->Launch(realStream) != ops::SUCCESS) {
        RT_GLOG(EXCEPTION) << "Launch failed for operator " << opRunner->GetOpName() << "Errno: " << errNo;
      }
    }

    events_[index + innerIndex * parallelDispatchNum_ + 1]->RecordEvent(realStreamId);
  }
}

void KernelLaunchGroupExecutor::DispatchSerialLaunchKernels() {
  void* mainStream = deviceContext_->deviceResManager_->GetCurrentStream();
  CHECK_IF_NULL(mainStream);
  const auto& serialLaunchKernels = *serialLaunchOps_;
  for (auto* opRunner : serialLaunchKernels) {
    CHECK_IF_NULL(opRunner);
    auto iter = serialLaunchKernelsToEvents_.find(opRunner);
    if (iter == serialLaunchKernelsToEvents_.end()) {
      RT_GLOG(EXCEPTION) << "Not find event for operator  : " << opRunner->GetOpName();
    }

    opRunner->UpdateTensors();
    SetOutputAndWsCacheMemory(opRunner);
    SetInputCacheMemory(opRunner);

    const auto& eventArray = iter->second;
    auto& waitEvent = eventArray[0];
    waitEvent->set_wait_stream(mainStream);
    waitEvent->WaitEventWithoutReset();

    // TODO: force resize. // NOLINT(readability/todo)

    if (auto errNo = opRunner->Launch(mainStream) != ops::SUCCESS) {
      RT_GLOG(EXCEPTION) << "Launch failed for operator " << opRunner->GetOpName() << "Errno: " << errNo;
    }

    auto& recordEvent = eventArray[1];
    recordEvent->set_record_stream(mainStream);
    recordEvent->RecordEvent();
  }
}

void KernelLaunchGroupExecutor::ParallelDispatchKernels() {
  RT_VLOG(VL_RUNTIME) << "Begin parallel dispatch kernels";

  // Record a event to default stream to notify parallel launch kernels execute on other stream.
  void* mainStream = deviceContext_->deviceResManager_->GetCurrentStream();
  CHECK_IF_NULL(mainStream);
  events_.front()->set_record_stream(mainStream);
  events_.front()->RecordEvent();

  // Dispatch kernel which can parallel launch.
  for (size_t i = 0; i < parallelDispatchNum_; i++) {
    const auto& q = queues_[i];
    q->Continue();
    q->Push([this, i]() { DispatchParallelLaunchKernels(i); }, TaskType::Other);
  }

  // Dispatch serial launch kernels: communication ops and the kernel need force resize.
  DispatchSerialLaunchKernels();

  for (auto& q : queues_) {
    q->Pause();
  }
  FxrtException::GetInstance().CheckException();

  // The default stream need wait all parallel launch kernel execute finish.
  events_.back()->set_wait_stream(mainStream);
  events_.back()->WaitEventWithoutReset();
  // Reset all event for reuse.
  for (auto& e : events_) {
    e->ResetEvent();
  }
  for (auto& item : serialLaunchKernelsToEvents_) {
    for (auto& e : item.second) {
      e->ResetEvent();
    }
  }
  // It must be ensured that the reset at the tail is executed within the step, an event create via
  // aclrtCreateEventExWithFlag is inserted between the parallel stream and the default stream.
  size_t parallelStreamSize = streams_.size();
  for (size_t i = 0; i < parallelStreamSize; ++i) {
    auto& e = eventsToDefaultStream_[i];
    e->RecordEvent(streams_[i].first);
    e->WaitEventWithoutReset(0);
  }
  RT_VLOG(VL_RUNTIME) << "End parallel dispatch kernels";
}

void KernelLaunchGroupExecutor::SetOutputAndWsCacheMemory(OpRunner* opRunner) {
  // Allocate trace memory for static memory step.
  const std::shared_ptr<std::unordered_map<OpRunner*, std::vector<KernelMemoryTraceBlockPtr>>>& allKernelBlockInfo =
      memoryCache_.GetAllKernelBlocksInfo();
  CHECK_IF_NULL(allKernelBlockInfo);
  const auto& iter = allKernelBlockInfo->find(opRunner);
  if (iter == allKernelBlockInfo->end()) {
    RT_VLOG(VL_RUNTIME) << "Not found kernel block info for kernel: " << opRunner->GetOpName();
  } else {
    const auto& kernelMemBlocks = iter->second;
    const auto& mergeBlocksWithDeviceContext = memoryCache_.GetMergeBlocks();
    CHECK_IF_NULL(mergeBlocksWithDeviceContext);
    const auto& mergeBlocks = mergeBlocksWithDeviceContext->at(deviceContext_);
    for (auto& block : kernelMemBlocks) {
      CHECK_IF_NULL(block);
      void* ptr = mergeBlocks.at(block->memoryTraceBlockIndex_)->start_ + block->offsetInMemoryTraceBlock_;
      CHECK_IF_NULL(ptr);
      if (block->memType_ == kOutputMem) {
        const auto* tensor = block->tensor_;
        CHECK_IF_NULL(tensor);
        std::lock_guard<SpinLock> lock(block->lock_);
        if (tensor->GetStorage()->Data() != ptr) {
          tensor->GetStorage()->SetData(ptr);
        }
      } else {
        CHECK_IF_FAIL(block->index_ != 0);
        opRunner->SetWorkspace(ptr);
      }
    }
  }
}

void KernelLaunchGroupExecutor::UpdateInputTensorData(
    const std::shared_ptr<std::unordered_map<const Tensor*, KernelMemoryTraceBlockPtr>>& tensorToKernelMemBlocks,
    const std::shared_ptr<std::map<const DeviceContext*, std::vector<MemoryTraceBlockPtr>>>&
        mergeBlocksWithDeviceContext,
    const ir::Tensor* tensor) const {
  const auto& iter = tensorToKernelMemBlocks->find(tensor);
  if (iter == tensorToKernelMemBlocks->end()) {
    return;
  }
  auto& kernelMemBlock = iter->second;
  const auto& mergeBlocks = mergeBlocksWithDeviceContext->at(kernelMemBlock->deviceContext_);
  void* ptr =
      mergeBlocks.at(kernelMemBlock->memoryTraceBlockIndex_)->start_ + kernelMemBlock->offsetInMemoryTraceBlock_;

  std::lock_guard<SpinLock> lock(kernelMemBlock->lock_);
  if (tensor->GetStorage()->Data() != ptr) {
    tensor->GetStorage()->SetData(ptr);
  }
}

void KernelLaunchGroupExecutor::SetInputCacheMemory(const OpRunner* opRunner) const {
  const auto& mergeBlocksWithDeviceContext = memoryCache_.GetMergeBlocks();
  CHECK_IF_NULL(mergeBlocksWithDeviceContext);

  const auto& tensorToKernelMemBlocks = memoryCache_.GetTensorToMemBlocksInfo();
  CHECK_IF_NULL(tensorToKernelMemBlocks);
  const auto& inputs = opRunner->GetInput();
  for (const auto& input : inputs) {
    if (input->IsTensor()) {
      const auto& tensor = input->ToTensor();
      UpdateInputTensorData(tensorToKernelMemBlocks, mergeBlocksWithDeviceContext, tensor.get());
    } else if (input->IsTuple()) {
      auto& tuple = input->ToTuple();
      CHECK_IF_NULL(tuple);
      size_t size = tuple->Size();
      for (size_t i = 0; i < size; ++i) {
        const auto& value = (*tuple)[i];
        // Not support nested tuple.
        CHECK_IF_FAIL(!value->IsTuple());
        if (value->IsTensor()) {
          const auto& tensor = value->ToTensor();
          UpdateInputTensorData(tensorToKernelMemBlocks, mergeBlocksWithDeviceContext, tensor.get());
        }
      }
    }
  }
}

void KernelLaunchGroupExecutor::AllocateGraphCacheMemory() const {
  const auto& mergeBlocksWithDeviceContext = memoryCache_.GetMergeBlocks();
  CHECK_IF_NULL(mergeBlocksWithDeviceContext);
  for (auto& item : *mergeBlocksWithDeviceContext) {
    const auto& deviceContext = item.first;
    CHECK_IF_NULL(deviceContext);
    const auto& mergeBlocks = item.second;
    for (auto& block : mergeBlocks) {
      CHECK_IF_NULL(block);
      static const size_t kMemoryAlignSize = 1024;
      void* blockAddr = deviceContext->deviceResManager_->AllocateMemory(block->size_ + kMemoryAlignSize);
      CHECK_IF_NULL(blockAddr);
      block->start_ = reinterpret_cast<uint8_t*>(blockAddr);
    }
  }
}

void KernelLaunchGroupExecutor::FreeGraphCacheMemory() const {
  const auto& mergeBlocksWithDeviceContext = memoryCache_.GetMergeBlocks();
  CHECK_IF_NULL(mergeBlocksWithDeviceContext);
  for (auto& item : *mergeBlocksWithDeviceContext) {
    const auto& deviceContext = item.first;
    CHECK_IF_NULL(deviceContext);
    const auto& mergeBlocks = item.second;
    for (auto& block : mergeBlocks) {
      CHECK_IF_NULL(block);
      deviceContext->deviceResManager_->FreeMemory(block->start_);
    }
  }
}

void KernelLaunchGroupExecutor::AllocateGraphOutputMemory() {
  const auto& graphOutputs = *graphOutputs_;
  for (const auto& outputTensor : graphOutputs) {
    if (outputTensor->GetStorage()->Data() == nullptr) {
      outputTensor->GetStorage()->AllocateMemory();
    }
  }
}
} // namespace runtime
} // namespace fxrt
