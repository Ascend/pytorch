#include "runtime/executor/op_runner.h"
#include "common/logger.h"
#include "ops/op_def/ops_name.h"
#include "ops/utils/utils.h"

namespace fxrt {
namespace runtime {
ops::OpsErrorCode OpRunner::InferShape() {
  if (isDynamicShape_) {
    RT_VLOG(VL_RUNTIME) << "Begin InferShape for op[" << ops::ToStr(opName_) << "], inputs=" << input_;
    auto ret = operator_->InferShape(input_, output_);
    RT_VLOG(VL_RUNTIME) << "End InferShape for op[" << ops::ToStr(opName_) << "]";
    return ret;
  }

  return ops::SUCCESS;
}

ops::OpsErrorCode OpRunner::CalcWorkspace() {
  RT_VLOG(VL_RUNTIME) << "Begin CalcWorkspace for op[" << ops::ToStr(opName_) << "], inputs=" << input_
                      << ", output=" << *output_ << ", workspaceSize=" << workspaceSize_;
  if (!operator_->GetOutputInputRefPairs().empty()) {
    UpdateRefNodeOutputMetadata();
  }
  auto ret = operator_->CalcWorkspace(input_, output_, &workspaceSize_);
  RT_VLOG(VL_RUNTIME) << "End CalcWorkspace for op[" << ops::ToStr(opName_) << "]";
  return ret;
}

ops::OpsErrorCode OpRunner::Launch() {
  void* stream = deviceContext_->deviceResManager_->GetCurrentStream();
  if (device_.type != hardware::DeviceType::CPU) {
    CHECK_IF_NULL(stream);
  }
  RT_VLOG(VL_RUNTIME) << "Begin launch op[" << ops::ToStr(opName_) << "], inputs=" << input_
                      << ", workspace=" << workspace_ << ", workspaceSize=" << workspaceSize_ << ", output=" << *output_
                      << ", stream=" << stream;
  auto ret = operator_->Launch(input_, workspace_, workspaceSize_, output_, stream);
  RT_VLOG(VL_RUNTIME) << "End launch op[" << ops::ToStr(opName_) << "]";
  return ret;
}

ops::OpsErrorCode OpRunner::Launch(void* stream) {
  CHECK_IF_NULL(stream);
  RT_VLOG(VL_RUNTIME) << "Begin launch op[" << ops::ToStr(opName_) << "], inputs=" << input_
                      << ", workspace=" << workspace_ << ", workspaceSize=" << workspaceSize_ << ", output=" << *output_
                      << ", stream=" << stream;
  auto ret = operator_->Launch(input_, workspace_, workspaceSize_, output_, stream);
  RT_VLOG(VL_RUNTIME) << "End launch op[" << ops::ToStr(opName_) << "]";
  return ret;
}

bool OpRunner::NeedLaunch() {
  return operator_->NeedLaunch();
}

void OpRunner::UpdateTensors() {
  for (auto& tensor : tensorsToUpdate_) {
    tensor->Update();
  }
}

void OpRunner::AllocateMemory() {
  // The output tensor will be allocated in torch op, skip allocate memory.
  if (operator_->GetOpType() == ops::OpType::TorchCallOp) {
    return;
  }

  // Allocate memory for output tensor.
  for (auto& storage : storagesToAlloc_) {
    if (storage->CheckOwnsData()) {
      RT_GLOG(EXCEPTION) << "Memory leak for output of operator: " << GetOpName();
    }
    storage->AllocateMemory();
  }
}

void OpRunner::AllocateWorkspaceMemory() {
  if (workspaceSize_ > 0) {
    workspace_ = alloc_.Allocate(workspaceSize_);
    CHECK_IF_NULL(workspace_);
  }
}

void OpRunner::FreeMemory() {
  // Free input tensors that were marked to free.
  for (auto& storage : storagesToFree_) {
    storage->FreeMemory();
  }

  // Free workspace memory.
  FreeWorkspaceMemory();
}

void OpRunner::FreeWorkspaceMemory() {
  if (workspaceSize_ > 0) {
    alloc_.Free(workspace_);
  }
}

void OpRunner::ForEachRefTensorPair(const RefTensorPairCallback& callback) const {
  const std::vector<std::pair<uint32_t, uint32_t>>& refPairs = operator_->GetOutputInputRefPairs();
  if (refPairs.empty()) {
    return;
  }
  for (auto [outputIndex, inputIndex] : refPairs) {
    CHECK_IF_FAIL(inputIndex < input_.size());
    auto& inputValue = input_[inputIndex];
    CHECK_IF_NULL(inputValue);
    CHECK_IF_FAIL(inputValue->IsTensor());
    auto& inputTensor = inputValue->ToTensor();
    CHECK_IF_NULL(inputTensor);

    CHECK_IF_NULL(output_);
    ir::TensorPtr outputTensor = nullptr;
    if (output_->IsTensor()) {
      CHECK_IF_FAIL(outputIndex == 0);
      outputTensor = output_->ToTensor();
    } else if (output_->IsTuple()) {
      auto& outputTuple = output_->ToTuple();
      CHECK_IF_FAIL(outputIndex < outputTuple->Size());
      auto& output = (*outputTuple)[outputIndex];
      CHECK_IF_NULL(output);
      CHECK_IF_FAIL(output->IsTensor());
      outputTensor = output->ToTensor();
    } else {
      RT_GLOG(EXCEPTION) << "Ref output of op[" << GetOpName()
                         << "] must be tensor or tuple, outputIndex=" << outputIndex << ", inputIndex=" << inputIndex
                         << ", output info: " << *output_;
    }
    CHECK_IF_NULL(outputTensor);
    callback(outputIndex, inputIndex, inputTensor, outputTensor);
  }
}

void OpRunner::UpdateRefNodeOutputValue() {
  ForEachRefTensorPair([this](
                           uint32_t outputIndex,
                           uint32_t inputIndex,
                           const ir::TensorPtr& inputTensor,
                           const ir::TensorPtr& outputTensor) {
    RT_VLOG(VL_RUNTIME) << "Update op[" << GetOpName() << "] output value, outputIndex: " << outputIndex
                        << ", inputIndex: " << inputIndex;
    outputTensor->SetStorage(inputTensor->GetStorage());
    outputTensor->SetOwnsStorage(false);
  });
}

void OpRunner::UpdateRefNodeOutputMetadata() {
  ForEachRefTensorPair([this](
                           uint32_t outputIndex,
                           uint32_t inputIndex,
                           const ir::TensorPtr& inputTensor,
                           const ir::TensorPtr& outputTensor) {
    if (inputTensor->Shape() != outputTensor->Shape()) {
      // View-like ref operators update output metadata separately. When shapes differ, reusing input strides may be
      // inaccurate.
      RT_VLOG(VL_RUNTIME) << "Skip generic ref metadata sync for op[" << GetOpName()
                          << "] because input and output shapes differ, outputIndex: " << outputIndex
                          << ", inputIndex: " << inputIndex << ", output shape: " << outputTensor->Shape()
                          << ", input shape: " << inputTensor->Shape();
      return;
    }
    if (operator_->NeedLaunch() && (!ops::IsTensorBaseFormat(inputTensor) || !ops::IsTensorBaseFormat(outputTensor))) {
      RT_GLOG(EXCEPTION) << "Ref-like operator " << GetOpName()
                         << " does not support special-format ref metadata sync. The generic ref metadata path only "
                         << "syncs strides and storageOffset, but special formats may also require format/storageShape "
                         << "updates. outputIndex: " << outputIndex << ", inputIndex: " << inputIndex
                         << ", input format: " << ir::FormatEnumToStr(inputTensor->Format())
                         << ", output format: " << ir::FormatEnumToStr(outputTensor->Format())
                         << ", input storageShape: " << inputTensor->StorageShape()
                         << ", output storageShape: " << outputTensor->StorageShape();
    }
    outputTensor->SetStrides(inputTensor->Strides());
    outputTensor->SetStorageOffset(inputTensor->StorageOffset());
  });
}
} // namespace runtime
} // namespace fxrt
