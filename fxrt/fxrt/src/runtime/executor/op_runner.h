#ifndef __RUNTIME_EXECUTOR_OP_RUNNER_H_
#define __RUNTIME_EXECUTOR_OP_RUNNER_H_

#include <vector>
#include <utility>
#include <memory>
#include <functional>

#include "ops/operator.h"
#include "ir/graph.h"
#include "ir/value/value.h"
#include "hardware/hardware_abstract/device_context.h"

namespace fxrt {
namespace runtime {
/**
 * @brief OpRunner class is responsible for executing operator in the runtime.
 * It manages the execution of an operator including shape inference,
 * workspace calculation, and actual launch.
 */
class OpRunner {
 public:
  OpRunner() = delete;
  OpRunner(
      ops::Op opName,
      const std::vector<ir::NodePtr>& inputs,
      const ir::ValuePtr& output,
      std::unique_ptr<ops::Operator>&& operatorPtr,
      void* stream,
      hardware::Device device,
      device::DeviceContext* deviceContext,
      bool isDynamicShape)
      : stream_(stream),
        alloc_(device),
        operator_(std::move(operatorPtr)),
        opName_(opName),
        device_(device),
        deviceContext_(deviceContext),
        isDynamicShape_(isDynamicShape) {
    output_ = output.get();
    auto inputSize = inputs.size();
    input_.resize(inputSize, nullptr);
    for (size_t i = 0; i < inputSize; ++i) {
      auto& inputNodePtr = inputs[i];
      input_[i] = inputNodePtr != nullptr ? inputNodePtr->output.get() : nullptr;
    }
  }

  OpRunner(OpRunner&& other) noexcept
      : storagesToFree_(std::move(other.storagesToFree_)),
        input_(std::move(other.input_)),
        workspace_(other.workspace_),
        workspaceSize_(other.workspaceSize_),
        output_(other.output_),
        stream_(other.stream_),
        alloc_(other.alloc_),
        operator_(std::move(other.operator_)),
        opName_(other.opName_),
        device_(other.device_),
        deviceContext_(other.deviceContext_),
        isDynamicShape_(other.isDynamicShape_) {}

  OpRunner& operator=(OpRunner&& other) noexcept {
    if (this != &other) {
      storagesToFree_ = std::move(other.storagesToFree_);
      input_ = std::move(other.input_);
      workspace_ = other.workspace_;
      workspaceSize_ = other.workspaceSize_;
      output_ = other.output_;
      stream_ = other.stream_;
      alloc_ = other.alloc_;
      operator_ = std::move(other.operator_);
      opName_ = other.opName_;
      device_ = other.device_;
      deviceContext_ = other.deviceContext_;
      isDynamicShape_ = other.isDynamicShape_;
    }
    return *this;
  }

  OpRunner(const OpRunner&) = delete;

  OpRunner& operator=(const OpRunner&) = delete;

  ~OpRunner() = default;

  /**
   * @brief Performs shape inference for the operator if it has dynamic shapes.
   * @return Operator error code indicating success or failure.
   */
  ops::OpsErrorCode InferShape();

  /**
   * @brief Calculates the required workspace size for the operator.
   * @return Operatoir error code indicating success or failure.
   */
  ops::OpsErrorCode CalcWorkspace();

  /**
   * @brief Launches the operator execution.
   * @return Operatoir error code indicating success or failure.
   */
  ops::OpsErrorCode Launch();

  ops::OpsErrorCode Launch(void* stream);

  bool NeedLaunch();

  /**
   * @brief Updates the tensors recorded in tensorsToUpdate_.
   */
  void UpdateTensors();

  /**
   * @brief Allocate device memory for output tensor and workspace.
   */
  void AllocateMemory();

  void AllocateWorkspaceMemory();

  /**
   * @brief Free device memory for input tensors and workspace.
   */
  void FreeMemory();

  void FreeWorkspaceMemory();

  /**
   * @brief Gets the IR node associated with this operator runner.
   * @return Pointer to the IR node.
   */
  const char* GetOpName() const {
    return ops::ToStr(opName_);
  }

  /**
   * @brief Sets the storages that should be freed after execution.
   * @param storagesToFree Vector of storages to free.
   */
  void SetStoragesToFree(std::vector<ir::Storage*>&& storagesToFree) {
    storagesToFree_ = std::move(storagesToFree);
  }

  /**
   * @brief Sets the storages that should be allocated by this operator.
   * @param storagesToAlloc Vector of storages to allocate.
   */
  void SetStoragesToAlloc(std::vector<ir::Storage*>&& storagesToAlloc) {
    storagesToAlloc_ = std::move(storagesToAlloc);
  }

  /**
   * @brief Sets the tensors that should be updated before operator execution.
   * @param tensorsToUpdate Vector of tensors to update.
   */
  void SetTensorsToUpdate(std::vector<ir::Tensor*>&& tensorsToUpdate) {
    tensorsToUpdate_ = std::move(tensorsToUpdate);
  }

  hardware::Device GetDevice() const {
    return device_;
  }

  void* GetWorkspace() const {
    return workspace_;
  }

  void SetWorkspace(void* workspace) {
    workspace_ = workspace;
  }

  size_t GetWorkspaceSize() const {
    return workspaceSize_;
  }

  const std::vector<const ir::Value*>& GetInput() const {
    return input_;
  }

  const ir::Value* GetOutput() const {
    return output_;
  }

  void UpdateRefNodeOutputValue();
  void UpdateRefNodeOutputMetadata();

  ops::OpType GetOpType() const {
    return operator_->GetOpType();
  }

  const ops::Operator* GetOperator() const {
    return operator_.get();
  }

 private:
  using RefTensorPairCallback = std::function<void(uint32_t, uint32_t, const ir::TensorPtr&, const ir::TensorPtr&)>;

  void ForEachRefTensorPair(const RefTensorPairCallback& callback) const;

  // Tensors that should be updated before operator execution.
  std::vector<ir::Tensor*> tensorsToUpdate_;

  // Storages that should be freed after operator execution.
  std::vector<ir::Storage*> storagesToFree_;

  // Storages that should be allocated by this operator.
  std::vector<ir::Storage*> storagesToAlloc_;

  // Input values for the operator.
  std::vector<const ir::Value*> input_;

  // Workspace memory pointer for the operator.
  void* workspace_{nullptr};

  // Size of the total workspace memory in bytes.
  size_t workspaceSize_{0};

  // Output value of the operator.
  ir::Value* output_;

  // Execution stream (e.g., Ascend stream) for the operator.
  void* stream_;

  // The allocator used for device memory management for workspace.
  Allocator alloc_;

  // The operator to implementation.
  std::unique_ptr<ops::Operator> operator_;

  ops::Op opName_;

  // On which type of device does this operator execute on.
  hardware::Device device_;

  // Device context for the operator execution.
  device::DeviceContext* deviceContext_;

  // Flag indicating if the operator has dynamic shapes.
  bool isDynamicShape_;
};
} // namespace runtime
} // namespace fxrt

#endif // __RUNTIME_EXECUTOR_OP_RUNNER_H_
