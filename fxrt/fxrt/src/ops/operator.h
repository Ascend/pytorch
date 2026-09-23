#ifndef __OPS_OPERATOR_H__
#define __OPS_OPERATOR_H__

#include <functional>
#include <string>
#include <vector>
#include <unordered_map>

#include "ops/utils/op_constants.h"
#include "common/visible.h"
#include "ir/graph.h"

namespace fxrt {
namespace ops {

// Deprecated Interface for kernel
class DAKernel {
 public:
  explicit DAKernel(ir::NodePtr node) : node_(node) {}
  virtual ~DAKernel() = default;

  virtual void Init() = 0;
  virtual void InferShape() = 0;
  virtual void Resize() = 0;
  virtual void Launch() = 0;

 protected:
  ir::NodePtr node_;
};

// Operator-related error codes. The error types within them will be further expanded in the future.
enum OpsErrorCode {
  SUCCESS = 0,
  INVALID_PARAM,
  INVALID_SHAPE,
  INVALID_INPUT_NUM,
  INVALID_DEVICE_ADDR,
  LAUNCH_OP_FAILED,
  UNKNOWN_ERROR = 1000
};

/**
 * @brief Enumeration of the possible types a Operator can hold.
 */
enum OpType { FxrtOp, CustomCallOp, TorchCallOp, PythonCallOp };

// @brief Abstract base class representing a computational kernel. A Operator encapsulates the core computation logic
// for a specific operator. Derived classes must implement shape inference and launch operations. Kernels of different
// device types share the InferShape function, but need to implement their respective Launch functions.
class DA_API Operator {
 public:
  Operator() = default;
  virtual ~Operator() = default;

  // Get the operator type.
  OpType GetOpType() const {
    return opType_;
  }
  // Set the operator type.
  void SetOpType(OpType opType) {
    opType_ = opType;
  }

  /**
   * @brief Initialize the operator with input and output.
   * @param input Vector of pointers to input data.
   * @param output Pointer to the output data.
   */
  virtual void Init(const std::vector<const ir::Value*>& input, const ir::Value* output) {}

  /**
   *  @brief Infer the output shape based on input shape or value.
   *  @param input Vector of pointers to input data.
   *  @param output Pointer to the output data, the inferred output shape needs to be updated to the output. Note: The
   *  output may be one of types such as Tensor, Tuple, etc.
   *  @return OpsErrorCode Error code indicating success or failure of shape inference.
   */
  virtual OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output);

  /**
   * @brief Calculate total workspace memory size requirements for the kernel computation.
   * @param input Vector of pointers to input data.
   * @param output Pointer to the output data.
   * @param workspaceSize Pointer to the workspace memory size, the workspace memory size in bytes needs to be updated
   * to the variable pointed by `workspaceSize`.
   * @return OpsErrorCode Error code indicating success or failure of workspace calculation.
   */
  virtual OpsErrorCode CalcWorkspace(
      const std::vector<const ir::Value*>& input,
      const ir::Value* output,
      size_t* workspaceSize) {
    return SUCCESS;
  }

  /**
   * @brief Launch the computational kernel operation to the target device. It handles device-specific async or sync
   * execution.
   *
   * Note: If the operator needs to update output shape after launch, the shape update logic must be implemented within
   * the Launch function. Please refer to the comment for function `NeedUpdateOutputShapeAfterLaunch`
   *
   * @param input Vector of pointers to input data. Contains all input data required for computation.
   * @param workspace The pointer to workspace data. Provides temporary memory for
   *                  intermediate calculations and storage during the operation.
   * @param workspaceSize The workspace memory buffer size in bytes.
   * @param output Pointer to the output data. Stores the result of the computation after
   *               successful execution.
   * @param stream Pointer to the device-specific execution stream (e.g., AclStream for Ascend NPU). Used for
   * asynchronous or synchronous operation. May be nullptr for synchronous CPU operations.
   * @return OpsErrorCode Return SUCCESS if execution completed successfully, or an appropriate
   *         error code if the operation failed.
   */
  virtual OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) = 0;

  /**
   * @brief This method indicates if the operator requires output shape updates after the Launch
   * function has completed execution, such as `Unique` op.
   * Note: When returning true, the shape update logic must be implemented within the Launch function itself.
   * When necessary, the Launch function needs to synchronize the stream first to ensure the kernel task execution
   * completes on the device, and then update the shape.
   *
   * @return bool Returns true if the output shape requires post-execution updates;
   *         otherwise returns false.
   */
  virtual bool NeedUpdateOutputShapeAfterLaunch() const {
    return false;
  }

  /**
   * @brief Get pairs of output and input indices that reference the same tensor.
   * This method returns a vector of pairs where each pair contains an output index
   * and an input index that share the same tensor data. This is used for operations
   * that can reuse input tensors as outputs (in-place operations).
   *
   * @return Vector of pairs, where each pair consists of (output index, input index)
   *         that reference the same tensor. Returns empty vector by default.
   */
  virtual std::vector<std::pair<uint32_t, uint32_t>> GetOutputInputRefPairs() const {
    return {};
  }

  /**
   * @brief This method indicates if the operator will be put into launch queue. If return true, this operator will be
   * put into launch queue, otherwise not.
   *
   * @return Whether the operator will be put into launch queue.
   */
  virtual bool NeedLaunch() {
    return true;
  }

 private:
  OpType opType_ = OpType::FxrtOp;
};
} // namespace ops
} // namespace fxrt
#endif // __OPS_OPERATOR_H__
