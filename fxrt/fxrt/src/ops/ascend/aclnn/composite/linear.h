#ifndef __OPS_ASCEND_ACLNN_COMPOSITE_LINEAR_H__
#define __OPS_ASCEND_ACLNN_COMPOSITE_LINEAR_H__

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"
#include "ir/tensor/tensor.h"

namespace fxrt {
namespace ops {
class FXRT_EXPORT AclnnLinear : public Operator {
 public:
  AclnnLinear() {
    // Initialize executors for different operations
    executorMatmul_ = std::make_unique<AclnnExecutor>("aclnnMatmul");
    executorAddmm_ = std::make_unique<AclnnExecutor>("aclnnAddmm");
    executorAdd_ = std::make_unique<AclnnExecutor>("aclnnAdd");
    executorMatmulNz_ = std::make_unique<AclnnExecutor>("aclnnMatmulWeightNz");
    executorAddmmNz_ = std::make_unique<AclnnExecutor>("aclnnAddmmWeightNz");
  }
  ~AclnnLinear() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;

 private:
  // Executors for different ACLNN operations
  std::unique_ptr<AclnnExecutor> executorMatmul_{nullptr}; // Matrix multiplication
  std::unique_ptr<AclnnExecutor> executorAddmm_{nullptr}; // Add + Matmul fused
  std::unique_ptr<AclnnExecutor> executorAdd_{nullptr}; // Element-wise add

  std::unique_ptr<AclnnExecutor> executorMatmulNz_{nullptr};
  std::unique_ptr<AclnnExecutor> executorAddmmNz_{nullptr};

  bool isBiasNone_{false}; // Whether bias is None
  size_t biasRank_{0}; // Rank of bias tensor
  size_t xRank_{0}; // Rank of input tensor
  int8_t cubeMathType_{0}; // Cube math type for matmul
  bool isWeightNz_{false}; // whether weight is in NZ format

  // Temporary tensors for intermediate computations
  ir::TensorPtr weightTransposeTensor_{nullptr}; // Transposed weight tensor
  ir::TensorPtr inputKernelTensor_{nullptr}; // Reshaped input tensor (2D)
  ir::TensorPtr outputKernelTensor_{nullptr}; // Reshaped output tensor (2D)
  ir::TensorPtr matmulTensor_{nullptr}; // Intermediate matmul result

  ir::ValuePtr alphaScalar_{nullptr};
  ir::ValuePtr betaScalar_{nullptr};

  /**
   * @brief Transpose the weight tensor by swapping the last two dimensions
   *
   * This creates a view of the weight tensor with transposed dimensions.
   * For a weight tensor of shape [..., M, N], it becomes [..., N, M].
   *
   * The w_t_tensor is already a clone (shallow copy) that shares storage with w_tensor.
   * We modify its shape and strides to create a transpose view.
   */
  std::vector<int64_t> TransposeWeight(
      const ir::TensorPtr& wTensor,
      const std::vector<int64_t>& wShape,
      ir::TensorPtr& weightTransposeTensor);

  /**
   * @brief Calculate storage shape for NZ format tensor
   *
   * For NZ format, storage shape is calculated based on view shape:
   * - Keep first N-2 dimensions
   * - Convert last 2 dimensions to block format: [ceil(W/16), ceil(H/16), 16, 16]
   *
   * @param viewShape The view shape (logical shape)
   * @param format The memory format
   * @param dtype The data type
   * @return The calculated storage shape
   */
  std::vector<int64_t> CalculateStorageShapeForNZ(
      const std::vector<int64_t>& viewShape,
      ir::MemoryFormat format,
      ir::DataType dtype);

  /**
   * @brief Set flattened 2D tensor storage info for ND linear operations
   *
   * This function reshapes an ND tensor to 2D for matrix multiplication.
   * It creates a view that flattens all dimensions except the last one.
   *
   * Example: [2, 3, 4, 5] -> [2*3*4, 5] = [24, 5]
   *
   * The tensor should be a clone (shallow copy) that shares storage with the original.
   * We modify its shape, strides, and storage shape to create a 2D view.
   */
  void SetFlatternNdLinearTensorStorageInfo(
      ir::TensorPtr& tensor,
      int64_t newShapeFirst,
      const std::vector<int64_t>& shape);

  static ir::ValuePtr CreateScalarValueOne(ir::DataType dtype);
};

} // namespace ops
} // namespace fxrt

#endif // __OPS_ASCEND_ACLNN_COMPOSITE_LINEAR_H__
