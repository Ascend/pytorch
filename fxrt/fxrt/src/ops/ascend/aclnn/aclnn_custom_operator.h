#ifndef __OPS_ASCEND_ACLNN_ACLNN_CUSTOM_OPERATOR_H__
#define __OPS_ASCEND_ACLNN_ACLNN_CUSTOM_OPERATOR_H__

#include <vector>
#include <optional>
#include <tuple>
#include <string>
#include <utility>
#include <memory>
#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace fxrt {
namespace ops {

/**
 * @brief Custom Aclnn operator base class, integrated with AclnnExecutor
 *
 * Users need to inherit this class and implement CalcWorkspace and Launch methods,
 * directly using AclnnExecutor. This ensures complete compatibility of custom operators
 * with the existing Aclnn system.
 *
 * Note: InferShape function is not concerned, shape inference is handled automatically by the system.
 */
class DA_API AclnnCustomOperator : public Operator {
 public:
  /**
   * @brief Constructor
   * @param aclnn_api_name Aclnn API name (e.g., "aclnnMul")
   */
  explicit AclnnCustomOperator(const std::string& aclnn_api_name) : aclnn_api_name_(aclnn_api_name) {
    // Initialize AclnnExecutor, need to pass API name
    executor_ = std::make_unique<AclnnExecutor>(aclnn_api_name);
  }

  virtual ~AclnnCustomOperator() = default;

  /**
   * @brief Calculate total workspace memory size requirements for the kernel computation.
   * @param input Vector of pointers to input data.
   * @param output Pointer to the output data.
   * @param workspace_size Pointer to the workspace memory size, the workspace memory size in bytes needs to be updated
   * to the variable pointed by `workspace_size`.
   * @return OpsErrorCode Error code indicating success or failure of workspace calculation.
   */
  virtual OpsErrorCode CalcWorkspace(
      const std::vector<const ir::Value*>& input,
      const ir::Value* output,
      size_t* workspaceSize) = 0;

  /**
   * @brief Launch the custom operator kernel on Ascend.
   *
   * @param input Vector of input ir::Value pointers.
   * @param workspace Pointer to the workspace memory.
   * @param workspaceSize Size of the workspace memory in bytes.
   * @param output Pointer to the output ir::Value.
   * @param stream Pointer to the stream for asynchronous execution.
   */
  virtual OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) = 0;

 protected:
  /**
   * @brief Get AclnnExecutor instance
   * Subclasses can use this method to access executor_
   */
  AclnnExecutor* GetExecutor() const {
    return executor_.get();
  }

  /**
   * @brief Get Aclnn API name
   */
  const std::string& GetAclnnApiName() const {
    return aclnn_api_name_;
  }

 private:
  std::string aclnn_api_name_; // Aclnn API name
  std::unique_ptr<AclnnExecutor> executor_; // AclnnExecutor instance
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_ACLNN_CUSTOM_OPERATOR_H__
