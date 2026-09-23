#ifndef __OPS_ASCEND_COMPOSITE_UNIFY_LINEAR_H__
#define __OPS_ASCEND_COMPOSITE_UNIFY_LINEAR_H__

#include "ops/operator.h"
#include "ops/ascend/aclnn/composite/linear.h"
#include "common/dynamic_lib_loader.h"

namespace fxrt {
namespace ops {

class UnifyLinear : public Operator {
 public:
  UnifyLinear();
  ~UnifyLinear() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;

 private:
  std::unique_ptr<Operator> CreateLinearOperator();

  std::unique_ptr<Operator> linear_op_;
  bool use_atb_linear_;
  bool atb_loaded_;
  void* atb_handle_;
  common::DynamicLibLoader lib_loader_;
};

} // namespace ops
} // namespace fxrt

#endif // __OPS_ASCEND_COMPOSITE_UNIFY_LINEAR_H__
