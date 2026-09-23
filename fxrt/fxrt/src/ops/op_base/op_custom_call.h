#ifndef __OPS_OP_BASE_OP_CUSTOM_CALL_H__
#define __OPS_OP_BASE_OP_CUSTOM_CALL_H__

#include <string>
#include <vector>

#include "ops/op_register.h"

namespace fxrt {
namespace ops {
class OpCustomCall : public Operator {
 public:
  OpCustomCall() = default;
  ~OpCustomCall() override = default;

  void Init(const std::vector<const ir::Value*>& inputs, const ir::Value* output);

  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;

  OpsErrorCode CalcWorkspace(
      const std::vector<const ir::Value*>& input,
      const ir::Value* output,
      size_t* workspaceSize);

  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream);

  bool NeedLaunch() override {
    return operatorPtr_->NeedLaunch();
  }

 protected:
  std::string opName_;
  std::shared_ptr<ops::Operator> operatorPtr_;
  std::vector<const ir::Value*> input_;
};
} // namespace ops
} // namespace fxrt

#endif // __OPS_OP_BASE_OP_CUSTOM_CALL_H__
