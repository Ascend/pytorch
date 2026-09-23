#include <vector>

#include "ops/ascend/aclnn/reduce_mean.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AclnnReduceMean::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[0]->ToTensor(),
      input[1]->ToTuple()->ToIntList(),
      input[2]->ToBool(),
      static_cast<fxrt::ir::DataType::Type>(input[3]->ToInt()),
      output->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnReduceMean::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  executor_->Launch(
      workspace,
      workspaceSize,
      stream,
      input[0]->ToTensor(),
      input[1]->ToTuple()->ToIntList(),
      input[2]->ToBool(),
      static_cast<fxrt::ir::DataType::Type>(input[3]->ToInt()),
      output->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(reduce_mean, AclnnReduceMean, Ascend);
} // namespace ops
} // namespace fxrt
