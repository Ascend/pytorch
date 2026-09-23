#ifndef __OPS_ASCEND_ACLNN_ACLNN_IOTA_H__
#define __OPS_ASCEND_ACLNN_ACLNN_IOTA_H__

#include <cstdint>
#include <memory>
#include <vector>

#include "ir/value/value.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"
#include "ops/operator.h"

namespace fxrt {
namespace ops {
class AclnnIota : public Operator {
 public:
  AclnnIota() {
    executor_ = std::make_unique<AclnnExecutor>("aclnnArange");
  }
  ~AclnnIota() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;

 private:
  ir::Value end_{static_cast<int64_t>(0)};
  std::unique_ptr<AclnnExecutor> executor_{nullptr};
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_ACLNN_IOTA_H__
