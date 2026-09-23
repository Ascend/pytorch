#ifndef __OPS_ASCEND_ACLNN_ACLNN_NARROW_VIEW_H__
#define __OPS_ASCEND_ACLNN_ACLNN_NARROW_VIEW_H__

#include "ops/ascend/aclnn/view_base.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace fxrt {
namespace ops {
class AclnnNarrowView : public AclnnViewBase {
 public:
  AclnnNarrowView() = default;
  ~AclnnNarrowView() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_ACLNN_NARROW_VIEW_H__
