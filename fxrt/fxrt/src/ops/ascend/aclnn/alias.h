#ifndef __OPS_ASCEND_ACLNN_ACLNN_ALIAS_H__
#define __OPS_ASCEND_ACLNN_ACLNN_ALIAS_H__

#include "ops/op_base/op_alias.h"

namespace fxrt {
namespace ops {
class AclnnAlias : public OpAlias {
 public:
  AclnnAlias() {}
  ~AclnnAlias() override = default;
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_ACLNN_ALIAS_H__
