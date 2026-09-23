#ifndef __OPS_ASCEND_ACLNN_ACLNN_CUSTOM_CALL_H__
#define __OPS_ASCEND_ACLNN_ACLNN_CUSTOM_CALL_H__

#include "ops/op_base/op_custom_call.h"

namespace fxrt {
namespace ops {
class AclnnCustomCall : public OpCustomCall {
 public:
  AclnnCustomCall() = default;
  ~AclnnCustomCall() = default;
};
} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_ACLNN_CUSTOM_CALL_H__
