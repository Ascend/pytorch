#include "ops/ascend/aclnn/utils/aclnn_common_meta.h"

namespace fxrt {
namespace ops {
DECLARE_ACLNN_COMMON_META_FUNC(aclCreateTensor);
DECLARE_ACLNN_COMMON_META_FUNC(aclCreateScalar);
DECLARE_ACLNN_COMMON_META_FUNC(aclCreateIntArray);
DECLARE_ACLNN_COMMON_META_FUNC(aclCreateFloatArray);
DECLARE_ACLNN_COMMON_META_FUNC(aclCreateBoolArray);
DECLARE_ACLNN_COMMON_META_FUNC(aclCreateTensorList);

DECLARE_ACLNN_COMMON_META_FUNC(aclDestroyTensor);
DECLARE_ACLNN_COMMON_META_FUNC(aclDestroyScalar);
DECLARE_ACLNN_COMMON_META_FUNC(aclDestroyIntArray);
DECLARE_ACLNN_COMMON_META_FUNC(aclDestroyFloatArray);
DECLARE_ACLNN_COMMON_META_FUNC(aclDestroyBoolArray);
DECLARE_ACLNN_COMMON_META_FUNC(aclDestroyTensorList);
DECLARE_ACLNN_COMMON_META_FUNC(aclDestroyAclOpExecutor);

DECLARE_ACLNN_COMMON_META_FUNC(aclnnInit);
DECLARE_ACLNN_COMMON_META_FUNC(aclnnFinalize);

DECLARE_ACLNN_COMMON_META_FUNC(aclSetAclOpExecutorRepeatable);

DECLARE_ACLNN_COMMON_META_FUNC(aclSetTensorAddr);
DECLARE_ACLNN_COMMON_META_FUNC(aclSetDynamicTensorAddr);

} // namespace ops
} // namespace fxrt
