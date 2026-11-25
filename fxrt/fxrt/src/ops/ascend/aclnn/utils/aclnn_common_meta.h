#ifndef __OPS_ASCEND_ACLNN_UTILS_ACLNN_COMMON_META_H__
#define __OPS_ASCEND_ACLNN_UTILS_ACLNN_COMMON_META_H__

#include "common/visible.h"
#include "aclnn_meta_api.h"

namespace fxrt {
namespace ops {

#define DECLARE_ACLNN_COMMON_META_FUNC(name) FXRT_EXPORT _##name##FuncPtr name##_ = nullptr

#define EXTERN_ACLNN_COMMON_META_FUNC(name) \
  extern _##name##FuncPtr name##_;          \
  inline constexpr const char* kName##name##_ = #name

EXTERN_ACLNN_COMMON_META_FUNC(aclCreateTensor);
EXTERN_ACLNN_COMMON_META_FUNC(aclCreateScalar);
EXTERN_ACLNN_COMMON_META_FUNC(aclCreateIntArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclCreateFloatArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclCreateBoolArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclCreateTensorList);

EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyTensor);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyScalar);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyIntArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyFloatArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyBoolArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyTensorList);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyAclOpExecutor);

EXTERN_ACLNN_COMMON_META_FUNC(aclnnInit);
EXTERN_ACLNN_COMMON_META_FUNC(aclnnFinalize);

EXTERN_ACLNN_COMMON_META_FUNC(aclSetAclOpExecutorRepeatable);

EXTERN_ACLNN_COMMON_META_FUNC(aclSetTensorAddr);
EXTERN_ACLNN_COMMON_META_FUNC(aclSetDynamicTensorAddr);

#define GET_ACLNN_COMMON_META_FUNC(name) \
  []() -> auto {                         \
    if (name##_ == nullptr) {            \
      LoadOpApiLib();                    \
    }                                    \
    return name##_;                      \
  }()

#define GET_ACLNN_OP_FUNC(name) GetAclnnOpApiFunc(name.c_str())

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_UTILS_ACLNN_COMMON_META_H__
