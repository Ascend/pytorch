#include "MlInterface.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace c10_npu {

namespace amlapi {
#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
    REGISTER_FUNCTION(libascend_ml, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)           \
    GET_FUNCTION(libascend_ml, funcName)

REGISTER_LIBRARY(libascend_ml)
LOAD_FUNCTION(AmlAicoreDetectOnline)

bool IsExistAmlAicoreDetectOnline()
{
    const static bool isExist = []() -> bool {
        static auto func = GET_FUNC(AmlAicoreDetectOnline);
        return func != nullptr;
    }();
    return isExist;
}

AmlStatus AmlAicoreDetectOnlineFace(int32_t deviceId, const AmlAicoreDetectAttr *attr)
{
    typedef AmlStatus (*amlAicoreDetectOnline)(int32_t, const AmlAicoreDetectAttr *);
    static amlAicoreDetectOnline func = nullptr;
    if (func == nullptr) {
        func = (amlAicoreDetectOnline) GET_FUNC(AmlAicoreDetectOnline);
    }
    TORCH_CHECK(func, "Failed to find function ", "AmlAicoreDetectOnline", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(deviceId, attr);
}

} // namespace amlapi
} // namespace c10_npu
