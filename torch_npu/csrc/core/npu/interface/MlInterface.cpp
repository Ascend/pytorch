#include "MlInterface.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace c10_npu {

namespace amlapi {
#undef TORCH_NPU_LOAD_FUNC
#define TORCH_NPU_LOAD_FUNC(funcName) \
    TORCH_NPU_REGISTER_FUNCTION(libascend_ml, funcName)

#undef TORCH_NPU_GET_FUNC
#define TORCH_NPU_GET_FUNC(funcName)           \
    TORCH_NPU_GET_FUNCTION(libascend_ml, funcName)

TORCH_NPU_REGISTER_LIBRARY(libascend_ml)
TORCH_NPU_LOAD_FUNC(AmlAicoreDetectOnline)
TORCH_NPU_LOAD_FUNC(AmlP2PDetectOnline)

bool IsExistAmlAicoreDetectOnline()
{
    const static bool isExist = []() -> bool {
        static auto func = TORCH_NPU_GET_FUNC(AmlAicoreDetectOnline);
        return func != nullptr;
    }();
    return isExist;
}

bool IsExistAmlP2PDetectOnline()
{
    const static bool isExist = []() -> bool {
        static auto func = TORCH_NPU_GET_FUNC(AmlP2PDetectOnline);
        return func != nullptr;
    }();
    return isExist;
}

AmlStatus AmlAicoreDetectOnlineFace(int32_t deviceId, const AmlAicoreDetectAttr *attr)
{
    typedef AmlStatus (*amlAicoreDetectOnline)(int32_t, const AmlAicoreDetectAttr *);
    static amlAicoreDetectOnline func = nullptr;
    if (func == nullptr) {
        func = (amlAicoreDetectOnline) TORCH_NPU_GET_FUNC(AmlAicoreDetectOnline);
    }
    TORCH_CHECK(func, "Failed to find function ", "AmlAicoreDetectOnline", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(deviceId, attr);
}

AmlStatus AmlP2PDetectOnlineFace(int32_t deviceId, void *comm, const AmlP2PDetectAttr *attr)
{
    typedef AmlStatus (*amlP2PDetectOnline)(int32_t, void *, const AmlP2PDetectAttr *);
    static amlP2PDetectOnline func = nullptr;
    if (func == nullptr) {
        func = (amlP2PDetectOnline) TORCH_NPU_GET_FUNC(AmlP2PDetectOnline);
    }
    TORCH_CHECK(func, "Failed to find function ", "AmlP2PDetectOnline", PTA_ERROR(ErrCode::NOT_FOUND));
    return func(deviceId, comm, attr);
}

} // namespace amlapi
} // namespace c10_npu
