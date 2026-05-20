#include "torch_npu/csrc/framework/interface/MsProfilerInterface.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "third_party/acl/inc/acl/acl_prof.h"

namespace at_npu {
namespace native {

#undef TORCH_NPU_LOAD_FUNC
#define TORCH_NPU_LOAD_FUNC(funcName) \
  TORCH_NPU_REGISTER_FUNCTION(libmsprofiler, funcName)

#undef TORCH_NPU_GET_FUNC
#define TORCH_NPU_GET_FUNC(funcName)              \
  TORCH_NPU_GET_FUNCTION(libmsprofiler, funcName)


TORCH_NPU_REGISTER_LIBRARY(libmsprofiler, RTLD_LAZY | RTLD_GLOBAL)
TORCH_NPU_LOAD_FUNC(aclprofWarmup)
TORCH_NPU_LOAD_FUNC(aclprofSetConfig)
TORCH_NPU_LOAD_FUNC(aclprofGetSupportedFeatures)
TORCH_NPU_LOAD_FUNC(aclprofGetSupportedFeaturesV2)
TORCH_NPU_LOAD_FUNC(aclprofRegisterDeviceCallback)
TORCH_NPU_LOAD_FUNC(aclprofMarkEx)

aclError AclProfilingRegisterDeviceCallback()
{
    typedef aclError (*AclProfRegisterDeviceCallbackFunc)();
    static AclProfRegisterDeviceCallbackFunc func = nullptr;
    if (func == nullptr) {
        func = (AclProfRegisterDeviceCallbackFunc)TORCH_NPU_GET_FUNC(aclprofRegisterDeviceCallback);
        if (func == nullptr) {
            return ACL_ERROR_PROF_MODULES_UNSUPPORTED;
        }
    }
    return func();
}

aclError AclProfilingWarmup(const aclprofConfig *profilerConfig)
{
    typedef aclError (*AclProfWarmupFunc)(const aclprofConfig *);
    static AclProfWarmupFunc func = nullptr;
    if (func == nullptr) {
        func = (AclProfWarmupFunc)TORCH_NPU_GET_FUNC(aclprofWarmup);
        if (func == nullptr) {
            return ACL_ERROR_PROF_MODULES_UNSUPPORTED;
        }
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofWarmup", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(profilerConfig);
}

aclError AclprofSetConfig(aclprofConfigType configType, const char* config, size_t configLength) {
    typedef aclError(*AclprofSetConfigFunc)(aclprofConfigType, const char *, size_t);
    static AclprofSetConfigFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofSetConfigFunc)TORCH_NPU_GET_FUNC(aclprofSetConfig);
        if (func == nullptr) {
            return ACL_ERROR_PROF_MODULES_UNSUPPORTED;
        }
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofSetConfig", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(configType, config, configLength);
}

aclError AclprofGetSupportedFeatures(size_t* featuresSize, void** featuresData)
{
    typedef aclError(*AclprofGetSupportedFeaturesFunc)(size_t*, void**);
    static AclprofGetSupportedFeaturesFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofGetSupportedFeaturesFunc)TORCH_NPU_GET_FUNC(aclprofGetSupportedFeaturesV2);
        if (func == nullptr) {
            func = (AclprofGetSupportedFeaturesFunc)TORCH_NPU_GET_FUNC(aclprofGetSupportedFeatures);
        }
    }

    if (func != nullptr) {
        return func(featuresSize, featuresData);
    }
    return ACL_ERROR_PROF_MODULES_UNSUPPORTED;
}

aclError AclProfilingMarkEx(const char *msg, size_t msgLen, aclrtStream stream)
{
    typedef aclError (*aclprofMarkExFunc) (const char *, size_t, aclrtStream);
    static aclprofMarkExFunc func = nullptr;
    if (func == nullptr) {
        func = (aclprofMarkExFunc)TORCH_NPU_GET_FUNC(aclprofMarkEx);
        if (func == nullptr) {
            return ACL_ERROR_PROF_MODULES_UNSUPPORTED;
        }
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofMarkEx", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(msg, msgLen, stream);
}

}
}
