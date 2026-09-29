#include "torch_npu/csrc/profiler/feature_mgr.h"

#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/framework/interface/MsProfilerInterface.h"


namespace torch_npu {
namespace profiler {

namespace {
const static char* VERSION = "master\0";

static std::unordered_map<std::string, FeatureType> NAME_TABLE = {
    {"ATTR", FeatureType::FEATURE_ATTR},
    {"MemoryAccess", FeatureType::FEATURE_MEMORY_ACCESS},
    {"AICORE_SHAPE", FeatureType::FEATURE_AICORE_SHAPE}};

// featureName, featureVersion
static std::unordered_map<FeatureType, std::string> FMK_FEATURES = {
    {FeatureType::FEATURE_ATTR, "1"},
    {FeatureType::FEATURE_MEMORY_ACCESS, "1"},
    {FeatureType::FEATURE_AICORE_SHAPE, "1"}};
} // namespace


void FeatureMgr::Init()
{
    size_t size = 0;
    void* dataPtr = nullptr;
    auto ret = at_npu::native::AclprofGetSupportedFeatures(&size, &dataPtr);
    if (ret == ACL_ERROR_PROF_MODULES_UNSUPPORTED) {
        ASCEND_LOGW("Not support to get feature list.");
        return;
    } else if (ret != ACL_SUCCESS) {
        ASCEND_LOGE("Failed to get feature list.");
        return;
    }
    FormatFeatureList(size, dataPtr);
}

void FeatureMgr::FormatFeatureList(size_t size, void* featuresData)
{
    FeatureRecord* features = static_cast<FeatureRecord*>(featuresData);
    std::string unsupportedFeatures;
    size_t i = 0;
    while ((features != nullptr) && (i < size)) {
        if (!IsTargetComponent(features->info.affectedComponent, features->info.affectedComponentVersion))  {
            ASCEND_LOGD("feature: %s, component is: %s, componentVersion is: %s",
                        features->featureName, features->info.affectedComponent, features->info.affectedComponentVersion);
            features++;
            i++;
            continue;
        }
        std::string featureName = features->featureName;
        auto it = NAME_TABLE.find(featureName);
        if (it == NAME_TABLE.end()) {
            if (!unsupportedFeatures.empty()) {
                unsupportedFeatures += "; ";
            }
            unsupportedFeatures += featureName + ", log is: " + features->info.infoLog;
            features++;
            i++;
            continue;
        }
        auto tempInfo = FeatureInfo(features->info.compatibility, features->info.featureVersion,
                                    features->info.affectedComponent, features->info.affectedComponentVersion,
                                    features->info.infoLog);
        if (tempInfo.compatibility[0] == '\0' || tempInfo.featureVersion[0] == '\0' ||
            tempInfo.affectedComponent[0] == '\0' || tempInfo.affectedComponentVersion[0] == '\0' ||
            tempInfo.infoLog[0] == '\0') {
            ASCEND_LOGE("Create feature info failed, feature name is: %s.", features->featureName);
            features++;
            i++;
            continue;
        }
        profFeatures_[NAME_TABLE[featureName]] = tempInfo;
        features++;
        i++;
    }
    if (!unsupportedFeatures.empty()) {
        TORCH_NPU_WARN_ONCE("Do not support features: ", unsupportedFeatures);
    }
}

bool FeatureMgr::IsTargetComponent(const char* component, const char* componentVersion)
{
    if (strcmp(component, "all") != 0 && strcmp(component, "PTA") != 0) {
        return false;
    }
    if (strcmp(componentVersion, "all") != 0 && strcmp(componentVersion, VERSION) != 0) {
        return false;
    }
    return true;
}

bool FeatureMgr::IsSupportFeature(FeatureType featureName)
{
    auto fmkIt = FMK_FEATURES.find(featureName);
    auto profIt = profFeatures_.find(featureName);
    if (fmkIt == FMK_FEATURES.end() || profIt == profFeatures_.end()) {
        switch (featureName) {
            case FeatureType::FEATURE_ATTR: {
                TORCH_NPU_WARN_ONCE("FMW or CANN do not support feature: ATTR.");
                break;
            }
            case FeatureType::FEATURE_MEMORY_ACCESS: {
                TORCH_NPU_WARN_ONCE("FMW or CANN do not support feature: MemoryAccess.");
                break;
            }
            case FeatureType::FEATURE_AICORE_SHAPE: {
                TORCH_NPU_WARN_ONCE("FMW or CANN do not support feature: AICORE_SHAPE.");
                break;
            }
            default:
                TORCH_NPU_WARN("FMW or CANN do not support this feature type is: ", static_cast<int>(featureName), ".");
                break;
        }
        return false;
    }

    std::string featureVersion = profFeatures_[featureName].featureVersion;
    if (FMK_FEATURES[featureName] > featureVersion) {
        return false;
    } else if (FMK_FEATURES[featureName] < featureVersion) {
        return (strcmp(profFeatures_[featureName].compatibility, "1") == 0);
    }
    return true;
}

} // profiler
} // torch_npu
