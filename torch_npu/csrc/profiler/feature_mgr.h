#pragma once

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>

#include "third_party/acl/inc/acl/acl_prof.h"

#include "torch_npu/csrc/toolkit/profiler/common/singleton.h"
namespace torch_npu {
namespace profiler {

enum class FeatureType {
    FEATURE_MIN = 0,
    FEATURE_ATTR,
    FEATURE_MAX,
};

struct FeatureInfo {
    char compatibility[16] = "\0";
    char featureVersion[16] = "\0";
    char affectedComponent[16] = "\0";
    char affectedComponentVersion[16] = "\0";
    char infoLog[128] = "\0";
    FeatureInfo() = default;
    FeatureInfo(const char* tempCompatibility, const char* tempFeatureVersion, const char* tempAffectedComponent,
                const char* tempAffectedComponentVersion, const char* tempInfoLog)
    {
        // 0 tempData, 1 structData
        std::vector<std::tuple<const char*, char*>> copyList = {
            {tempCompatibility, compatibility},
            {tempFeatureVersion, featureVersion},
            {tempAffectedComponent, affectedComponent},
            {tempAffectedComponentVersion, affectedComponentVersion},
            {tempInfoLog, infoLog},
        };
        std::all_of(copyList.begin(), copyList.end(), [](std::tuple<const char*, char*>& copyNode) {
            std::strcpy(std::get<1>(copyNode), std::get<0>(copyNode));
            return true;
        });
    }
    virtual ~FeatureInfo() {}
};

struct FeatureRecord {
    char featureName[64] = "\0";
    FeatureInfo info;
    FeatureRecord() = default;
    virtual ~FeatureRecord() {}
};

class FeatureMgr : public torch_npu::toolkit::profiler::Singleton<FeatureMgr> {
friend class torch_npu::toolkit::profiler::Singleton<FeatureMgr>;
public:
    FeatureMgr() = default;
    virtual ~FeatureMgr() {}
    void Init();
    bool IsSupportFeature(FeatureType featureName);
private:
    void FormatFeatureList(size_t size, void* featuresData);
    bool IsTargetComponent(const char* component, const char* componentVersion);
private:
    std::unordered_map<FeatureType, FeatureInfo> profFeatures_;
};
} // profiler
} // torch_npu
