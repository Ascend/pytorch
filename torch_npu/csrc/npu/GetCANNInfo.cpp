#include <iostream>
#include "torch_npu/csrc/npu/GetCANNInfo.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "third_party/acl/inc/acl/acl.h"


std::unordered_map<std::string, aclCANNPackageName> packageNameMap = {
    {"CANN", ACL_PKG_NAME_CANN},
    {"RUNTIME", ACL_PKG_NAME_RUNTIME},
    {"COMPILER", ACL_PKG_NAME_COMPILER},
    {"HCCL", ACL_PKG_NAME_HCCL},
    {"TOOLKIT", ACL_PKG_NAME_TOOLKIT},
    {"OPP", ACL_PKG_NAME_OPP},
    {"OPP_KERNEL", ACL_PKG_NAME_OPP_KERNEL},
    {"DRIVER", ACL_PKG_NAME_DRIVER}
};

double VersionToNum(std::string versionStr)
{
    std::smatch results;
    int major = -1;
    int minor = -1;
    int release = -1;
    int RCVersion = -51;
    int TVersion = -1;
    int alphaVersion = 0;
    if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).RC([0-9]+)"))) {
        major = stoi(results[1]);
        minor = stoi(results[2]);
        RCVersion = stoi(results[3]);
    } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).([0-9]+)"))) {
        major = stoi(results[1]);
        minor = stoi(results[2]);
        release = stoi(results[3]);
    } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).T([0-9]+)"))) {
        major = stoi(results[1]);
        minor = stoi(results[2]);
        TVersion = stoi(results[3]);
    } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).RC([0-9]+).alpha([0-9]+)"))) {
        major = stoi(results[1]);
        minor = stoi(results[2]);
        RCVersion = stoi(results[3]);
        alphaVersion = stoi(results[4]);
    } else {
        TORCH_NPU_WARN_ONCE("Version: " + versionStr + " is invalid.");
        return 0.0;
    }

    double num = ((major + 1) * 100000000) + ((minor + 1) * 1000000) + ((release + 1) * 10000) + ((RCVersion + 1) * 100 + 5000) + ((TVersion + 1) * 100) - (100 - alphaVersion);
    return num;
}

std::unordered_map<std::string, std::string> CANNVersionCache;

std::string GetCANNVersion(const std::string& module)
{
    auto it = CANNVersionCache.find(module);
    if (it != CANNVersionCache.end()) {
        return it->second;
    }
    auto find_module = packageNameMap.find(module);
    if (find_module == packageNameMap.end()) {
        TORCH_NPU_WARN_ONCE("module " + module + "is invalid.");
        CANNVersionCache[module] = "";
        return "";
    }
    aclCANNPackageName name = find_module->second;
    aclCANNPackageVersion version;
    aclError ret = c10_npu::acl::AclsysGetCANNVersion(name, &version);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        TORCH_NPU_WARN_ONCE("Failed to find function aclsysGetCANNVersion");
        CANNVersionCache[module] = "";
        return "";
    }
    std::string module_version = version.version;
    CANNVersionCache[module] = module_version;
    return module_version;
}

bool IsGteCANNVersion(const std::string version, const std::string module)
{
    static std::string baseVersion = "8.1.RC1";
    if (version.compare(baseVersion) < 0) {
        TORCH_CHECK(false, "When the version is less than \"8.1.RC1\", this function is not supported.", PTA_ERROR(ErrCode::VALUE));
    }
    std::string currentVersion = GetCANNVersion(module);
    double current_num = VersionToNum(currentVersion);
    double boundary_num = VersionToNum(version);
    if (current_num >= boundary_num) {
        return true;
    } else {
        return false;
    }
}