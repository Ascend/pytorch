#include <iostream>
#include "torch_npu/csrc/core/npu/GetCANNInfo.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "third_party/acl/inc/acl/acl.h"


constexpr size_t kVersionIndex1 = 1;
constexpr size_t kVersionIndex2 = 2;
constexpr size_t kVersionIndex3 = 3;
constexpr size_t kVersionIndex4 = 4;

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

int64_t VersionToNum(std::string versionStr)
{
    std::smatch results;
    int64_t major = -1;
    int64_t minor = -1;
    int64_t release = -1;
    int64_t RCVersion = -51;
    int64_t TVersion = -1;
    int64_t alphaVersion = 0;
    if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).RC([0-9]+)"))) {
        major = stoll(results[kVersionIndex1]);
        minor = stoll(results[kVersionIndex2]);
        RCVersion = stoll(results[kVersionIndex3]);
    } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).([0-9]+)"))) {
        major = stoll(results[kVersionIndex1]);
        minor = stoll(results[kVersionIndex2]);
        release = stoll(results[kVersionIndex3]);
    } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).T([0-9]+)"))) {
        major = stoll(results[kVersionIndex1]);
        minor = stoll(results[kVersionIndex2]);
        TVersion = stoll(results[kVersionIndex3]);
    } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).RC([0-9]+).alpha([0-9]+)"))) {
        major = stoll(results[kVersionIndex1]);
        minor = stoll(results[kVersionIndex2]);
        RCVersion = stoll(results[kVersionIndex3]);
        alphaVersion = stoll(results[kVersionIndex4]);
    } else {
        TORCH_NPU_WARN_ONCE("Version: " + versionStr + " is invalid.");
        return 0;
    }

    int64_t num = ((major + 1) * 100000000) +
                 ((minor + 1) * 1000000) +
                 ((release + 1) * 10000) +
                 ((RCVersion + 1) * 100 + 5000) +
                 ((TVersion + 1) * 100) - (100 - alphaVersion);
    return num;
}

double DriverVersionToNum(std::string versionStr)
{
    std::smatch results;
    int major = -1;
    int minor = -1;
    int release = -1;
    int TVersion = -1;
    int RCVersion = -51;
    int bVersion = 0;
    // driver version check only supports pattern listed here:
    // 24.1.0,24.1.RC1,24.1.rc1,24.1.RC1.B10,24.1.rc1.b10,24.1.T1
    if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).RC([0-9]+)"))) {
        major = stoi(results[kVersionIndex1]);
        minor = stoi(results[kVersionIndex2]);
        RCVersion = stoi(results[kVersionIndex3]);
    } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).rc([0-9]+)"))) {
        major = stoi(results[kVersionIndex1]);
        minor = stoi(results[kVersionIndex2]);
        RCVersion = stoi(results[kVersionIndex3]);
    } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).([0-9]+)"))) {
        major = stoi(results[kVersionIndex1]);
        minor = stoi(results[kVersionIndex2]);
        release = stoi(results[kVersionIndex3]);
    } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).T([0-9]+)"))) {
        major = stoi(results[kVersionIndex1]);
        minor = stoi(results[kVersionIndex2]);
        TVersion = stoi(results[kVersionIndex3]);
    } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).RC([0-9]+).B([0-9]+)"))) {
        major = stoi(results[kVersionIndex1]);
        minor = stoi(results[kVersionIndex2]);
        RCVersion = stoi(results[kVersionIndex3]);
        bVersion = stoi(results[kVersionIndex4]);
    } else if (std::regex_match(versionStr, results, std::regex("([0-9]+).([0-9]+).rc([0-9]+).b([0-9]+)"))) {
        major = stoi(results[kVersionIndex1]);
        minor = stoi(results[kVersionIndex2]);
        RCVersion = stoi(results[kVersionIndex3]);
        bVersion = stoi(results[kVersionIndex4]);
    } else {
        TORCH_NPU_WARN_ONCE("Driver Version: " + versionStr + " is invalid or not supported yet.");
        return 0.0;
    }

    double num = ((static_cast<double>(major) + 1.0) * 100000000) +
                 ((static_cast<double>(minor) + 1.0) * 1000000) +
                 ((static_cast<double>(release) + 1.0) * 10000) +
                 ((static_cast<double>(RCVersion) + 1.0) * 100 + 5000) +
                 ((static_cast<double>(TVersion) + 1.0) * 100) +
                 static_cast<double>(bVersion);
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
    static std::string unsupportedModule = "DRIVER";
    if (module.compare(unsupportedModule) == 0) {
        TORCH_CHECK(false, "When the module is DRIVER, this function is not supported.", PTA_ERROR(ErrCode::VALUE));
    }
    if (version.compare(baseVersion) < 0) {
        TORCH_CHECK(false, "When the version " + version + " is less than \"8.1.RC1\", this function is not supported.", PTA_ERROR(ErrCode::VALUE));
    }
    std::string currentVersion = GetCANNVersion(module);
    int64_t current_num = VersionToNum(currentVersion);
    int64_t boundary_num = VersionToNum(version);
    if (current_num >= boundary_num) {
        return true;
    } else {
        return false;
    }
}

bool IsGteDriverVersion(const std::string driverVersion)
{
    // if cann does not support AclsysGetCANNVersion，GetCANNVersion("DRIVER") will return "".
    // The result of this function will be false, even if current driver version meets the requirement.
    const static std::string baseCANNVersion = "8.1.RC1";
    std::string currentCANNVersion = GetCANNVersion("CANN");
    int64_t currentCannNum = VersionToNum(currentCANNVersion);
    int64_t boundaryCannNum = VersionToNum(baseCANNVersion);
    if (currentCannNum < boundaryCannNum) {
        TORCH_CHECK(false, "When the cann version is less than \"8.1.RC1\", this function is not supported.",
                    PTA_ERROR(ErrCode::VALUE));
    }
    // check driver version
    std::string currentDriverVersion = GetCANNVersion("DRIVER");
    double currentDriverNum = DriverVersionToNum(currentDriverVersion);
    double boundaryDriverNum = DriverVersionToNum(driverVersion);
    return currentDriverNum >= boundaryDriverNum;
}