#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <sstream>
#include "torch_npu/csrc/core/npu/GetCANNInfo.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "third_party/acl/inc/acl/acl.h"


constexpr size_t kVersionIndex1 = 1;
constexpr size_t kVersionIndex2 = 2;
constexpr size_t kVersionIndex3 = 3;
constexpr size_t kVersionIndex4 = 4;

constexpr size_t tokenNum3 = 3;
constexpr size_t tokenNum4 = 4;

constexpr size_t index0 = 0;
constexpr size_t index1 = 1;
constexpr size_t index2 = 2;
constexpr size_t index3 = 3;

constexpr size_t validLength1 = 1;
constexpr size_t validLength2 = 2;
constexpr size_t validLength4 = 4;
constexpr size_t validLength5 = 5;

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

std::vector<std::string> SplitVersionStr(const std::string& str)
{
    std::vector<std::string> tokens;
    std::string token;
    for (char c : str) {
        if (c == '.') {
            tokens.push_back(token);
            token.clear();
        } else {
            token += c;
        }
    }
    tokens.push_back(token);

    return tokens;
}

bool isDigits(const std::string& s)
{
    if (s.empty()) {
        return false;
    }
    for (char c : s) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

int64_t ExtractNumFromStr(const std::string& str)
{
    if (!isDigits(str)) {
        return -1;
    }
    try {
        return std::stoll(str);
    } catch (...) {
        return -1;
    }
}

bool StartsWith(const std::string& str, const std::string& prefix)
{
    return str.size() >= prefix.size() && str.substr(0, prefix.size()) == prefix;
}


int64_t VersionToNum(std::string versionStr)
{
    int64_t major = -1;
    int64_t minor = -1;
    int64_t release = -1;
    int64_t RCVersion = -51;  // Ensure that when there is no matching format, the calculation result of the corresponding item is 0.
    int64_t TVersion = -1;
    int64_t alphaVersion = 0;
    int64_t weight = 0;
    int64_t prerelease = 0;

    std::vector<std::string> tokens = SplitVersionStr(versionStr);

    if (tokens.size() < tokenNum3) {
        TORCH_NPU_WARN_ONCE("Version: \"" + versionStr + "\" is invalid or not supported yet.");
        return 0;
    }

    major = ExtractNumFromStr(tokens[index0]);
    minor = ExtractNumFromStr(tokens[index1]);
    if (major == -1 || minor == -1) {
        TORCH_NPU_WARN_ONCE("Version: \"" + versionStr + "\" is invalid or not supported yet.");
        return 0;
    }

    bool parsed = false;
    if (tokens.size() == tokenNum3) {
        if (StartsWith(tokens[index2], "RC") && tokens[index2].length() > validLength2) {   // ([0-9]+).([0-9]+).RC([0-9]+)
            std::string rcNumStr = tokens[index2].substr(2);
            RCVersion = ExtractNumFromStr(rcNumStr);
            if (RCVersion != -1) {
                parsed = true;
            } else {
                RCVersion = -51;
            }
        }
        if (!parsed && StartsWith(tokens[index2], "T") && tokens[index2].length() > validLength1) {  // ([0-9]+).([0-9]+).T([0-9]+)
            std::string tNumStr = tokens[index2].substr(1);
            TVersion = ExtractNumFromStr(tNumStr);
            if (TVersion != -1) {
                parsed = true;
            }
        }
        if (!parsed && isDigits(tokens[index2])) {  // ([0-9]+).([0-9]+).([0-9]+)
            release = ExtractNumFromStr(tokens[index2]);
            if (release != -1) {
                parsed = true;
            }
        }
    }

    if (!parsed && tokens.size() == tokenNum4) {
        if (StartsWith(tokens[index2], "RC") && tokens[index2].length() > validLength2 && StartsWith(tokens[index3], "alpha") && tokens[index3].length() > validLength5) {  // ([0-9]+).([0-9]+).RC([0-9]+).alpha([0-9]+)
            std::string rcNumStr = tokens[index2].substr(2);
            RCVersion = ExtractNumFromStr(rcNumStr);
            std::string alphaNumStr = tokens[index3].substr(5);
            alphaVersion = ExtractNumFromStr(alphaNumStr);
            if (RCVersion != -1 && alphaVersion != -1) {
                parsed = true;
            } else {
                RCVersion = -51;
                alphaVersion = 0;
            }
        }
        if (!parsed && isDigits(tokens[index2]) && StartsWith(tokens[index3], "alpha") && tokens[index3].length() > validLength5) {  // ([0-9]+).([0-9]+).([0-9]+).alpha([0-9]+)
            release = ExtractNumFromStr(tokens[index2]);
            weight = 300;
            std::string preNumStr = tokens[index3].substr(5);
            prerelease = ExtractNumFromStr(preNumStr);
            if (release != -1 && prerelease != -1) {
                parsed = true;
            } else {
                prerelease = 0;
            }
        }
    }

    if (!parsed) {
        TORCH_NPU_WARN_ONCE("Version: \"" + versionStr + "\" is invalid or not supported yet.");
        return 0;
    }

    int64_t num = ((major + 1) * 100000000) +
                 ((minor + 1) * 1000000) +
                 ((release + 1) * 10000) +
                 ((RCVersion + 1) * 100 + 5000) +
                 ((TVersion + 1) * 100) - (100 - alphaVersion) -
                 weight + prerelease;
    return num;
}

int64_t DriverVersionToNum(std::string versionStr)
{
    std::smatch results;
    int64_t major = -1;
    int64_t minor = -1;
    int64_t release = -1;
    int64_t TVersion = -1;
    int64_t RCVersion = -51;
    int64_t patch = 0;
    int64_t bVersion = 1;
    int64_t alphaVersion = 0;
    // driver version check only supports pattern listed here:
    // pattern is major.minor.release.patch. release:num or RC+num or T+num, patch: num or alpha+num or beta+num.
    std::vector<std::string> tokens = SplitVersionStr(versionStr);

    if (tokens.size() < tokenNum3) {
        TORCH_NPU_WARN_ONCE("Driver Version: \"" + versionStr + "\" is invalid or not supported yet.");
        return 0;
    }

    major = ExtractNumFromStr(tokens[index0]);
    minor = ExtractNumFromStr(tokens[index1]);
    if (major == -1 || minor == -1) {
        TORCH_NPU_WARN_ONCE("Driver Version: \"" + versionStr + "\" is invalid or not supported yet.");
        return 0;
    }

    bool parsed = false;
    if (tokens.size() == tokenNum3) {
        if (StartsWith(tokens[index2], "RC") && tokens[index2].length() > validLength2) {   // ([0-9]+).([0-9]+).RC([0-9]+)
            std::string rcNumStr = tokens[index2].substr(2);
            RCVersion = ExtractNumFromStr(rcNumStr);
            if (RCVersion != -1) {
                parsed = true;
            } else {
                RCVersion = -51;
            }
        }
        if (!parsed && StartsWith(tokens[index2], "T") && tokens[index2].length() > validLength1) {  // ([0-9]+).([0-9]+).T([0-9]+)
            std::string tNumStr = tokens[index2].substr(1);
            TVersion = ExtractNumFromStr(tNumStr);
            if (TVersion != -1) {
                parsed = true;
            }
        }
        if (!parsed && isDigits(tokens[index2])) {  // ([0-9]+).([0-9]+).([0-9]+)
            release = ExtractNumFromStr(tokens[index2]);
            if (release != -1) {
                parsed = true;
            }
        }
    }

    if (!parsed && tokens.size() == tokenNum4) {
        if (isDigits(tokens[index2]) && isDigits(tokens[index3])) {  // ([0-9]+).([0-9]+).([0-9]+).([0-9]+)
            release = ExtractNumFromStr(tokens[index2]);
            patch = ExtractNumFromStr(tokens[index3]);
            if (release != -1 && patch != -1) {
                parsed = true;
            } else {
                patch = 0;
            }
        }
        if (!parsed && StartsWith(tokens[index2], "RC") && tokens[index2].length() > validLength2 && isDigits(tokens[index3])) {  // ([0-9]+).([0-9]+).RC([0-9]+).([0-9]+)
            std::string rcNumStr = tokens[index2].substr(2);
            RCVersion = ExtractNumFromStr(rcNumStr);
            patch = ExtractNumFromStr(tokens[index3]);
            if (RCVersion != -1 && patch != -1) {
                parsed = true;
            } else {
                RCVersion = -51;
                patch = 0;
            }
        }
        if (!parsed && StartsWith(tokens[index2], "RC") && tokens[index2].length() > validLength2 && StartsWith(tokens[index3], "alpha") && tokens[index3].length() > validLength5) {  // ([0-9]+).([0-9]+).RC([0-9]+).alpha([0-9]+)
            std::string rcNumStra = tokens[index2].substr(2);
            RCVersion = ExtractNumFromStr(rcNumStra);
            std::string alphaNumStr = tokens[index3].substr(5);
            alphaVersion = ExtractNumFromStr(alphaNumStr);
            if (RCVersion != -1 && alphaVersion != -1) {
                parsed = true;
            } else {
                RCVersion = -51;
                alphaVersion = 0;
            }
        }
        if (!parsed && StartsWith(tokens[index2], "RC") && tokens[index2].length() > validLength2 && StartsWith(tokens[index3], "beta") && tokens[index3].length() > validLength4) {  // ([0-9]+).([0-9]+).RC([0-9]+).beta([0-9]+)
            std::string rcNumStrb = tokens[index2].substr(2);
            RCVersion = ExtractNumFromStr(rcNumStrb);
            std::string betaNumStr = tokens[index3].substr(4);
            bVersion = ExtractNumFromStr(betaNumStr);
            if (RCVersion != -1 && bVersion != -1) {
                parsed = true;
            } else {
                RCVersion = -51;
                bVersion = 1;
            }
        }
    }

    if (!parsed) {
        TORCH_NPU_WARN_ONCE("Driver Version: \"" + versionStr + "\" is invalid or not supported yet.");
        return 0;
    }

    int64_t num = ((major + 1) * 100000000) +
                  ((minor + 1) * 1000000) +
                  ((release + 1) * 10000) +
                  ((RCVersion + 1) * 100 + 5000) +
                  ((TVersion + 1) * 100) -
                  (alphaVersion != 0 ? 1 : 0) * (100 - alphaVersion) +
                  (bVersion - 1) + patch;
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
        TORCH_NPU_WARN_ONCE("module " + module + " is invalid.");
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
        TORCH_CHECK(false, "When the module is DRIVER, IsGteCANNVersion is not supported.", PTA_ERROR(ErrCode::VALUE));
    }
    if (version.compare(baseVersion) < 0) {
        TORCH_CHECK(false, "When the version " + version + " is less than \"8.1.RC1\", GetCANNVersion is not supported.", PTA_ERROR(ErrCode::VALUE));
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
        TORCH_NPU_WARN_ONCE("When the cann version is less than \"8.1.RC1\", GetCANNVersion is not supported.");
        return false;
    }
    // check driver version
    std::string currentDriverVersion = GetCANNVersion("DRIVER");
    double currentDriverNum = DriverVersionToNum(currentDriverVersion);
    double boundaryDriverNum = DriverVersionToNum(driverVersion);
    return currentDriverNum >= boundaryDriverNum;
}