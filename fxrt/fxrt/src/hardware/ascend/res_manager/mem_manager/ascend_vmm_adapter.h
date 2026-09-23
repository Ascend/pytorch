#ifndef FXRT_SRC_HARDWARE_ASCEND_ASCEND_VMM_ADAPTER_H_
#define FXRT_SRC_HARDWARE_ASCEND_ASCEND_VMM_ADAPTER_H_

#include <atomic>
#include <memory>
#include <map>
#include <vector>
#include <set>
#include <string>
#include <fstream>
#include <iostream>

#include "acl/acl.h"
#include "hardware/hardware_abstract/dlopen_macro.h"
#include "common/common.h"

#include "common/visible.h"

namespace fxrt {
namespace device {
namespace ascend {
using DeviceMemPtr = void(*);
class FXRT_EXPORT AscendVmmAdapter {
 public:
  static AscendVmmAdapter& GetInstance() {
    static AscendVmmAdapter instance{};
    return instance;
  }

  AscendVmmAdapter() {
    vmmAlignSize_ = kDefaultAlignSize;

    RT_VLOG(VL_HARDWARE) << "VMM align size is " << vmmAlignSize_;
  }
  ~AscendVmmAdapter() = default;

 public:
  size_t GetRoundUpAlignSize(size_t inputSize) const;
  size_t GetRoundDownAlignSize(size_t inputSize) const;

  void ClearAllMemory();
  size_t AllocDeviceMem(size_t size, DeviceMemPtr* addr);
  size_t MmapDeviceMem(const size_t size, const DeviceMemPtr addr, const size_t maxSize);
  size_t EagerFreeDeviceMem(const DeviceMemPtr addr, const size_t size);
  size_t GetAllocatedSize() {
    return physicalHandleSize_ * vmmAlignSize_;
  }

  size_t EmptyCache();

  static const bool IsEnabled() {
    static bool isEnableVmm = IsVmmEnabled();
    return isEnableVmm;
  }

 private:
  static const bool IsVmmEnabled() {
    if (!CheckVmmDriverVersion()) {
      return false;
    }

    RT_VLOG(VL_HARDWARE) << "VMM is enabled.";
    return true;
  }

 private:
  uint64_t vmmAlignSize_;
  DeviceMemPtr FindVmmSegment(const DeviceMemPtr addr);
  size_t GetHandleSize(size_t inputSize);
  std::atomic<size_t> physicalHandleSize_{0};
  std::map<DeviceMemPtr, aclrtDrvMemHandle> vmmMap_;
  std::vector<DeviceMemPtr> allReserveMems_;
  std::set<aclrtDrvMemHandle> cachedHandleSets_;
  static constexpr uint64_t kMB = 1024 * 1024;
  static constexpr uint64_t kDefaultAlignSize = 2 * kMB;
  static int StringToMB(const std::string& str) {
    std::stringstream ss(str);
    int num;
    std::string unit;
    if (!(ss >> num)) {
      RT_GLOG(ERROR) << "No valid number could be extracted from the string, " << str;
    }
    if (!(ss >> unit) || unit != "MB") {
      RT_GLOG(ERROR) << "The unit of the string is not MB, " << str;
    }
    if (ss.rdbuf()->in_avail() > 0) {
      RT_GLOG(ERROR) << "The string has extra characters, " << str;
    }
    return num;
  }
  static bool CheckVmmDriverVersion() {
    // Get driver version
    constexpr auto ascendInstallInfo = "/etc/ascend_install.info";
    const std::string DRIVER_INSTALL_PATH_PARAM = "Driver_Install_Path_Param=";
    std::string driverPath = "/usr/local/Ascend";

    std::ifstream ascendInstallFile(ascendInstallInfo);
    if (!ascendInstallFile.is_open()) {
      RT_VLOG(VL_HARDWARE) << "Open file " << ascendInstallInfo << " failed.";
    } else {
      std::string line;
      while (std::getline(ascendInstallFile, line)) {
        size_t pos = line.find(DRIVER_INSTALL_PATH_PARAM);
        if (pos != std::string::npos) {
          // Extract the path after "Driver_Install_Path_Param="
          driverPath = line.substr(pos + DRIVER_INSTALL_PATH_PARAM.length());
          RT_VLOG(VL_HARDWARE) << "Driver path is " << driverPath;
          break;
        }
      }
    }

    auto splitString = [](const std::string& str, char delimiter) -> std::vector<std::string> {
      std::vector<std::string> tokens;
      std::string token;
      std::istringstream tokenStream(str);
      while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
      }
      return tokens;
    };

    auto driverVersionInfo = driverPath + "/driver/version.info";
    const std::string DRIVER_VERSION_PARAM = "Version=";
    std::ifstream driverVersionFile(driverVersionInfo);
    if (!driverVersionFile.is_open()) {
      RT_VLOG(VL_HARDWARE) << "Open file " << driverVersionInfo << " failed.";
    } else {
      std::string line;
      while (std::getline(driverVersionFile, line)) {
        size_t pos = line.find(DRIVER_VERSION_PARAM);
        if (pos != std::string::npos) {
          // Extract the version after "Version="
          std::string driverVersion = line.substr(pos + DRIVER_VERSION_PARAM.length());
          auto splitVersion = splitString(driverVersion, '.');
          RT_VLOG(VL_HARDWARE) << "Driver version is " << driverVersion << ", major version is " << splitVersion[0];
          if (splitVersion[0] < "24") {
            RT_VLOG(VL_HARDWARE) << "Driver version is less than 24.0.0, vmm is disabled by default, drvier_version: "
                                 << driverVersion;
            return false;
          }
          break;
        }
      }
    }
    return true;
  }
};
} // namespace ascend
} // namespace device
} // namespace fxrt

#endif
