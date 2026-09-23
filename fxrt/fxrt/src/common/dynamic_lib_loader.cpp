#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h>
#endif
#include <limits.h>
#include <string>
#include <map>

#include "common/common.h"
#include "common/dynamic_lib_loader.h"

namespace fxrt {
namespace common {
namespace {
std::string GetErrorMsg() {
#ifndef _WIN32
  const char* result = dlerror();
  return (result == nullptr) ? "Unknown" : result;
#else
  return std::to_string(GetLastError());
#endif
}
} // namespace

DynamicLibLoader::~DynamicLibLoader() {
  for (const auto& [dlName, handle] : allHandles_) {
    if (dlclose(handle) != 0) {
      RT_GLOG(ERROR) << "Closing dynamic lib: " << dlName << "failed, error message: " << GetErrorMsg();
    }
    RT_VLOG(VL_COMMON) << "Close dynamic library: " << dlName << " successfully.";
  }
}

std::string DynamicLibLoader::GetFilePathFromDlInfo() {
  Dl_info dlInfo;
  if (dladdr(reinterpret_cast<void*>(__builtin_return_address(0)), &dlInfo) == 0) {
    RT_GLOG(ERROR) << "Get file path by dladdr failed";
    return "";
  }
  std::string curSoPath = dlInfo.dli_fname;
  RT_VLOG(VL_COMMON) << "Current so path : " << curSoPath;

  auto lastSlashPos = curSoPath.find_last_of("/");
  if (curSoPath.empty() || lastSlashPos == std::string::npos) {
    RT_GLOG(ERROR) << "Current so path empty or the path [" << curSoPath << "] is invalid.";
    return "";
  }
  // During project build, place the current shared library (libfxrt_common.so) and various plugins in
  // the same directory.
  auto dynamicLibPath = curSoPath.substr(0, lastSlashPos);
  RT_VLOG(VL_COMMON) << "Current so dir path : " << dynamicLibPath;
  if (dynamicLibPath.size() >= PATH_MAX) {
    RT_GLOG(ERROR) << "Current path [" << dynamicLibPath << "] is invalid.";
    return "";
  }
  char realPathMem[PATH_MAX] = {0};
  if (realpath(dynamicLibPath.c_str(), realPathMem) == nullptr) {
    RT_GLOG(ERROR) << "Dynamic library path is invalid: [" << dynamicLibPath << "], skip!";
    return "";
  }
  return std::string(realPathMem);
}

bool DynamicLibLoader::LoadDynamicLib(const std::string& dlName, std::stringstream* errMsg) {
  CHECK_IF_NULL(errMsg);
  if (dlName.empty()) {
    RT_GLOG(ERROR) << "Dynamic library name is empty";
    *errMsg << "Dynamic library name is empty" << std::endl;
    return false;
  }
  if (allHandles_.find(dlName) != allHandles_.end()) {
    RT_VLOG(VL_COMMON) << "Dynamic library: " << dlName << " already loaded";
    return true;
  }
  void* handle = dlopen((filePath_ + "/" + dlName).c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    std::string errMsgStr = GetErrorMsg();
    RT_VLOG(VL_COMMON) << "Load dynamic library: " << dlName << " failed. " << errMsgStr;
    *errMsg << "Load dynamic library: " << dlName << " failed. " << errMsgStr << std::endl;
    return false;
  }
  allHandles_[dlName] = handle;
  RT_VLOG(VL_COMMON) << "Load dynamic library: " << dlName << " successfully.";
  return true;
}

void DynamicLibLoader::CloseDynamicLib(const std::string& dlName) {
  if (allHandles_.find(dlName) == allHandles_.end()) {
    RT_VLOG(VL_COMMON) << "Dynamic library: " << dlName << " not found";
    return;
  }
  if (dlclose(allHandles_[dlName]) != 0) {
    RT_GLOG(ERROR) << "Closing dynamic lib: " << dlName << "failed, error message: " << GetErrorMsg();
  }
  allHandles_.erase(dlName);
  RT_VLOG(VL_COMMON) << "Close dynamic library: " << dlName << " successfully.";
}

void* DynamicLibLoader::GetHandle(const std::string& dlName) const {
  auto it = allHandles_.find(dlName);
  if (it != allHandles_.end()) {
    return it->second;
  }
  return nullptr;
}

} // namespace common
} // namespace fxrt
