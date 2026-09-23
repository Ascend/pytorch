#include "hardware/hardware_abstract/device_context_manager.h"
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif
#ifdef __linux__
#include <sys/wait.h>
#endif // #ifdef __linux__
#include <dirent.h>
#include <algorithm>
#include <memory>
#include <string>
#include <set>
#include <fstream>
#include "hardware/hardware_abstract/dlopen_macro.h"
#include "hardware/hardware_abstract/multi_stream_controller.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"

#include "common/logger.h"

namespace fxrt {
namespace {
size_t constexpr GetStrLen(const char* const str) {
  if (*str == '\0') {
    return 0;
  } else {
    return GetStrLen(str + 1) + 1;
  }
}

constexpr auto kCudaHomeEnv = "CUDA_HOME";
constexpr auto kNvccVersionKeyWords = "Cuda compilation tools, release ";
constexpr size_t kNvccVersionKeyWordsSize = GetStrLen(kNvccVersionKeyWords);
constexpr auto kSuccessKeyWord = "Success";
constexpr size_t kSuccessKeyWordSize = GetStrLen(kSuccessKeyWord);
constexpr size_t kBufferSize = 999;
#if defined(_WIN32)
constexpr bool kIsWindowsPlatform = true;
#else
constexpr bool kIsWindowsPlatform = false;
#endif
} // namespace
namespace device {
DeviceContextManager::~DeviceContextManager() {
  Clear();
}

DeviceContextManager& DeviceContextManager::GetInstance() {
  static DeviceContextManager instance{};
  instance.LoadPlugin();
  return instance;
}

void DeviceContextManager::Register(const std::string& deviceName, DeviceContextCreator&& deviceContextCreator) {
  RT_VLOG(VL_HARDWARE) << "Register device context creator for device: " << deviceName;
  if (deviceContextCreators_.find(deviceName) == deviceContextCreators_.end()) {
    (void)deviceContextCreators_.emplace(deviceName, deviceContextCreator);
  }
}

void DeviceContextManager::ClearDeviceContexts() {
  multiStreamControllers_.clear();
  for (auto& iter : deviceContexts_) {
    RT_VLOG(VL_HARDWARE) << "Release device " << iter.first;
    if (iter.second == nullptr) {
      RT_GLOG(ERROR) << "device context is null";
    }
    iter.second->Destroy();
  }
  backendToDeviceContext_.clear();
  deviceContexts_.clear();
}

void DeviceContextManager::ChildAfterFork() {
  RT_VLOG(VL_HARDWARE) << "DeviceContextManager reinitialize after fork.";
  RT_VLOG(VL_HARDWARE) << "Clear deviceContexts_.";
  deviceContexts_.clear();
  RT_VLOG(VL_HARDWARE) << "DeviceContextManager reinitialize after fork done.";
}

void DeviceContextManager::BindDeviceCtx() const {
  for (auto& iter : deviceContexts_) {
    if (iter.second == nullptr) {
      RT_GLOG(ERROR) << "device context is null";
    }
    if (iter.second->deviceResManager_ == nullptr) {
      RT_GLOG(ERROR) << "device res manager is null";
    }
    if (!iter.second->deviceResManager_->BindDeviceToCurrentThread(true)) {
      RT_GLOG(ERROR) << "Bind device failed";
    }
  }
}

DeviceContext* DeviceContextManager::GetOrCreateDeviceContext(const DeviceContextKey& deviceContextKey) {
  std::string deviceContextKeyStr = deviceContextKey.ToString();
  std::string name = deviceContextKey.deviceName_;

  auto deviceContextIter = deviceContexts_.find(deviceContextKeyStr);
  if (deviceContextIter != deviceContexts_.end()) {
    return deviceContextIter->second.get();
  }

  std::shared_ptr<DeviceContext> deviceContext;
  auto creatorIter = deviceContextCreators_.find(name);
  if (creatorIter != deviceContextCreators_.end()) {
    deviceContext = (creatorIter->second)(deviceContextKey);
    if (deviceContext == nullptr) {
      RT_GLOG(ERROR) << "Create device context failed, device context key: " << deviceContextKeyStr;
      return nullptr;
    }
    deviceContext->Initialize();
    if (deviceContext->deviceResManager_ == nullptr) {
      RT_GLOG(ERROR) << "Create device res manager failed, device context key: " << deviceContextKeyStr;
      return nullptr;
    }
    deviceContexts_[deviceContextKeyStr] = deviceContext;
    backendToDeviceContext_[name] = deviceContext;
    multiStreamControllers_[name] = std::make_shared<MultiStreamController>(deviceContext->deviceResManager_.get());
  } else {
    RT_GLOG(EXCEPTION) << "Create device context failed, please make sure target device:" << name
                       << " is available, error message of loading plugins: " << GetErrorMsg();
  }
  return deviceContext.get();
}

DeviceContextPtr DeviceContextManager::GetDeviceContext(const std::string& deviceTarget) {
  if (backendToDeviceContext_.count(deviceTarget) == 0) {
    RT_VLOG(VL_HARDWARE) << "Device context of device " << deviceTarget << " is not created yet.";
    return nullptr;
  }
  return backendToDeviceContext_[deviceTarget];
}

MultiStreamControllerPtr& DeviceContextManager::GetMultiStreamController(const std::string& deviceName) {
  auto&& iter = multiStreamControllers_.find(deviceName);
  if (iter != multiStreamControllers_.end()) {
    return iter->second;
  }
  RT_GLOG(ERROR) << "Found multi stream controller failed, and try to initialize, deviceName : " << deviceName << ".";
  uint32_t deviceId = fxrt::collective::CollectiveManager::Instance().local_rank_id();
  DeviceContextKey hostKey = {deviceName, deviceId};
  const auto& realDeviceContext = GetOrCreateDeviceContext(hostKey);
  if (realDeviceContext == nullptr) {
    RT_GLOG(ERROR) << "get or create device context failed";
  }
  auto&& iterAgain = multiStreamControllers_.find(deviceName);
  if (iterAgain == multiStreamControllers_.end()) {
    RT_GLOG(ERROR) << "Get multi stream controller failed, deviceName : " << deviceName << ".";
  }
  return iterAgain->second;
}

void DeviceContextManager::WaitTaskFinishOnDevice() const {
  for (const auto& item : deviceContexts_) {
    auto deviceContext = item.second;
    try {
      if (deviceContext != nullptr && !deviceContext->deviceResManager_->SyncAllStreams()) {
        RT_GLOG(ERROR) << "SyncStream failed";
        return;
      }
    } catch (const std::exception& ex) {
      RT_GLOG(ERROR) << "SyncStream failed, exception:" << ex.what();
      return;
    }
  }
}

void DeviceContextManager::SyncAllStreams() const {
  for (const auto& item : deviceContexts_) {
    auto deviceContext = item.second;
    if (deviceContext != nullptr && !deviceContext->deviceResManager_->SyncAllStreams()) {
      RT_GLOG(ERROR) << "SyncStream failed, device info: " << deviceContext->GetDeviceContextKey().ToString();
    }
  }
}

std::string DeviceContextManager::GetErrorMsg() const {
  return dlopenErrorMsg_.str();
}

void DeviceContextManager::LoadPlugin() {
  if (loadInit_) {
    return;
  }
  loadInit_ = true;

  DIR* dir = opendir(dynamicLibLoader_.GetDynamicLibFilePath().c_str());
  if (dir == nullptr) {
    RT_GLOG(ERROR) << "Open plugin dir failed, plugin path:" << dynamicLibLoader_.GetDynamicLibFilePath();
    dlopenErrorMsg_ << "Open plugin dir failed, plugin path:" << dynamicLibLoader_.GetDynamicLibFilePath() << std::endl;
    return;
  }
  struct dirent* entry;
  std::set<std::string> pluginFiles;
  while ((entry = readdir(dir)) != nullptr) {
    std::string pluginFile = entry->d_name;
    constexpr auto pluginPrefix = "libhardware_";
    if (pluginFile.find(pluginPrefix) == std::string::npos) {
      continue;
    }
    if (pluginFile.find("libhardware_abstract") != std::string::npos) {
      continue;
    }
    if (pluginFile.find_first_of(".") == std::string::npos) {
      continue;
    }
    pluginFiles.insert(pluginFile);
  }

  for (const auto& targetPluginFile : pluginFiles) {
    if (!dynamicLibLoader_.LoadDynamicLib(targetPluginFile, &dlopenErrorMsg_)) {
      RT_GLOG(ERROR) << "Load " << targetPluginFile << " plugin file failed, error message: " << dlopenErrorMsg_.str();
    }
  }

  (void)closedir(dir);
}

void DeviceContextManager::Clear() {
  backendToDeviceContext_.clear();
  deviceContexts_.clear();
  deviceContextCreators_.clear();
  multiStreamControllers_.clear();
}

} // namespace device
} // namespace fxrt
