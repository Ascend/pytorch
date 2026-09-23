#ifndef FXRT_SRC_HARDWARE_DEVICE_CONTEXT_MANAGER_H_
#define FXRT_SRC_HARDWARE_DEVICE_CONTEXT_MANAGER_H_

#include <set>
#include <any>
#include <map>
#include <string>
#include <memory>
#include <utility>
#include <functional>
#include <mutex>
#include <vector>
#include "hardware/hardware_abstract/device_context.h"
#include "common/visible.h"
#include "common/dynamic_lib_loader.h"

namespace fxrt {
namespace device {
class MultiStreamController;
using DeviceContextCreator = std::function<std::shared_ptr<DeviceContext>(const DeviceContextKey&)>;
using MultiStreamControllerPtr = std::shared_ptr<MultiStreamController>;

class FXRT_EXPORT DeviceContextManager {
 public:
  static DeviceContextManager& GetInstance();
  ~DeviceContextManager();
  void Register(const std::string& deviceName, DeviceContextCreator&& deviceContextCreator);
  DeviceContext* GetOrCreateDeviceContext(const DeviceContextKey& deviceContextKey);
  // Return the device context of the specified device target.
  // The difference between this method and 'GetOrCreateDeviceContext' is this method only query device context by
  // device target(without device id) since fxrt only supports 'single process, single device'.
  DeviceContextPtr GetDeviceContext(const std::string& deviceTarget);
  MultiStreamControllerPtr& GetMultiStreamController(const std::string& deviceName);
  void ClearDeviceContexts();
  void ChildAfterFork();
  void WaitTaskFinishOnDevice() const;
  void SyncAllStreams() const;
  std::string GetErrorMsg() const;
  void BindDeviceCtx() const;

 private:
  DeviceContextManager() = default;
  void LoadPlugin();
  void Clear();

  common::DynamicLibLoader dynamicLibLoader_;
  bool loadInit_;

  // The string converted from DeviceContextKey -> DeviceContextPtr.
  std::map<std::string, DeviceContextPtr> deviceContexts_;
  // The name of device -> vector of DeviceContextPtr.
  std::map<std::string, DeviceContextPtr> backendToDeviceContext_;
  // The name of device -> DeviceContextCreator.
  std::map<std::string, DeviceContextCreator> deviceContextCreators_;
  // record error message of dlopen, print when create deviceContext failed.
  std::stringstream dlopenErrorMsg_;

  // Since multi device is not supported currently, here use device target type to improve performance.
  // Device target type : 0, 1, 2, 3, and real device support : 'Ascend' 'CPU'.
  std::map<std::string, MultiStreamControllerPtr> multiStreamControllers_;
};

class FXRT_EXPORT DeviceContextRegister {
 public:
  DeviceContextRegister(const std::string& deviceName, DeviceContextCreator&& runtimeCreator) {
    DeviceContextManager::GetInstance().Register(deviceName, std::move(runtimeCreator));
  }
  ~DeviceContextRegister() = default;
};

#define FXRT_REGISTER_DEVICE(DEVICE_NAME, DEVICE_CONTEXT_CLASS)          \
  static const DeviceContextRegister g_device_##DEVICE_NAME##_reg(       \
      DEVICE_NAME, [](const DeviceContextKey& deviceContextKey) {        \
        return std::make_shared<DEVICE_CONTEXT_CLASS>(deviceContextKey); \
      })
} // namespace device
} // namespace fxrt
#endif // FXRT_SRC_HARDWARE_DEVICE_CONTEXT_MANAGER_H_
