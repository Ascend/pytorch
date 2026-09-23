#include <cstdlib>
#include <stdexcept>

#include "ir/tensor/storage.h"
#include "hardware/hardware_abstract/device_context_manager.h"

namespace fxrt {
Allocator::Allocator(hardware::Device device) {
#ifdef MRT_IR_ABI0_META_ALLOC
  // ABI=0 meta graphs: no device plugin; malloc/free only.
  (void)device;
  deviceResManager_ = nullptr;
#else
  device::DeviceContextKey deviceContextKey = device::DeviceToDeviceContextKey(device);
  auto deviceContext = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(deviceContextKey);
  deviceResManager_ = deviceContext->deviceResManager_.get();
#endif
}

void* Allocator::Allocate(size_t sizeBytes) const {
#ifdef MRT_IR_ABI0_META_ALLOC
  if (deviceResManager_ == nullptr) {
    return std::malloc(sizeBytes);
  }
#endif
  return deviceResManager_->AllocateMemory(sizeBytes);
}

void Allocator::Free(void* ptr) const {
#ifdef MRT_IR_ABI0_META_ALLOC
  if (deviceResManager_ == nullptr) {
    std::free(ptr);
    return;
  }
#endif
  deviceResManager_->FreeMemory(ptr);
}

namespace ir {
Storage::Storage(size_t sizeBytes, hardware::Device device) : sizeBytes_(sizeBytes), alloc_(device), device_(device) {}

Storage::Storage(void* data, size_t sizeBytes, hardware::Device device)
    : data_(data), sizeBytes_(sizeBytes), alloc_(device), device_(device) {}

Storage::~Storage() {
  if (ownsData_ && data_ != nullptr) {
    alloc_.Free(data_);
  }
}

void Storage::Resize(size_t sizeBytes) {
  sizeBytes_ = sizeBytes;
  if (!ownsData_) {
    return;
  }
  if (data_ != nullptr) {
    RT_GLOG(EXCEPTION) << "Device memory leak detected, device type: " << GetDeviceNameByType(device_.type);
  }
}

void Storage::AllocateMemory() {
  if (ownsData_) {
    RT_GLOG(EXCEPTION)
        << "Device memory has already been allocated, or a device memory leak has occurred, device type: "
        << GetDeviceNameByType(device_.type) << ", data: " << data_;
  }
  data_ = alloc_.Allocate(sizeBytes_);
  CHECK_IF_NULL(data_);
  ownsData_ = true;
}

void Storage::FreeMemory() {
  if (!ownsData_) {
    RT_GLOG(EXCEPTION) << "Can not free memory for a storage which doesn't own data, this Storage is used to "
                          "reference memory passed in from external sources.";
  }

  // Free memory from at::Tensor
  if (fromAten_) {
    if (deleter_ == nullptr) {
      RT_GLOG(EXCEPTION) << "Deleter function is null, can not free memory from aten.";
    }
    deleter_(dataToRelease_);
    deleter_ = nullptr;
    fromAten_ = false;
    ownsData_ = false;
    return;
  }

  CHECK_IF_NULL(data_);
  alloc_.Free(data_);
  ownsData_ = false;
}

void* Storage::Release() {
  if (!ownsData_) {
    RT_GLOG(EXCEPTION)
        << "Can not release memory to other from a storage which doesn't own data, this Storage is used to "
           "reference memory passed in from external sources.";
  }
  void* p = data_;
  deleter_ = nullptr;
  fromAten_ = false;
  ownsData_ = false;
  return p;
}

void Storage::SetDataPtrFromAten(void* data, void* data_to_release, const DeleterFn&& deleter, size_t sizeBytes) {
  if (data_ && ownsData_) {
    FreeMemory(); // free old memory
  }
  data_ = data;
  dataToRelease_ = data_to_release;
  deleter_ = std::move(deleter);
  fromAten_ = true;
  ownsData_ = true;
  sizeBytes_ = sizeBytes;
}

} // namespace ir
} // namespace fxrt
