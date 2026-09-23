#ifndef __IR_TENSOR_STORAGE_H__
#define __IR_TENSOR_STORAGE_H__

#include <cstddef>
#include <memory>
#include <cstring>

#include "common/common.h"
#include "common/visible.h"
#include "hardware/device.h"
#include "ir/common/dtype.h"
#include "ir/common/intrusive_ptr.h"

namespace fxrt {
namespace device {
class DeviceResManager;
}

using DeleterFn = std::function<void(void*)>;
class FXRT_EXPORT Allocator {
 public:
  Allocator() = delete;
  explicit Allocator(hardware::Device device);

  void* Allocate(size_t sizeBytes) const;
  void Free(void* ptr) const;

 private:
  device::DeviceResManager* deviceResManager_{nullptr};
};

namespace ir {

/**
 * @brief Implementation of the storage for a tensor.
 *
 * This class manages a block of memory on a specific device.
 * It is reference-counted and managed by the Storage class.
 */
class FXRT_EXPORT Storage : public RefCounted {
 public:
  /**
   * @brief Constructs a Storage, allocating memory.
   * @param sizeBytes The size of the storage in bytes.
   * @param device The device where the storage is located.
   */
  Storage(size_t sizeBytes, hardware::Device device);
  /**
   * @brief Constructs a Storage from an existing buffer.
   * The storage does not own the data and will not free it.
   * @param data Pointer to the existing data.
   * @param sizeBytes The size of the storage in bytes.
   * @param device The device where the storage is located.
   */
  Storage(void* data, size_t sizeBytes, hardware::Device device);
  /**
   * @brief Destructor. Frees the allocated memory if it owns it.
   */
  ~Storage();

  /**
   * @brief Gets a const pointer to the data.
   * @return A const void pointer to the data.
   */
  const void* Data() const {
    return data_;
  }
  /**
   * @brief Gets a pointer to the data.
   * @return A void pointer to the data.
   */
  void* Data() {
    return data_;
  }
  /**
   * @brief Gets the size of the storage in bytes.
   * @return The size in bytes.
   */
  size_t SizeBytes() const {
    return sizeBytes_;
  }
  /**
   * @brief Gets the device of the storage.
   * @return The device.
   */
  hardware::Device GetDevice() const {
    return device_;
  }

  void SetData(void* data) {
    CHECK_IF_FAIL(!ownsData_);
    data_ = data;
  }

  void SetDataPtrFromAten(void* data, void* data_to_release, const DeleterFn&& deleter, size_t sizeBytes);

  DeleterFn GetDeleter() const {
    return deleter_;
  }

  void Resize(size_t sizeBytes);

  /**
   * @brief Retrieves the allocator instance associated with this object.
   * @return The allocator used for memory management.
   */
  Allocator GetAllocator() const {
    return alloc_;
  }

  /**
   * @brief Allocates memory using the configured allocator according to device type.
   * This function checks for duplicate memory allocation or memory leaks.
   */
  void AllocateMemory();

  /**
   * @brief Frees the currently allocated memory, if owned.
   */
  void FreeMemory();

  /**
   * @brief Check whether this Storage currently owns the data.
   * If true, the buffer pointed to by data_ is managed by this Storage object.
   */
  bool CheckOwnsData() const {
    return ownsData_;
  }

  /**
   * @brief Releases ownership of the managed pointer.
   * @return The raw data pointer.
   */
  void* Release();

 private:
  void* data_{nullptr}; ///< Pointer to the allocated memory.
  void* dataToRelease_{nullptr}; ///< Pointer to the memory to release.
  size_t sizeBytes_{0}; ///< Size of the memory in bytes.
  Allocator alloc_;
  hardware::Device device_; ///< The device where the memory is allocated.
  bool ownsData_{false}; ///< Whether the storage can own data.

  bool fromAten_{false}; ///< Whether the storage is from aten.
  DeleterFn deleter_ = nullptr; ///< Deleter function pointer for external memory management.
};

using StoragePtr = IntrusivePtr<Storage>;

} // namespace ir
} // namespace fxrt

#endif // __IR_TENSOR_STORAGE_H__
