#ifndef __IR_TENSOR_TENSOR_H__
#define __IR_TENSOR_TENSOR_H__

#include <cstdint>
#include <iostream>
#include <vector>
#include <utility>
#include <string>
#include <numeric>

#include "hardware/device.h"
#include "ir/common/dtype.h"
#include "ir/common/intrusive_ptr.h"
#include "ir/tensor/storage.h"
#include "ir/tensor/format.h"
#include "ir/symbolic/symbolic.h"

namespace fxrt {
namespace ir {
class Tensor; // Forward declaration
using TensorPtr = IntrusivePtr<Tensor>;

/**
 * @brief A multi-dimensional array (tensor).
 *
 * This class holds the metadata of a tensor, such as its dimensions, data type,
 * and a reference to the underlying storage.
 */
class Tensor : public RefCounted {
 public:
  /**
   * @brief Constructs an empty Tensor with uninitialized data.
   * A new storage is allocated for the tensor.
   * @param shape The dimensions of the tensor.
   * @param dtype The data type of the tensor.
   * @param device The device to allocate the tensor on.
   * @return The newly created tensor.
   */
  Tensor(const std::vector<int64_t>& shape, DataType dtype, hardware::Device device);
  /**
   * @brief Constructs a Tensor from an existing Storage.
   * @param storage The underlying storage for the tensor data.
   * @param dtype The data type of the tensor elements.
   * @param shape The dimensions of the tensor.
   */
  Tensor(StoragePtr storage, const std::vector<int64_t>& shape, DataType dtype);
  /**
   * @brief Constructs a Tensor from an existing data blob.
   * The tensor does not own the memory.
   * @param data Pointer to the data.
   * @param shape The dimensions of the tensor.
   * @param dtype The data type of the tensor.
   * @param device The device where the data is located.
   * @return The newly created tensor.
   */
  Tensor(void* data, const std::vector<int64_t>& shape, DataType dtype, hardware::Device device);

  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&&) = delete;
  Tensor& operator=(Tensor&&) = delete;

  /**
   * @brief Gets the data type of the tensor.
   * @return The data type.
   */
  DataType Dtype() const {
    return dtype_;
  }
  /**
   * @brief Gets the dimensions of the tensor.
   * @return A const reference to the vector of dimensions.
   */
  const std::vector<int64_t>& Shape() const {
    return shape_;
  }
  /**
   * @brief Gets the dimensions of the tensor.
   * @return A mutable reference to the vector of dimensions.
   */
  std::vector<int64_t>& Shape() {
    return shape_;
  }
  /**
   * @brief Gets the memory format of the tensor.
   * @return The memory format enum value.
   */
  MemoryFormat Format() const {
    return memoryFormat_;
  }
  /**
   * @brief Gets the strides of the tensor.
   * @return A const reference to the vector of strides.
   */
  const std::vector<int64_t>& Strides() const {
    return strides_;
  }
  /**
   * @brief Set the Strides of the tensor.
   * @param strides A const reference to the vector of strides.
   */
  void SetStrides(const std::vector<int64_t>& strides) {
    strides_ = strides;
  }
  /**
   * @brief Checks if the tensor is contiguous in memory.
   * @return true if the tensor is contiguous, false otherwise.
   */
  bool IsContiguous() const;
  /**
   * @brief Gets the number of dimensions of the tensor.
   * @return The number of dimensions.
   */
  int64_t Dim() const {
    return shape_.size();
  }
  /**
   * @brief Gets the total number of elements in the tensor.
   * @return The number of elements, or -1 for dynamic shapes.
   */
  int64_t Numel() const {
    return numel_;
  }
  /**
   * @brief Checks if the tensor has a dynamic shape.
   * @return true if the shape is dynamic, false otherwise.
   */
  bool HasDynamicShape() const {
    return numel_ < 0;
  }

  /**
   * @brief Checks if the tensor has a symbolic shape.
   * @return true if the shape is symbolic, false otherwise.
   */
  bool HasSymbolicShape() const {
    return !symbolicShape_.empty();
  }
  /**
   * @brief Evaluates the symbolic shape and stores it in the concrete shape.
   */
  void EvalSymbolicShape();
  /**
   * @brief Gets the symbolic shape of the tensor.
   * @return A const reference to the vector of symbolic shape expressions.
   */
  const std::vector<SymbolicExprPtr>& GetSymbolicShape() const {
    return symbolicShape_;
  }
  /**
   * @brief Sets the symbolic shape of the tensor.
   * @param shape The new symbolic shape to set.
   */
  void SetSymbolicShape(const std::vector<SymbolicExprPtr>& shape);
  /**
   * @brief Gets the device where the tensor data is stored.
   * @return The device.
   */
  hardware::Device GetDevice() const {
    return storage_->GetDevice();
  }
  /**
   * @brief Gets the underlying storage of the tensor.
   * @return The storage.
   */
  const StoragePtr& GetStorage() const {
    return storage_;
  }
  /**
   * @brief Gets the storage offset of the tensor.
   * @return The storage offset.
   */
  int64_t StorageOffset() const {
    return storageOffset_;
  }
  /**
   * @brief Set the Storage Offset of the tensor.
   * @param storageOffset The storage offset.
   */
  void SetStorageOffset(int64_t storageOffset) {
    storageOffset_ = storageOffset;
  }
  /**
   * @brief Get the storage shape of the tensor.
   * @return The storage shape.
   */
  const std::vector<int64_t>& StorageShape() const {
    return storageShape_;
  }
  /**
   * @brief Set the Storage Shape of the tensor.
   * @param storageShape The storage shape to be set.
   */
  void SetStorageShape(const std::vector<int64_t>& storageShape) {
    storageShape_ = storageShape;
  }
  /**
   * @brief Resizes the storage of the tensor.
   * Note: The shape and dtype must be set before resizing the storage.
   */
  void Resize();
  /**
   * @brief Updates the data of the tensor.
   * @param data Pointer to the new data.
   */
  void UpdateData(void* data);
  /**
   * @brief Gets a raw const pointer to the tensor's data.
   * This pointer takes into account the storage offset.
   * @return A const void pointer to the data.
   */
  const void* DataPtr() const {
    if (numel_ == 0) {
      return nullptr;
    }
    if (storage_->Data() == nullptr) {
      return nullptr;
    }
    const auto offsetBytes = static_cast<size_t>(storageOffset_) * dtype_.GetSize();
    CHECK_IF_FAIL(offsetBytes < storage_->SizeBytes());
    return static_cast<const char*>(storage_->Data()) + offsetBytes;
  }
  /**
   * @brief Gets a raw pointer to the tensor's data.
   * This pointer takes into account the storage offset.
   * @return A void pointer to the data.
   */
  void* DataPtr() {
    return const_cast<void*>(static_cast<const Tensor*>(this)->DataPtr());
  }
  /**
   * @brief Sets the data type of the tensor.
   * @param dtype The new data type to set.
   */
  void SetDtype(DataType dtype) {
    dtype_ = dtype;
  }
  /**
   * @brief Sets the shape of the tensor.
   * @param shape The new shape to set.
   */
  void SetShape(const std::vector<int64_t>& shape) {
    shape_ = shape;
  }
  /**
   * @brief Sets the shape of the tensor.
   * @param shape The new shape to set.
   */
  void SetShape(std::vector<int64_t>&& shape) {
    shape_ = std::move(shape);
  }
  /**
   * @brief Sets the storage of the tensor.
   * @param storage The new storage to set.
   */
  void SetStorage(const StoragePtr& storage) {
    storage_ = storage;
  }
  /**
   * @brief Check whether this tensor currently owns the Storage.
   */
  bool CheckOwnsStorage() const {
    return ownsStorage_;
  }
  /**
   * @brief Sets whether this tensor owns the storage.
   * @param val True if the tensor should own the storage, false otherwise.
   */
  void SetOwnsStorage(bool ownsStorage) {
    ownsStorage_ = ownsStorage;
  }
  /**
   * @brief Sets the memory format of the tensor.
   * @param memoryFromat The memory format enum value.
   */
  void SetFormat(MemoryFormat memoryFormat) {
    memoryFormat_ = memoryFormat;
  }

  IntrusivePtr<Tensor> ShallowClone() const;

  /**
   * @brief Function type for updating a tensor.
   * @param tensor The tensor to update.
   */
  using TensorUpdater = std::function<void(Tensor*)>;

  /**
   * @brief Sets a tensor updater function.
   * @param updater The tensor updater function.
   */
  void SetUpdater(TensorUpdater&& updater) {
    updater_ = std::move(updater);
  }

  /**
   * @brief Updates the tensor using the updater function.
   */
  void Update() {
    if (updater_ != nullptr) {
      updater_(this);
      updater_ = nullptr;
    }
  }

  /**
   * @brief Creates a deep copy of this Tensor object.
   * @return A new Tensor object with copied data and metadata.
   * The new Storage will own its data (ownsData_ = true).
   */
  TensorPtr DeepCopy() const;

 private:
  /**
   * @brief Computes the strides from the dimensions.
   */
  void ComputeStrides();

  DataType dtype_; ///< The data type of the elements.
  std::vector<int64_t> shape_; ///< The dimensions of the tensor.
  std::vector<SymbolicExprPtr> symbolicShape_; ///< The symbolic dimensions of the tensor.
  std::vector<int64_t> strides_; ///< The strides of the tensor.
  MemoryFormat memoryFormat_{FORMAT_ND}; ///< The memory format of the tensor.
  int64_t numel_ = 0; ///< The total number of elements.
  StoragePtr storage_{nullptr}; ///< The underlying storage.
  std::vector<int64_t> storageShape_; ///< The underlying storage shape of the tensor.
  int64_t storageOffset_ = 0; ///< The offset in the storage, in number of elements.
  bool ownsStorage_{true}; ///< Whether the tensor owns the storage.
  TensorUpdater updater_{nullptr}; ///< The tensor updater function.
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
std::ostream& operator<<(std::ostream& os, const TensorPtr& tensor);
std::string ShapeToString(const std::vector<int64_t>& shape);
} // namespace ir
} // namespace fxrt

#endif // __IR_TENSOR_TENSOR_H__
