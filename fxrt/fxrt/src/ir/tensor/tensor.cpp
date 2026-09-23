#include <stdexcept>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>

#include "common/common.h"
#include "ir/tensor/tensor.h"
#include "ir/symbolic/symbolic.h"
#include "ir/tensor/format.h"
#include "ir/common/intrusive_ptr.h"

namespace fxrt {
namespace ir {

namespace {
int64_t CalculateNumel(const std::vector<int64_t>& shape, bool allow_dynamic) {
  int64_t numel = 1;
  for (const auto& dim : shape) {
    if (dim < 0) {
      if (allow_dynamic) {
        return -1;
      } else {
        RT_GLOG(EXCEPTION) << "Creating Tensor from existing data does not support dynamic shapes.";
      }
    }
    numel *= dim;
  }
  return numel;
}

template <typename T>
void PrintData(std::ostream& os, const void* data, size_t numel, size_t limit) {
  const auto* d = static_cast<const T*>(data);
  for (size_t i = 0; i < std::min(numel, limit); ++i) {
    // Promote char types to int for printing
    os << +d[i];
    if (i < std::min(numel, limit) - 1) {
      os << ", ";
    }
  }
  if (numel > limit) {
    os << ", ...";
  }
}

} // namespace

/**
 * @brief Computes the strides of the tensor based on its dimensions.
 * The strides are computed for a contiguous tensor in row-major order.
 * If the shape is dynamic, strides after the dynamic dimension will be -1.
 */
void Tensor::ComputeStrides() {
  if (shape_.empty()) {
    return;
  }
  strides_.resize(shape_.size());
  int64_t stride = 1;
  for (int i = shape_.size() - 1; i >= 0; --i) {
    strides_[i] = stride;
    if (stride != -1) {
      if (shape_[i] < 0) {
        stride = -1;
      } else {
        stride *= shape_[i];
      }
    }
  }
}

bool Tensor::IsContiguous() const {
  if (strides_.empty()) {
    return true;
  }
  if (shape_.size() != strides_.size()) {
    return false;
  }
  int64_t cumulatedStride = 1;
  int64_t shapeLen = static_cast<int64_t>(shape_.size());
  for (int64_t i = shapeLen - 1; i >= 0; --i) {
    if (strides_[i] != cumulatedStride) {
      return false;
    }
    cumulatedStride *= shape_[i];
  }
  return true;
}

Tensor::Tensor(const std::vector<int64_t>& shape, DataType dtype, hardware::Device device)
    : dtype_(dtype), shape_(shape) {
  numel_ = CalculateNumel(shape_, true);
  size_t sizeBytes = 0;
  if (!HasDynamicShape()) {
    sizeBytes = numel_ * dtype_.GetSize();
  }

  storage_ = MakeIntrusive<Storage>(sizeBytes, device);
}

void Tensor::Resize() {
  CHECK_IF_NULL(storage_);
  numel_ = CalculateNumel(shape_, false);
  if (ownsStorage_) {
    size_t sizeBytes = numel_ * dtype_.GetSize();
    storage_->Resize(sizeBytes);
  }
}

void Tensor::UpdateData(void* data) {
  storage_->SetData(data);
}

Tensor::Tensor(StoragePtr storage, const std::vector<int64_t>& shape, DataType dtype)
    : dtype_(dtype), shape_(shape), storage_(storage) {
  numel_ = CalculateNumel(shape_, true);
  if (!HasDynamicShape()) {
    if (storage_->SizeBytes() < numel_ * dtype_.GetSize()) {
      RT_GLOG(EXCEPTION) << "Storage size is smaller than required by tensor dimensions and data type.";
    }
  }
}

Tensor::Tensor(void* data, const std::vector<int64_t>& shape, DataType dtype, hardware::Device device)
    : dtype_(dtype), shape_(shape) {
  numel_ = CalculateNumel(shape_, false);
  size_t sizeBytes = numel_ * dtype_.GetSize();

  storage_ = MakeIntrusive<Storage>(data, sizeBytes, device);
}

void Tensor::EvalSymbolicShape() {
  if (!HasSymbolicShape()) {
    return;
  }
  for (size_t i = 0; i < symbolicShape_.size(); ++i) {
    shape_[i] = symbolicShape_[i]->Evaluate();
  }
  Resize();
}

void Tensor::SetSymbolicShape(const std::vector<SymbolicExprPtr>& shape) {
  symbolicShape_ = shape;
  shape_.resize(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    if (auto c = dynamic_cast<const SymbolicConst*>(shape[i].get())) {
      shape_[i] = c->GetValue();
    } else {
      shape_[i] = -1;
    }
  }
  numel_ = CalculateNumel(shape_, true);
}

TensorPtr Tensor::ShallowClone() const {
  // Shallow copy: create new tensor sharing the same storage (view)
  // The Tensor constructor with StoragePtr takes the StoragePtr by value,
  // which calls IntrusivePtr's copy constructor. This only increments the
  // reference count of the Storage object, it does NOT create a new Storage.
  // Therefore, the cloned tensor shares the same underlying storage as the original.
  auto clonedTensor = MakeIntrusive<Tensor>(storage_, shape_, dtype_);

  // Copy all metadata
  clonedTensor->strides_ = strides_;
  clonedTensor->memoryFormat_ = memoryFormat_;
  clonedTensor->storageShape_ = storageShape_;
  clonedTensor->storageOffset_ = storageOffset_;
  clonedTensor->symbolicShape_ = symbolicShape_;

  // Verify that the cloned tensor shares the same storage
  // (This is a sanity check - the storage pointers should be the same)
  CHECK_IF_FAIL(clonedTensor->GetStorage().get() == storage_.get());

  return clonedTensor;
}

TensorPtr Tensor::DeepCopy() const {
  // Create new Storage without copying data, but ensure ownsData_ is True
  size_t sizeBytes = 0;
  if (!HasDynamicShape()) {
    sizeBytes = numel_ * dtype_.GetSize();
  }

  // Use constructor Storage(size_t sizeBytes, hardware::Device device), which sets ownsData_=true
  auto new_storage = MakeIntrusive<Storage>(sizeBytes, GetDevice());

  // Create new Tensor object
  auto new_tensor = MakeIntrusive<Tensor>(new_storage, shape_, dtype_);

  // Copy metadata
  new_tensor->SetStrides(strides_);
  new_tensor->SetFormat(memoryFormat_);
  new_tensor->SetStorageShape(storageShape_);
  new_tensor->SetStorageOffset(storageOffset_);

  // Deep copy symbolic shape (recursively copy SymbolicExpr)
  if (!symbolicShape_.empty()) {
    new_tensor->SetSymbolicShape(symbolicShape_);
  }

  return new_tensor;
}

std::ostream& operator<<(std::ostream& os, const TensorPtr& tensor) {
  if (!tensor) {
    os << "Null";
  } else {
    os << *tensor;
  }
  return os;
}

std::string ShapeToString(const std::vector<int64_t>& shape) {
  std::string str = "[";
  const size_t count = shape.size();
  for (size_t i = 0; i < count; ++i) {
    if (i > 0) {
      str.append(", ");
    }
    str.append(std::to_string(shape[i]));
  }
  return str.append("]");
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
  constexpr size_t numelLimit = 30;
  os << "Tensor(shape=";
  const auto& shape = tensor.Shape();
  os << ShapeToString(shape);
  if (tensor.HasSymbolicShape()) {
    os << ", sym_shape=[";
    for (auto& dim : tensor.GetSymbolicShape()) {
      if (&dim != &tensor.GetSymbolicShape().front()) {
        os << ", ";
      }
      os << dim->ToString();
    }
    os << "]";
  }
  os << ", numel: " << tensor.Numel();
  os << ", strides: " << ShapeToString(tensor.Strides());
  os << ", contiguous=" << (tensor.IsContiguous() ? "true" : "false");
  os << ", dtype=" << tensor.Dtype().ToString();
  os << ", storageShape: " << ShapeToString(tensor.StorageShape());
  os << ", offset: " << tensor.StorageOffset();
  os << ", device=[type=" << hardware::GetDeviceNameByType(tensor.GetDevice().type)
     << ", index:" << int(tensor.GetDevice().index) << "]";
  os << ", data=[";
  if (tensor.DataPtr()) {
    if (tensor.GetDevice().type != hardware::DeviceType::CPU) {
      os << tensor.DataPtr();
    } else if (tensor.HasDynamicShape()) {
      os << "dynamic shape, not materialized";
    } else if (tensor.Numel() > 0) {
      switch (tensor.Dtype()) {
        case DataType::Float32:
          PrintData<float>(os, tensor.DataPtr(), tensor.Numel(), numelLimit);
          break;
        case DataType::Float64:
          PrintData<double>(os, tensor.DataPtr(), tensor.Numel(), numelLimit);
          break;
        case DataType::Int8:
          PrintData<int8_t>(os, tensor.DataPtr(), tensor.Numel(), numelLimit);
          break;
        case DataType::Int16:
          PrintData<int16_t>(os, tensor.DataPtr(), tensor.Numel(), numelLimit);
          break;
        case DataType::Int32:
          PrintData<int32_t>(os, tensor.DataPtr(), tensor.Numel(), numelLimit);
          break;
        case DataType::Int64:
          PrintData<int64_t>(os, tensor.DataPtr(), tensor.Numel(), numelLimit);
          break;
        case DataType::UInt8:
          PrintData<uint8_t>(os, tensor.DataPtr(), tensor.Numel(), numelLimit);
          break;
        case DataType::Bool:
          os << std::boolalpha;
          PrintData<bool>(os, tensor.DataPtr(), tensor.Numel(), numelLimit);
          break;
        default:
          os << "...";
          break;
      }
    }
  } else {
    os << "null";
  }
  os << "]";
  os << ", format=" << FormatEnumToStr(tensor.Format()) << ")";
  os << " storage: ";
  os << tensor.GetStorage().get();
  return os;
}
} // namespace ir
} // namespace fxrt
