#include <typeinfo>

#include <torch_npu/csrc/core/NPUBridge.h>

namespace torch_npu {
NPUStorageImpl* NPUBridge::GetNpuStorageImpl(c10::StorageImpl* storageImpl) {
  return static_cast<NPUStorageImpl*>(storageImpl);
}

NPUStorageImpl* NPUBridge::GetNpuStorageImpl(c10::Storage&& storage) {
  return static_cast<NPUStorageImpl*>(storage.unsafeGetStorageImpl());
}

NPUStorageImpl* NPUBridge::GetNpuStorageImpl(const at::Tensor& tensor) {
  return static_cast<NPUStorageImpl*>(tensor.storage().unsafeGetStorageImpl());
}

NPUStorageDesc& NPUBridge::GetNpuStorageImplDesc(const at::Tensor& tensor) {
  // from_blob tensors (legacy serialization _write_file) carry a plain
  // c10::StorageImpl; reading npu_desc_ on them is out-of-bounds.
  auto* storage_impl = tensor.storage().unsafeGetStorageImpl();
  TORCH_CHECK(
      typeid(*storage_impl) == typeid(NPUStorageImpl),
      "The npu storage desc is unavailable: the tensor's storage is not an NPUStorageImpl.");
  return static_cast<NPUStorageImpl*>(storage_impl)->npu_desc_;
}

NPUTensorImpl* NPUBridge::GetNpuTensorImpl(const at::Tensor& tensor) {
  return static_cast<NPUTensorImpl*>(tensor.unsafeGetTensorImpl());
}
} // namespace torch_npu