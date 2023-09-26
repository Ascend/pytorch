#pragma once
#include<c10/core/StorageImpl.h>
#include"torch_npu/csrc/core/NPUTensorImpl.h"
#include"torch_npu/csrc/core/NPUStorageImpl.h"

namespace torch_npu {

class NPUBridge {
public:
  // at::tensor to NPUStorageImpl
  static NPUStorageImpl* GetNpuStorageImpl(const at::Tensor &tensor);

  // c10::StorageImpl to NPUStorageImpl
  static NPUStorageImpl* GetNpuStorageImpl(c10::StorageImpl* storageImpl);

  // c10::Storage to NPUStorageImpl
  static NPUStorageImpl* GetNpuStorageImpl(c10::Storage&& storage);

  // tensor to NPUStorageDesc
  static NPUStorageDesc& GetNpuStorageImplDesc(const at::Tensor &tensor);

  // tensor to NPUTensorImpl
  static NPUTensorImpl* GetNpuTensorImpl(const at::Tensor& tensor);
};
}