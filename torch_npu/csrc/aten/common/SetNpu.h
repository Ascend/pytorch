#ifndef SETNPU_H_
#define SETNPU_H_

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/common/ResizeNpu.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"

namespace at_npu {
namespace native {

at::Tensor set_tensor_with_storage_format(c10::Storage src);

} // namespace native
} // namespace at_npu
#endif // SETNPU_H
