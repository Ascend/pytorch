#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

std::vector<at::Tensor> NPUNativeFunctions::unflatten_dense_tensors(const at::Tensor& flat, at::TensorList tensors) {
  std::vector<at::Tensor> outputs;
  outputs.reserve(tensors.size());
  size_t offset = 0;
  for (const auto & tensor : tensors) {
    auto numel = NPUNativeFunctions::npu_format_cast(tensor,
        torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_.origin_format_).numel();
    // If unflatten an empty tensor, create a new empty tensor using
    // flat tensor Options.
    // This can avoid the unflattened empty tensor to share the same storage
    // with other unflatten tensors.
    if (numel == 0) {
      outputs.push_back(at::empty({0}, flat.options()));
    } else {
      outputs.push_back(flat.narrow(0, offset, numel).view(tensor.sizes()));
      offset += numel;
    }
  }
  return outputs;
}


}
}
