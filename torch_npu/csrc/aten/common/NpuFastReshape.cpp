#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/InferFormat.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

void npu_fast_reshape_(at::Tensor& tensor)
{
    /**
      [NOTE] For some reshape cases such as view, unsqueeze, squeeze, flatten,
      storages of them remain unchanged. So we can refresh reshape tensor's metadata
      to obtain matched tensor.
      */

    // restriction 1
    if (!tensor.is_contiguous()) {
        return;
    }
    // restriction 2
    if (!FormatHelper::IsBaseFormatType(tensor)) {
        return;
    }
    // restriction 3: reshape case without any numels change
    if ((tensor.numel() != StorageDescHelper::GetMemorySize(tensor)) ||
        StorageDescHelper::MetaDataAreMatch(&tensor)) {
        return;
    }

    // refresh matadata to input tensor
    StorageDescHelper::ReflushDescBySelf(tensor);
    auto base_format = InferFormat::GuessBaseFormat(tensor.sizes());
    NPUNativeFunctions::npu_format_cast_(tensor, base_format);
}

} // namespace native
} // namespace at_npu
