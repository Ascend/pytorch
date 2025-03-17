#include <torch/library.h>
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

// True if `self` and `from` have compatible tensor type so that `from`'s
// TensorImpl can be copied to `self`.
bool _has_compatible_shallow_copy_type(
    const at::Tensor &self,
    const at::Tensor &from)
{
    c10::DispatchKeySet self_key = self.key_set();
    c10::DispatchKeySet from_key = from.key_set();
    auto is_dense = [](c10::DispatchKeySet ts) {
        return ts.has(c10::DispatchKey::CPU) || ts.has(c10::DispatchKey::PrivateUse1);
    };
    return (self_key == from_key) || (is_dense(self_key) && is_dense(from_key));
}

TORCH_LIBRARY_IMPL(aten, CatchAll, m) {
    m.impl("_has_compatible_shallow_copy_type", TORCH_FN(_has_compatible_shallow_copy_type));
}

} // namespace native
} // namespace at_npu
