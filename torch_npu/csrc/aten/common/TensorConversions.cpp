#include <ATen/ATen.h>
#include <ATen/ops/_to_sparse_native.h>

#include <optional>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::_to_sparse(const at::Tensor& self, int64_t sparse_dim)
{
    return at::native::dense_to_sparse(self, sparse_dim);
}

at::Tensor NPUNativeFunctions::_to_sparse(
    const at::Tensor& self,
    ::std::optional<at::Layout> layout,
    at::OptionalIntArrayRef blocksize,
    ::std::optional<int64_t> dense_dim)
{
    return at::native::dense_to_sparse(self, layout, blocksize, dense_dim);
}

} // namespace native
} // namespace at_npu
