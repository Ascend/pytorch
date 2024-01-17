#include <ATen/NativeFunctions.h>
#include <torch/library.h>

namespace {
TORCH_LIBRARY_IMPL(aten, SparsePrivateUse1, m) {
    m.impl("_sparse_coo_tensor_with_dims_and_tensors", at::native::new_with_dims_and_tensor_sparse_symint);
    m.impl("_nnz", at::native::_nnz_sparse);
    m.impl("_indices", at::native::_indices_sparse);
    m.impl("_values", at::native::_values_sparse);
    m.impl("sparse_dim", at::native::sparse_dim_sparse);
    m.impl("dense_dim", at::native::dense_dim_sparse);
    m.impl("empty.memory_format", at::native::empty_sparse);
    m.impl("is_coalesced", at::native::is_coalesced_sparse);
    m.impl("_coalesced_", at::native::_coalesced_sparse_);
    m.impl("copy_", at::native::copy_sparse_wrapper_);
    m.impl("copy_sparse_to_sparse_", at::native::copy_sparse_);
    m.impl("neg", at::native::neg_sparse);
    m.impl("neg_", at::native::neg_sparse_);
    m.impl("neg.out", at::native::neg_out_sparse);
}
}
