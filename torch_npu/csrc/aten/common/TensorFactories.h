#pragma once

#include <c10/core/TensorOptions.h>
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace at_npu {
namespace native {

inline void check_size_nonnegative(c10::IntArrayRef& size) {
  for (auto& x : size) {
    TORCH_CHECK(
        x >= 0,
        "Trying to create tensor with negative dimension ",
        x,
        ": ",
        size, OPS_ERROR(ErrCode::VALUE));
  }
}

inline void check_args(int64_t row, int64_t col, const c10::TensorOptions& options) {
    TORCH_CHECK(row >= 0, "row must be non-negative, got", row, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(col >= 0, "col must be non-negative, got", col, OPS_ERROR(ErrCode::VALUE));
    if (options.has_layout()) {
        TORCH_CHECK(
            options.layout() == at::kStrided,
            "only support layout=torch.strided, got",
            options.layout(), OPS_ERROR(ErrCode::TYPE));
    }
}

inline int64_t get_tril_size(int64_t row, int64_t col, int64_t offset) {
  // number of elements in the first row of the tril
  auto m_first_row = offset > 0 ?
    std::min<int64_t>(col, 1 + offset) : // upper bounded by col
    row + offset > 0; // either 0 or 1
  // number of elements in the last row of the tril, bounded by [0, col]
  auto m_last_row = std::max<int64_t>(0, std::min<int64_t>(col, row + offset));
  // number of rows, bounded by [0, row]
  auto n_row_all = std::max<int64_t>(0, std::min<int64_t>(row, row + offset));
  auto n_row_trapezoid = (m_last_row - m_first_row + 1);

  // calculate # of elements in the top trapezoid
  auto tril_size = (m_first_row + m_last_row) * n_row_trapezoid >> 1;

  // calculate # of elements in the bottom rectangle if there is any
  auto diff_row = n_row_all - n_row_trapezoid;
  if (diff_row > 0) {
    tril_size += diff_row * col;
  }

  return tril_size;
}

} // namespace native
} // namespace at_npu
