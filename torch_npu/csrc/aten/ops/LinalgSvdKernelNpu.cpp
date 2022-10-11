// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
int64_t batch_count(const at::Tensor& batched_matrices) {
  int64_t result = 1;
  auto number = 2;
  for (int64_t i = 0; i < batched_matrices.ndimension() - number; i++) {
    result *= batched_matrices.size(i);
  }
  return result;
}

void single_check_errors(int64_t info, const char* name, bool allow_singular = false, int64_t batch_idx = -1) {
  std::string batch_info = "";
  if (batch_idx >= 0) {
      batch_info = ": For batch " + std::to_string(batch_idx);
  }
  if (info < 0) {
    AT_ERROR(name, batch_info, ": Argument ", -info, " has illegal value");
  } else if (info > 0) {
    if (strstr(name, "svd")) {
      AT_ERROR(name, ": the updating process of SBDSDC did not converge (error: ", info, ")");
    } else if (strstr(name, "symeig")) {
      AT_ERROR(name, batch_info, ": the algorithm failed to converge; ", info,
          " off-diagonal elements of an intermediate tridiagonal form did not converge to zero.");
    } else if (!allow_singular) {
      AT_ERROR(name, batch_info, ": U(", info, ",", info, ") is zero, singular U.");
    }
  }
}

void batch_check_errors(std::vector<int64_t>& infos, const char* name, bool allow_singular = false) {
  for (size_t i = 0; i < infos.size(); i++) {
    auto info = infos[i];
    single_check_errors(info, name, allow_singular, i);
  }
}

/*
 * Clones a Tensor so that the following conditions hold:
 * If we think of a Tensor of having size (B, M, N), where B is any number
 * of batch dimensions, then:
 * - Each (M, N) matrix is in column major form
 * - Let Tensor P have size (B, M, N) and Q have size (B, M', N').
 *   Then when laid out in memory, the M by N matrix starting at
 *   P.data_ptr()[B * M * N] is of the same corresponding batch as the M' by N'
 *   matrix starting at Q.data_ptr()[B * M' * N'].
 */
static inline at::Tensor cloneBatchedColumnMajor(const at::Tensor& src) {
  // If src is already in batched column major format, then
  // this will be efficient (no reordering of the data will occur)
  // because the first transpose will make the tensor contiguous,
  // and cloning a contiguous tensor is fast.
  auto result = src.mT().clone(at::MemoryFormat::Contiguous);
  result.transpose_(-2, -1);
  return result;
}

/*
 * contig chooses between C-contig (true) and F-contig (false)
 */
static inline c10::MaybeOwned<at::Tensor> borrow_else_clone(const bool cond, const at::Tensor& borrow, const at::Tensor& clone, const bool contig) {
  return cond ? c10::MaybeOwned<at::Tensor>::borrowed(borrow)
              : c10::MaybeOwned<at::Tensor>::owned(contig ? clone.clone(at::MemoryFormat::Contiguous)
                                                      : cloneBatchedColumnMajor(clone));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _svd_helper(const at::Tensor& self, bool some, bool compute_uv) {
  TORCH_CHECK(self.dtype() == at::kFloat, "svd_npu only supported Float, but get", self.dtype());
  std::vector<int64_t> infos(batch_count(self), 0);
  int64_t m = self.size(-2);
  int64_t n = self.size(-1);
  int64_t k = std::min(m, n);

  at::Tensor U_working_copy, S_working_copy, VT_working_copy;
  auto sizes = self.sizes().vec();
  
  auto number_a = 2;
  auto number_b = 1;
  sizes[self.dim() - number_b] = (compute_uv && some) ? std::min(m, n) : m;
  U_working_copy = OpPreparation::ApplyTensor(self, sizes);
  
  sizes[self.dim() - number_a] = n;
  sizes[self.dim() - number_b] = (compute_uv && some) ? k : n;
  VT_working_copy = OpPreparation::ApplyTensor(self, sizes);

  sizes.pop_back();
  sizes[self.dim() - number_a] = std::min(m, n);
  S_working_copy = OpPreparation::ApplyTensor(self, sizes);

  if (self.numel() > 0) {
    OpCommand cmd;
    cmd.Name("Svd")
      .Input(self)
      .Output(S_working_copy)
      .Output(U_working_copy)
      .Output(VT_working_copy)
      .Attr("compute_uv", compute_uv)
      .Attr("full_matrices", !some)
      .Run();

    if (self.dim() > number_a) {
      batch_check_errors(infos, "svd_npu");
    } else {
      single_check_errors(infos[0], "svd_npu");
    }

    if (!compute_uv) {
      VT_working_copy.zero_();
      U_working_copy.zero_();
    }
  } else {
    U_working_copy.zero_();
    VT_working_copy.zero_();
  }

  return std::make_tuple(U_working_copy, S_working_copy, VT_working_copy);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> NPUNativeFunctions::_linalg_svd_out(
                                 const at::Tensor& A,
                                 const bool full_matrices,
                                 const bool compute_uv,
                                 at::Tensor& U,
                                 at::Tensor& S,
                                 at::Tensor& Vh) {
  // Half optimisation half precondition for some parts of the LAPACK / cuSOLVER
  // In particular, the call to lapackSvd to compute lwork fails otherwise
  if (A.numel() == 0) {
    // Needed in the case that we have e.g. A.shape == (3, 0) and full_matrices=True
    // We fill U or Vh with the identity matrix as it's a valid SVD for the empty matrix
    if (compute_uv && full_matrices) {
      if (U.numel() != 0) {
        U.zero_();
        U.diagonal(0, -2, -1).fill_(1.);
      }
      if (Vh.numel() != 0) {
        Vh.zero_();
        Vh.diagonal(0, -2, -1).fill_(1.);
      }
    }
    return std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>(U, S, Vh);
  }

  const bool use_cusolver = false;

  // A always needs to be copied as its contents will be destroyed during the computaton of the SVD
  // Now, MAGMA needs the copy to be on CPU, while cuSOLVER needs it to be on CUDA, so we'll defer
  // the copy as a column major matrix to the backends.
  const auto info = at::zeros(at::IntArrayRef(A.sizes().begin(), A.sizes().end() - 2), A.options().dtype(at::kInt));

  // Prepare S
  const auto S_ = S.expect_contiguous();

  // Prepare U / Vh
  // U_ and Vh_ are just going to be accessed whenever compute_uv == true
  const auto U_ready = !compute_uv || U.mT().is_contiguous();
  const auto U_ = borrow_else_clone(U_ready, U, U, /*C-contig*/false);
  const auto Vh_ready = !compute_uv
                            || (!use_cusolver && Vh.mT().is_contiguous())
                            || (use_cusolver && Vh.is_contiguous());
  const auto Vh_ = borrow_else_clone(Vh_ready, Vh, Vh, /*C-contig*/use_cusolver);

  at::Tensor U_tmp, S_tmp, V_tmp;
  std::tie(U_tmp, S_tmp, V_tmp) = _svd_helper(A, full_matrices, compute_uv);

  if (!U_ready) {
    U.copy_(U_tmp);
  }
  if (!S.is_same(S_tmp)) {
    S.copy_(S_tmp);
  }
  if (!Vh_ready) {
    Vh.copy_(V_tmp);
  }

  at::_linalg_check_errors(info, "linalg.svd", /*is_matrix*/A.dim() == 2);
}

} // namespace native
} // namespace at_npu
