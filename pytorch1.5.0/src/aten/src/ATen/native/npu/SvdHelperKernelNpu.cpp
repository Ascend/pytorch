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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

int64_t batch_count(const Tensor& batched_matrices) {
  int64_t result = 1;
  for (int64_t i = 0; i < batched_matrices.ndimension() - 2; i++) {
    result *= batched_matrices.size(i);
  }
  return result;
}

void single_check_errors(int64_t info, const char* name, bool allow_singular=false, int64_t batch_idx=-1) {
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

void batch_check_errors(std::vector<int64_t>& infos, const char* name, bool allow_singular=false) {
  for (size_t i = 0; i < infos.size(); i++) {
    auto info = infos[i];
    single_check_errors(info, name, allow_singular, i);
  }
}

std::tuple<Tensor, Tensor, Tensor> _svd_helper_npu(const Tensor& self, bool some, bool compute_uv) {
  TORCH_CHECK(self.dtype() == at::kFloat, "svd_npu only supported Float, but get", self.dtype());
  std::vector<int64_t> infos(batch_count(self), 0);
  int64_t m = self.size(-2);
  int64_t n = self.size(-1);
  int64_t k = std::min(m, n);

  Tensor U_working_copy, S_working_copy, VT_working_copy;
  auto sizes = self.sizes().vec();

  sizes[self.dim() - 1] = (compute_uv && some) ? std::min(m, n) : m;
  U_working_copy = OpPreparation::ApplyTensor(self, sizes);

  sizes[self.dim() - 2] = n;
  sizes[self.dim() - 1] = (compute_uv && some) ? k : n;
  VT_working_copy = OpPreparation::ApplyTensor(self, sizes);

  sizes.pop_back();
  sizes[self.dim() - 2] = std::min(m, n);
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

    if (self.dim() > 2) {
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

}
}
