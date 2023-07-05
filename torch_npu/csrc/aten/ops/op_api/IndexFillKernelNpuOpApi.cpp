// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {

constexpr int MAX_DIM = 1;

static std::vector<int64_t> GetIndexVector(const at::Tensor& index) {
  std::vector<int64_t> idx_vec;
  if (index.dim() == 0) {
    idx_vec.emplace_back(static_cast<int64_t>(index.item().to<int>()));
  } else {
    for (int64_t i = 0; i < index.sizes()[0]; i++) {
      int64_t idx = static_cast<int64_t>(index[i].item().to<int>());
      idx_vec.emplace_back(idx);
    }
  }
  return idx_vec;
}

at::Tensor& NPUNativeOpApiFunctions::index_fill_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                                                 const at::Tensor& value) {
  DO_COMPATIBILITY(aclnnInplaceIndexFillTensor, NPUNativeFunctions::index_fill_(self, dim, index, value));
  TORCH_CHECK(value.dim() == 0, "Value should be a 0-dimensional tensor, but got ", value.dim());
  TORCH_CHECK(index.dim() <= MAX_DIM, "Index has to be a vector/scalar.");

  at::Scalar value_scalar = value.item();
  std::vector<int64_t> idx_vec = GetIndexVector(index);
  at::IntArrayRef index_array = at::IntArrayRef(idx_vec);
  
  EXEC_NPU_CMD(aclnnInplaceIndexFillTensor, self, dim, index_array, value_scalar);
  return self;
}

at::Tensor NPUNativeOpApiFunctions::index_fill(const at::Tensor& self, int64_t dim, const at::Tensor& index,
                                               const at::Tensor& value) {
  DO_COMPATIBILITY(aclnnIndexFillTensor, NPUNativeFunctions::index_fill(self, dim, index, value));
  TORCH_CHECK(value.dim() == 0, "Value should be a 0-dimensional tensor, but got ", value.dim());
  TORCH_CHECK(index.dim() <= MAX_DIM, "Index has to be a vector/scalar.");

  at::Scalar value_scalar = value.item();
  std::vector<int64_t> idx_vec = GetIndexVector(index);
  at::IntArrayRef index_array = at::IntArrayRef(idx_vec);

  // construct output tensor
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self);

  EXEC_NPU_CMD(aclnnIndexFillTensor, self, dim, index_array, value_scalar, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::index_fill_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                                                 const at::Scalar& value) {
  DO_COMPATIBILITY(aclnnInplaceIndexFillTensor, NPUNativeFunctions::index_fill_(self, dim, index, value));
  TORCH_CHECK(index.dim() <= MAX_DIM, "Index has to be a vector/scalar.");

  std::vector<int64_t> idx_vec = GetIndexVector(index);
  at::IntArrayRef index_array = at::IntArrayRef(idx_vec);

  EXEC_NPU_CMD(aclnnInplaceIndexFillTensor, self, dim, index_array, value);
  return self;
}

at::Tensor NPUNativeOpApiFunctions::index_fill(const at::Tensor& self, int64_t dim, const at::Tensor& index,
                                               const at::Scalar& value) {
  DO_COMPATIBILITY(aclnnIndexFillTensor, NPUNativeFunctions::index_fill(self, dim, index, value));
  TORCH_CHECK(index.dim() <= MAX_DIM, "Index has to be a vector/scalar.");

  std::vector<int64_t> idx_vec = GetIndexVector(index);
  at::IntArrayRef index_array = at::IntArrayRef(idx_vec);

  // construct output tensor
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self);

  EXEC_NPU_CMD(aclnnIndexFillTensor, self, dim, index_array, value, result);
  return result;
}

}  // namespace native
}  // namespace at_npu
