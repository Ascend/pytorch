// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include <ATen/native/npu/contiguous/ReshapeOpt.h>

namespace at {
namespace native {
namespace npu {

class ReshapeV2ContiguousOpt : public ContiguousOpt {
 public:
  bool Optimizer(const Tensor& src, Tensor& self) override {
    if (check_reshape_match(src, self)) {
      if ((!c10::npu::NpuRunMode::IsGraphMode()) && can_use_memory_repoint(src) &&
          reshape_match_by_memory_repoint(src, self)) {
        return true;
      }
      RECORD_HOST_FUNCTION("View_d2dCopyAsync", std::vector<c10::IValue>({src}));
      at::npu_reshape_out(self, src, self.sizes());
      return true;
    }
    return false;
  }

  bool CanOptimizer(const Tensor& src) override {
    return check_reshape_match(src);
  }

 private:
  template <typename dataDtype>
  void ResetDataPtr(const Tensor& src, Tensor& self, dataDtype* value) {
    dataDtype* src_data_ptr = value + src.storage_offset();
    at::DataPtr self_data_ptr =
        at::DataPtr(src_data_ptr, self.storage().device());
    self.storage().set_data_ptr(std::move(self_data_ptr));
  }

  bool reshape_match_by_memory_repoint(const Tensor& src, Tensor& self) {
    // Memory-repointing optimization hasn't been fully demonstrated, Only FP16
    // and FP32 are supported.
    RECORD_HOST_FUNCTION("memory_repoint", std::vector<c10::IValue>({src}));
    switch (src.scalar_type()) {
      case at::ScalarType::Half:
        ResetDataPtr(
            src, self, static_cast<at::Half*>(src.storage().data_ptr().get()));
        return true;
      case at::ScalarType::Float:
        ResetDataPtr(
            src, self, static_cast<float*>(src.storage().data_ptr().get()));
        return true;
      case at::ScalarType::Byte:
        ResetDataPtr(
            src, self, static_cast<uint8_t*>(src.storage().data_ptr().get()));
        return true;
      case at::ScalarType::Char:
        ResetDataPtr(
            src, self, static_cast<int8_t*>(src.storage().data_ptr().get()));
        return true;
      case at::ScalarType::Short:
        ResetDataPtr(
            src, self, static_cast<int16_t*>(src.storage().data_ptr().get()));
        return true;
      case at::ScalarType::Int:
        ResetDataPtr(
            src, self, static_cast<int*>(src.storage().data_ptr().get()));
        return true;
      case at::ScalarType::Long:
        ResetDataPtr(
            src, self, static_cast<int64_t*>(src.storage().data_ptr().get()));
        return true;
      default:
        // Turn to conducting d2dCopyAsync for other dtypes.
        return false;
    }
  }

  bool can_use_memory_repoint(const Tensor& tensor) {
    auto tensorNpuDesc = tensor.storage().get_npu_desc();
    if (FormatHelper::IsBaseFormatType(tensor)) {
      return true;
    }

    if (tensorNpuDesc.npu_format_ == ACL_FORMAT_FRACTAL_NZ) {
      // No padding
      if ((tensor.size(-1) % 16 == 0) && (tensor.size(-2) % 16 == 0)) {
        return true;
      }
      return false;
    }
    return false;
  }
}; // class ReshapeV2ContiguousOpt

REGISTER_COPY_OPT(reshapeV2, ReshapeV2ContiguousOpt)

} // namespace npu
} // namespace native
} // namespace at