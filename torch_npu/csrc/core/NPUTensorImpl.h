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


#pragma once

#include <ATen/Tensor.h>
#include <c10/core/TensorImpl.h>
#include "torch_npu/csrc/core/NPUStorageImpl.h"

namespace torch_npu {

// NPUTensorImpl class is derived from c10::TensorImpl, and it is only used to handle an NPU tensor.
// Its scope is just to handle an NPUTensor.
class NPUTensorImpl : public c10::TensorImpl {
public:
  explicit NPUTensorImpl(c10::Storage&& storage, const c10::intrusive_ptr<c10::StorageImpl> storage_impl, const caffe2::TypeMeta& data_type);

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) final;

  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const final;
  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const final;

public:
  NPUTensorImpl(const NPUTensorImpl&) = delete;
  NPUTensorImpl& operator=(const NPUTensorImpl&) = delete;
  NPUTensorImpl(NPUTensorImpl&&) = default;
  NPUTensorImpl& operator=(NPUTensorImpl&&) = default;
  ~NPUTensorImpl();

private:
  c10::intrusive_ptr<c10::StorageImpl> _storage_impl;

};

}  // namespace torch_npu
