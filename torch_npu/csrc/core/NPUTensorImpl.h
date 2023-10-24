#pragma once

#include <ATen/Tensor.h>
#include <c10/core/TensorImpl.h>
#include "torch_npu/csrc/core/NPUStorageImpl.h"

namespace torch_npu {

// NPUTensorImpl class is derived from c10::TensorImpl, and it is only used to handle an NPU tensor.
// Its scope is just to handle an NPUTensor.
class NPUTensorImpl : public c10::TensorImpl {
public:
  explicit NPUTensorImpl(c10::Storage&& storage, const caffe2::TypeMeta& data_type);

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
};

}  // namespace torch_npu
