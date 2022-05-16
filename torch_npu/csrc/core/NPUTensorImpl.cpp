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

#include <c10/core/ScalarType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/core/NPUTensorImpl.h"
#include "third_party/acl/inc/acl/acl_rt.h"

namespace torch_npu
{
  NPUTensorImpl::NPUTensorImpl(c10::Storage &&storage, const caffe2::TypeMeta &data_type)
      : c10::TensorImpl(std::move(storage),
                        c10::DispatchKeySet{at_npu::key::NativeDispatchKey,
                                            at_npu::key::NativeAutogradDispatchKey},
                        data_type)
  {
    is_non_overlapping_and_dense_ = false;
  }

  void NPUTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl> &impl)
  {
    NPUTensorImpl *npu_impl = static_cast<NPUTensorImpl *>(impl.get());
    copy_tensor_metadata(
        npu_impl,
        this,
        version_counter(),
        allow_tensor_metadata_change());
    npu_impl->refresh_numel();
    npu_impl->refresh_contiguous();
  }

  c10::intrusive_ptr<c10::TensorImpl> NPUTensorImpl::shallow_copy_and_detach(
      const c10::VariableVersion &version_counter,
      bool allow_tensor_metadata_change) const
  {
    auto impl = c10::make_intrusive<NPUTensorImpl>(c10::Storage(this->storage()), this->data_type_);
    copy_tensor_metadata(
        this,
        impl.get(),
        version_counter,
        allow_tensor_metadata_change);
    impl->refresh_numel();
    impl->refresh_contiguous();
    return impl;
  }

  c10::intrusive_ptr<c10::TensorImpl> NPUTensorImpl::shallow_copy_and_detach(
      c10::VariableVersion &&version_counter,
      bool allow_tensor_metadata_change) const
  {
    auto impl = c10::make_intrusive<NPUTensorImpl>(c10::Storage(this->storage()), this->data_type_);
    copy_tensor_metadata(
        this,
        impl.get(),
        std::move(version_counter),
        allow_tensor_metadata_change);
    impl->refresh_numel();
    impl->refresh_contiguous();
    return impl;
  }

}
