#pragma once

#include <ATen/Tensor.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/typeid.h>
#include <c10/util/order_preserving_flat_hash_map.h>

#include "third_party/acl/inc/acl/acl_rt.h"
#include "third_party/acl/inc/acl/acl_base.h"

namespace torch_npu {

struct NPUStorageDesc {
public:
    struct use_byte_size_t {};

    c10::SmallVector<int64_t, 5> base_sizes_;
    c10::SmallVector<int64_t, 5> base_strides_;
    c10::SmallVector<int64_t, 5> storage_sizes_;
    int64_t base_offset_ = 0; // no use
    use_byte_size_t base_dtype_ = {}; // no use
    aclFormat origin_format_ = ACL_FORMAT_UNDEFINED;
    aclFormat npu_format_ = ACL_FORMAT_ND;
    // used to make CANN GE tensor from storagImpl
    caffe2::TypeMeta data_type_;
};

struct NPUStorageImpl : public c10::StorageImpl {
  explicit NPUStorageImpl(use_byte_size_t use_byte_size,
      size_t size_bytes,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable);
  ~NPUStorageImpl() override = default;

  void release_resources() override;

  // not private
  NPUStorageDesc npu_desc_;

  NPUStorageDesc get_npu_desc() const {
    return npu_desc_;
  }
};

c10::intrusive_ptr<c10::StorageImpl> make_npu_storage_impl(
    c10::StorageImpl::use_byte_size_t,
    c10::SymInt size_bytes,
    c10::Allocator* allocator,
    bool resizable);

}

