#pragma once

#include <ATen/Tensor.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/typeid.h>
#include <torch_npu/csrc/framework/graph/util/NPUGraph.h>
#include <c10/util/order_preserving_flat_hash_map.h>

#include "third_party/acl/inc/acl/acl_rt.h"
#include "third_party/acl/inc/acl/acl_base.h"

namespace torch_npu {

struct NPUStorageDesc {
public:
  struct use_byte_size_t {};

  c10::SmallVector<int64_t,5> base_sizes_;
  c10::SmallVector<int64_t,5> base_strides_;
  c10::SmallVector<int64_t,5> storage_sizes_;
  int64_t base_offset_ = 0; // no use
  use_byte_size_t base_dtype_; // no use
  aclFormat origin_format_;
  aclFormat npu_format_ = ACL_FORMAT_ND;
  // used to make CANN GE tensor from storagImpl
  caffe2::TypeMeta data_type_;
};

struct NpuGraphDesc {
public:
  NpuGraphDesc() {
    static int64_t idx = 0;
    unique_id = idx++;
  }

  uint64_t unique_id = 0;
  at_npu::native::Value graph_value;
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

  std::unique_ptr<NpuGraphDesc> npu_graph_desc = nullptr;

  NPUStorageDesc get_npu_desc() const {
    return npu_desc_;
  }

  const NpuGraphDesc& get_npu_graph_desc() const {
    if (npu_graph_desc == nullptr) {
      AT_ERROR("npu graph desc has not been initialized");
    }
    return *npu_graph_desc;
  }

  NpuGraphDesc& get_mutable_npu_graph_desc() const {
    if (npu_graph_desc == nullptr) {
      AT_ERROR("npu graph desc has not been initialized");
    }
    return *npu_graph_desc;
  }
};

}