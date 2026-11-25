#include "opp/aclnn_tensor_utils.h"
#include "opp/aclnn_api.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "common/logger.h"
#include "exe_graph/graph_builder_utils.h"

namespace fx_converter {
namespace {

struct TensorViewLayout {
  std::vector<int64_t> logical_shape;
  std::vector<int64_t> stride;
  int64_t storage_offset = 0;
};

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
  }
  return strides;
}

std::vector<int64_t> ShapeToVector(const gert::Shape &shape) {
  std::vector<int64_t> dims(shape.GetDimNum());
  for (size_t i = 0; i < dims.size(); ++i) {
    dims[i] = shape.GetDim(static_cast<int64_t>(i));
  }
  return dims;
}

std::vector<int64_t> StrideToVector(const gert::Stride &stride) {
  std::vector<int64_t> values(stride.GetDimNum());
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = stride.GetStride(i);
  }
  return values;
}

bool StorageSpan(const TensorViewLayout &layout, int64_t *span) {
  if (span == nullptr || layout.logical_shape.size() != layout.stride.size()) {
    return false;
  }
  if (layout.storage_offset < 0) {
    return false;
  }
  int64_t max_offset = layout.storage_offset;
  for (size_t i = 0; i < layout.logical_shape.size(); ++i) {
    const int64_t size = layout.logical_shape[i];
    const int64_t stride = layout.stride[i];
    if (size < 0 || stride < 0) {
      return false;
    }
    if (size == 0) {
      *span = 0;
      return true;
    }
    if (size > 1 && stride > 0 && size - 1 > (std::numeric_limits<int64_t>::max() - max_offset) / stride) {
      return false;
    }
    max_offset += (size - 1) * stride;
  }
  *span = max_offset + 1;
  return true;
}

std::string ShapeStr(const std::vector<int64_t> &s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i != 0) {
      out += ",";
    }
    out += std::to_string(s[i]);
  }
  out += "]";
  return out;
}

TensorViewLayout WithEffectiveStride(TensorViewLayout layout) {
  if (layout.stride.empty()) {
    layout.stride = ContiguousStrides(layout.logical_shape);
  }
  return layout;
}

bool GetTensorViewLayout(const gert::Tensor *tensor, TensorViewLayout *layout) {
  if (tensor == nullptr || layout == nullptr || tensor->GetVersion() != gert::kTensorV2) {
    return false;
  }
  const auto *view = reinterpret_cast<const gert::TensorV2 *>(tensor);
  layout->logical_shape = ShapeToVector(view->GetOriginShape());
  layout->stride = StrideToVector(view->GetStride());
  layout->storage_offset = view->GetOffset();
  return true;
}

bool IsBaseGeFormat(ge::Format format) {
  return format == ge::FORMAT_ND || format == ge::FORMAT_NCHW || format == ge::FORMAT_NHWC ||
         format == ge::FORMAT_NCDHW;
}

// Base layouts stay ACL_FORMAT_ND. A private format is the network input's
// storage format and has to reach aclnn; a new output tensor is allocated ND.
aclFormat AclFormatOf(const gert::Tensor *tensor) {
  if (tensor == nullptr) {
    return ACL_FORMAT_ND;
  }
  const ge::Format format = tensor->GetStorageFormat();
  if (IsBaseGeFormat(format)) {
    return ACL_FORMAT_ND;
  }
  return static_cast<aclFormat>(static_cast<int32_t>(format));
}

std::vector<int64_t> PrivateStorageDims(const gert::Tensor *tensor) {
  const gert::Shape &storage = tensor->GetStorageShape();
  std::vector<int64_t> dims(storage.GetDimNum());
  for (size_t i = 0; i < dims.size(); ++i) {
    dims[i] = storage.GetDim(static_cast<int64_t>(i));
  }
  return dims;
}

std::vector<int64_t> AclStorageDims(const gert::Tensor *tensor, int64_t nd_span) {
  if (AclFormatOf(tensor) == ACL_FORMAT_ND) {
    return {nd_span};
  }
  std::vector<int64_t> dims = PrivateStorageDims(tensor);
  if (dims.empty()) {
    return {nd_span};
  }
  return dims;
}

aclTensor *ToAclTensorFromViewLayout(const gert::Tensor *tensor, ge::DataType dtype,
                                     const TensorViewLayout &layout) {
  const TensorViewLayout effective = WithEffectiveStride(layout);
  int64_t storage_span = 0;
  if (!StorageSpan(effective, &storage_span)) {
    RT_GLOG(ERROR) << "ToAclTensor: invalid runtime layout logical=" << ShapeStr(effective.logical_shape)
                   << " stride=" << ShapeStr(effective.stride);
    return nullptr;
  }
  const std::vector<int64_t> storage_dims = AclStorageDims(tensor, storage_span);
  return aclnn_api::aclCreateTensor(effective.logical_shape.data(), effective.logical_shape.size(), ToAclDataType(dtype),
                         effective.stride.data(), effective.storage_offset, AclFormatOf(tensor), storage_dims.data(),
                         storage_dims.size(),
                         const_cast<void *>(tensor->GetAddr()));
}

aclTensor *ToAclTensorBroadcastFromLayout(const gert::Tensor *tensor, const TensorViewLayout &layout,
                                          const std::vector<int64_t> &logical_shape) {
  const TensorViewLayout effective = WithEffectiveStride(layout);
  if (effective.logical_shape.size() != logical_shape.size() ||
      effective.stride.size() != effective.logical_shape.size()) {
    RT_GLOG(ERROR) << "ToAclTensorBroadcastTo: runtime layout rank mismatch src="
                   << ShapeStr(effective.logical_shape) << " target=" << ShapeStr(logical_shape);
    return nullptr;
  }
  auto stride = effective.stride;
  bool has_broadcast = false;
  for (size_t i = 0; i < logical_shape.size(); ++i) {
    const int64_t src = effective.logical_shape[i];
    const int64_t dst = logical_shape[i];
    if (src == dst) {
      continue;
    }
    if (src == 1 && dst != 1) {
      stride[i] = 0;
      has_broadcast = true;
      continue;
    }
    RT_GLOG(ERROR) << "ToAclTensorBroadcastTo: cannot broadcast runtime layout src="
                   << ShapeStr(effective.logical_shape) << " target=" << ShapeStr(logical_shape);
    return nullptr;
  }
  if (!has_broadcast) {
    return ToAclTensorFromViewLayout(tensor, tensor->GetDataType(), effective);
  }
  TensorViewLayout expanded = effective;
  expanded.logical_shape = logical_shape;
  expanded.stride = std::move(stride);
  return ToAclTensorFromViewLayout(tensor, tensor->GetDataType(), expanded);
}

aclTensor *ToAclTensorTransposedLastTwoFromLayout(const gert::Tensor *tensor, const TensorViewLayout &layout) {
  TensorViewLayout effective = WithEffectiveStride(layout);
  if (effective.logical_shape.size() < 2U || effective.stride.size() != effective.logical_shape.size()) {
    return ToAclTensorFromViewLayout(tensor, tensor->GetDataType(), effective);
  }
  int64_t storage_span = 0;
  if (!StorageSpan(effective, &storage_span)) {
    RT_GLOG(ERROR) << "ToAclTensorTransposedLastTwo: invalid runtime layout logical="
                   << ShapeStr(effective.logical_shape) << " stride=" << ShapeStr(effective.stride);
    return nullptr;
  }
  const size_t r2 = effective.logical_shape.size() - 2U;
  const size_t r1 = effective.logical_shape.size() - 1U;
  std::swap(effective.logical_shape[r2], effective.logical_shape[r1]);
  std::swap(effective.stride[r2], effective.stride[r1]);
  const std::vector<int64_t> storage_dims{storage_span};
  return aclnn_api::aclCreateTensor(effective.logical_shape.data(), effective.logical_shape.size(),
                         ToAclDataType(tensor->GetDataType()), effective.stride.data(), effective.storage_offset,
                         ACL_FORMAT_ND, storage_dims.data(), storage_dims.size(),
                         const_cast<void *>(tensor->GetAddr()));
}

}  // namespace

std::vector<int64_t> TensorLogicalShape(const gert::Tensor *tensor) {
  if (tensor == nullptr) {
    return {};
  }
  // ExecuteCustomOp inputs may be downgraded to TensorV1 by GE while retaining
  // the original logical shape separately from the physical storage span.
  return ShapeToVector(tensor->GetOriginShape());
}

bool IsTensorLastTwoTransposed(const gert::Tensor *tensor) {
  TensorViewLayout layout;
  if (!GetTensorViewLayout(tensor, &layout) || layout.stride.empty()) {
    return false;
  }
  const auto &logical = layout.logical_shape;
  if (logical.size() < 2U || layout.stride.size() != logical.size()) {
    return false;
  }
  if (layout.stride == ContiguousStrides(logical)) {
    return false;
  }
  // Infer the contiguous parent by swapping the last two logical dims. Do not
  // use NPU-padded storage_shape: ATen strides are computed from logical sizes.
  auto physical = logical;
  const size_t r2 = physical.size() - 2U;
  const size_t r1 = physical.size() - 1U;
  std::swap(physical[r2], physical[r1]);
  auto expected = ContiguousStrides(physical);
  std::swap(expected[r2], expected[r1]);
  return layout.stride == expected;
}

aclDataType ToAclDataType(ge::DataType dt) {
  switch (dt) {
    case ge::DT_FLOAT:
      return ACL_FLOAT;
    case ge::DT_FLOAT16:
      return ACL_FLOAT16;
    case ge::DT_BF16:
      return ACL_BF16;
    case ge::DT_INT32:
      return ACL_INT32;
    case ge::DT_INT64:
      return ACL_INT64;
    case ge::DT_INT8:
      return ACL_INT8;
    case ge::DT_INT16:
      return ACL_INT16;
    case ge::DT_UINT8:
      return ACL_UINT8;
    case ge::DT_BOOL:
      return ACL_BOOL;
    case ge::DT_DOUBLE:
      return ACL_DOUBLE;
    default:
      return ACL_FLOAT;
  }
}

aclTensor *ToAclTensor(const gert::Tensor *tensor) {
  if (tensor == nullptr) {
    return nullptr;
  }
  return ToAclTensorWithDtype(tensor, tensor->GetDataType());
}

aclTensor *ToAclTensorWithDtype(const gert::Tensor *tensor, ge::DataType dtype) {
  if (tensor == nullptr) {
    return nullptr;
  }
  TensorViewLayout layout;
  if (GetTensorViewLayout(tensor, &layout)) {
    return ToAclTensorFromViewLayout(tensor, dtype, layout);
  }
  const auto shape = TensorLogicalShape(tensor);
  const auto strides = ContiguousStrides(shape);
  if (AclFormatOf(tensor) == ACL_FORMAT_ND) {
    return aclnn_api::aclCreateTensor(shape.data(), shape.size(), ToAclDataType(dtype), strides.data(), 0, ACL_FORMAT_ND,
                           shape.data(), shape.size(), const_cast<void *>(tensor->GetAddr()));
  }
  const int64_t span = std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1),
                                       [](int64_t acc, int64_t dim) { return acc * dim; });
  std::vector<int64_t> dims = PrivateStorageDims(tensor);
  if (dims.empty()) {
    dims = {span};
  }
  return aclnn_api::aclCreateTensor(shape.data(), shape.size(), ToAclDataType(dtype), strides.data(), 0, AclFormatOf(tensor),
                         dims.data(), dims.size(), const_cast<void *>(tensor->GetAddr()));
}

aclTensor *ToAclTensorTransposedLastTwo(const gert::Tensor *tensor) {
  if (tensor == nullptr) {
    return nullptr;
  }
  TensorViewLayout layout;
  if (GetTensorViewLayout(tensor, &layout)) {
    return ToAclTensorTransposedLastTwoFromLayout(tensor, layout);
  }
  const auto storage = TensorLogicalShape(tensor);
  if (storage.size() < 2U) {
    return ToAclTensor(tensor);
  }
  auto view = storage;
  auto strides = ContiguousStrides(storage);
  const size_t r2 = view.size() - 2U;
  const size_t r1 = view.size() - 1U;
  std::swap(view[r2], view[r1]);
  std::swap(strides[r2], strides[r1]);
  // aclnn ND tensors use a 1-D storage size (numel), not the physical rank-2 shape.
  // Passing [K,M] here made Mm/Bmm ignore the view stride and read a reshape.
  int64_t numel = 1;
  for (int64_t d : storage) {
    if (d < 0) {
      return ToAclTensor(tensor);
    }
    numel *= d;
  }
  const std::vector<int64_t> storage_dims{numel};
  return aclnn_api::aclCreateTensor(view.data(), view.size(), ToAclDataType(tensor->GetDataType()), strides.data(), 0,
                         ACL_FORMAT_ND, storage_dims.data(), storage_dims.size(),
                         const_cast<void *>(tensor->GetAddr()));
}

aclTensor *ToAclTensorAssumingLastTwoTransposed(const gert::Tensor *tensor) {
  if (tensor == nullptr) {
    return nullptr;
  }
  TensorViewLayout layout;
  layout.logical_shape = TensorLogicalShape(tensor);
  if (layout.logical_shape.size() < 2U) {
    return nullptr;
  }
  auto physical_shape = layout.logical_shape;
  const size_t r2 = physical_shape.size() - 2U;
  const size_t r1 = physical_shape.size() - 1U;
  std::swap(physical_shape[r2], physical_shape[r1]);
  layout.stride = ContiguousStrides(physical_shape);
  std::swap(layout.stride[r2], layout.stride[r1]);
  return ToAclTensorFromViewLayout(tensor, tensor->GetDataType(), layout);
}

aclTensor *ToAclTensorBroadcastTo(const gert::Tensor *tensor, const std::vector<int64_t> &logical_shape) {
  if (tensor == nullptr) {
    return nullptr;
  }
  TensorViewLayout layout;
  if (GetTensorViewLayout(tensor, &layout)) {
    if (layout.logical_shape == logical_shape) {
      return ToAclTensorFromViewLayout(tensor, tensor->GetDataType(), layout);
    }
    return ToAclTensorBroadcastFromLayout(tensor, layout, logical_shape);
  }
  const auto physical_shape = TensorLogicalShape(tensor);
  if (physical_shape.size() != logical_shape.size() || physical_shape == logical_shape) {
    return ToAclTensor(tensor);
  }

  auto strides = ContiguousStrides(physical_shape);
  int64_t storage_numel = 1;
  bool has_broadcast = false;
  for (size_t i = 0; i < physical_shape.size(); ++i) {
    const int64_t physical = physical_shape[i];
    const int64_t logical = logical_shape[i];
    if (physical < 0 || logical < 0 || (physical != logical && physical != 1)) {
      return ToAclTensor(tensor);
    }
    if (physical == 1 && logical != 1) {
      strides[i] = 0;
      has_broadcast = true;
    }
    if (physical != 0 && storage_numel > std::numeric_limits<int64_t>::max() / physical) {
      return ToAclTensor(tensor);
    }
    storage_numel *= physical;
  }
  if (!has_broadcast) {
    return ToAclTensor(tensor);
  }

  const std::vector<int64_t> storage_dims{storage_numel};
  return aclnn_api::aclCreateTensor(logical_shape.data(), logical_shape.size(), ToAclDataType(tensor->GetDataType()),
                         strides.data(), 0, ACL_FORMAT_ND, storage_dims.data(), storage_dims.size(),
                         const_cast<void *>(tensor->GetAddr()));
}

void DestroyAclTensor(aclTensor *t) {
  if (t != nullptr) {
    (void)aclnn_api::aclDestroyTensor(t);
  }
}

void DestroyAclScalar(aclScalar *s) {
  if (s != nullptr) {
    (void)aclnn_api::aclDestroyScalar(s);
  }
}

}  // namespace fx_converter
