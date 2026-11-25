#include <cstdint>
#include <limits>
#include <new>

#include "common/logger.h"
#include "exe_graph/fxrt_kernel_export.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/tensor.h"

namespace {
constexpr int64_t kRuntimeDenseStrideSentinel = std::numeric_limits<int64_t>::min();

ge::graphStatus BuildLayoutShape(gert::KernelContext *context) {
  if (context == nullptr || context->GetInputNum() != 2U || context->GetOutputNum() != 1U) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const auto *layout = context->GetInputPointer<gert::Tensor>(0U);
  const auto *runtime_shape = context->GetInputPointer<gert::StorageShape>(1U);
  auto *shape = context->GetOutputPointer<gert::StorageShape>(0U);
  if (layout == nullptr || runtime_shape == nullptr || shape == nullptr || layout->GetDataType() != ge::DT_INT64 ||
      layout->GetPlacement() != gert::kOnHost || layout->GetShapeSize() < 0 ||
      layout->GetShapeSize() > gert::Stride::kMaxDimNum) return ge::GRAPH_PARAM_INVALID;
  const size_t rank = static_cast<size_t>(layout->GetShapeSize());
  const auto *values = layout->GetData<int64_t>();
  if (rank != 0U && values == nullptr) return ge::GRAPH_PARAM_INVALID;
  shape->MutableStorageShape().SetDimNum(rank);
  shape->MutableOriginShape().SetDimNum(rank);
  for (size_t i = 0; i < rank; ++i) {
    if (values[i] < 0) {
      RT_GLOG(ERROR) << "FxBuildLayoutShape: negative runtime dimension " << values[i] << " at axis " << i;
      return ge::GRAPH_PARAM_INVALID;
    }
    shape->MutableStorageShape().SetDim(i, values[i]);
    shape->MutableOriginShape().SetDim(i, values[i]);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateLayoutShape(gert::KernelContext *context) {
  if (context == nullptr || context->GetOutputNum() != 1U) return ge::GRAPH_PARAM_INVALID;
  auto *shape = new (std::nothrow) gert::StorageShape();
  if (shape == nullptr) return ge::GRAPH_FAILED;
  context->GetOutput(0U)->SetWithDefaultDeleter(shape);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildStorageCapacityShape(gert::KernelContext *context) {
  if (context == nullptr || context->GetInputNum() != 2U || context->GetOutputNum() != 1U) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const auto *data = context->GetInputPointer<gert::GertTensorData>(0U);
  const size_t item_size = context->GetInputValue<size_t>(1U);
  auto *shape = context->GetOutputPointer<gert::StorageShape>(0U);
  if (data == nullptr || shape == nullptr || item_size == 0U || data->GetSize() % item_size != 0U ||
      data->GetSize() / item_size > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const auto capacity = static_cast<int64_t>(data->GetSize() / item_size);
  shape->MutableStorageShape().SetDimNum(1U);
  shape->MutableOriginShape().SetDimNum(1U);
  shape->MutableStorageShape().SetDim(0, capacity);
  shape->MutableOriginShape().SetDim(0, capacity);
  return ge::GRAPH_SUCCESS;
}

// Borrow storage for an ABI that explicitly understands TensorV2. This header
// never owns device memory; last-reader dependencies retain the allocation.
ge::graphStatus BuildStridedTensor(gert::KernelContext *context) {
  if (context == nullptr || context->GetInputNum() != 7U) {
    RT_GLOG(ERROR) << "FxBuildStridedTensor: expected seven inputs";
    return ge::GRAPH_PARAM_INVALID;
  }
  const auto *data = context->GetInputPointer<gert::GertTensorData>(0U);
  const auto dtype = static_cast<ge::DataType>(context->GetInputValue<int32_t>(1U));
  const auto placement = static_cast<gert::TensorPlacement>(context->GetInputValue<int32_t>(2U));
  const auto storage_format = static_cast<ge::Format>(context->GetInputValue<int32_t>(3U));
  const auto origin_format = static_cast<ge::Format>(context->GetInputValue<int32_t>(4U));
  const gert::StorageFormat format(origin_format, storage_format, gert::ExpandDimsType());
  const auto *layout = context->GetInputPointer<gert::Tensor>(5U);
  auto *output = context->GetOutputPointer<gert::TensorV2>(0U);
  const auto runtime_offset = context->GetInputValue<int64_t>(6U);
  if (runtime_offset < 0) {
    RT_GLOG(ERROR) << "FxBuildStridedTensor: invalid runtime offset=" << runtime_offset;
    return ge::GRAPH_PARAM_INVALID;
  }
  if (data == nullptr || layout == nullptr ||
      output == nullptr || layout->GetDataType() != ge::DT_INT64 || layout->GetPlacement() != gert::kOnHost) {
    RT_GLOG(ERROR) << "FxBuildStridedTensor: missing input or invalid layout tensor";
    return ge::GRAPH_PARAM_INVALID;
  }
  const auto *layout_values = layout->GetData<int64_t>();
  const int64_t layout_size = layout->GetShapeSize();
  if (layout_values == nullptr || layout_size < 1 || layout_size % 2 == 0) {
    RT_GLOG(ERROR) << "FxBuildStridedTensor: malformed layout tensor size=" << layout_size;
    return ge::GRAPH_PARAM_INVALID;
  }
  const size_t rank = static_cast<size_t>((layout_size - 1) / 2);
  if (rank > gert::Stride::kMaxDimNum) return ge::GRAPH_PARAM_INVALID;
  gert::StorageShape shape;
  shape.MutableStorageShape().SetDimNum(rank);
  shape.MutableOriginShape().SetDimNum(rank);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t dim = layout_values[i];
    if (dim < 0) {
      RT_GLOG(ERROR) << "FxBuildStridedTensor: negative runtime dimension " << dim << " at axis " << i;
      return ge::GRAPH_PARAM_INVALID;
    }
    shape.MutableStorageShape().SetDim(i, dim);
    shape.MutableOriginShape().SetDim(i, dim);
  }
  const size_t stride_start = rank;
  const int64_t static_offset = layout_values[2U * rank];
  if (static_offset < 0 || runtime_offset > std::numeric_limits<int64_t>::max() - static_offset) {
    RT_GLOG(ERROR) << "FxBuildStridedTensor: invalid offsets static=" << static_offset
                   << " runtime=" << runtime_offset;
    return ge::GRAPH_PARAM_INVALID;
  }
  const auto offset = static_offset + runtime_offset;
  if (offset < 0) {
    RT_GLOG(ERROR) << "FxBuildStridedTensor: invalid offset=" << offset << " stride_rank=" << rank;
    return ge::GRAPH_PARAM_INVALID;
  }
  gert::Stride effective_stride;
  effective_stride.SetDimNum(rank);
  bool derive_dense_stride = false;
  for (size_t i = 0; i < rank; ++i) {
    const auto value = layout_values[stride_start + i];
    if (value < 0 && value != kRuntimeDenseStrideSentinel) {
      RT_GLOG(ERROR) << "FxBuildStridedTensor: invalid stride " << value << " at axis " << i;
      return ge::GRAPH_PARAM_INVALID;
    }
    derive_dense_stride = derive_dense_stride || value == kRuntimeDenseStrideSentinel;
  }
  if (derive_dense_stride) {
    int64_t next = 1;
    for (size_t i = rank; i > 0U; --i) {
      const size_t dim = i - 1U;
      const int64_t size = shape.GetOriginShape().GetDim(dim);
      if (size < 0 || (size != 0 && next > std::numeric_limits<int64_t>::max() / size)) {
        RT_GLOG(ERROR) << "FxBuildStridedTensor: cannot derive dense stride at axis " << dim
                       << " size=" << size << " next=" << next;
        return ge::GRAPH_PARAM_INVALID;
      }
      effective_stride.SetStride(dim, next);
      next *= size;
    }
  } else {
    for (size_t i = 0; i < rank; ++i) {
      if (layout_values[stride_start + i] < 0) return ge::GRAPH_PARAM_INVALID;
      effective_stride.SetStride(i, layout_values[stride_start + i]);
    }
  }
  uint64_t span = static_cast<uint64_t>(offset);
  bool empty = false;
  for (size_t i = 0; i < effective_stride.GetDimNum(); ++i) {
    const auto size = shape.GetOriginShape().GetDim(i);
    const auto step = effective_stride.GetStride(i);
    if (size < 0) {
      RT_GLOG(ERROR) << "FxBuildStridedTensor: negative runtime dimension " << size << " at axis " << i;
      return ge::GRAPH_PARAM_INVALID;
    }
    if (size == 0) { empty = true; continue; }
    if (step != 0 && static_cast<uint64_t>(size - 1) >
        (std::numeric_limits<uint64_t>::max() - span) / static_cast<uint64_t>(step)) {
      RT_GLOG(ERROR) << "FxBuildStridedTensor: storage span overflows at axis " << i
                     << " offset=" << offset << " size=" << size << " stride=" << step;
      return ge::GRAPH_PARAM_INVALID;
    }
    span += static_cast<uint64_t>(size - 1) * static_cast<uint64_t>(step);
  }
  const auto item_size = ge::GetSizeByDataType(dtype);
  if (item_size <= 0 || (!empty && (span == std::numeric_limits<uint64_t>::max() ||
      span + 1U > data->GetSize() / static_cast<uint64_t>(item_size)))) {
    RT_GLOG(ERROR) << "FxBuildStridedTensor: layout exceeds storage, offset=" << offset
                   << " last_element=" << span << " storage_bytes=" << data->GetSize()
                   << " item_size=" << item_size << " empty=" << empty;
    return ge::GRAPH_PARAM_INVALID;
  }
  *output = gert::TensorV2(shape, format, placement, dtype,
                          const_cast<void *>(data->GetAddr()), nullptr, effective_stride, offset);
  // Sizes describe the logical view; capacity still describes the allocation.
  output->SetSize(data->GetSize());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateStridedTensor(gert::KernelContext *context) {
  if (context == nullptr || context->GetOutputNum() != 1U) return ge::GRAPH_PARAM_INVALID;
  auto *value = new (std::nothrow) gert::TensorV2();
  if (value == nullptr) return ge::GRAPH_FAILED;
  context->GetOutput(0U)->SetWithDefaultDeleter(value);
  return ge::GRAPH_SUCCESS;
}
}  // namespace
FXRT_EXPORT_KERNEL_RUN(FxBuildStridedTensor, BuildStridedTensor)
FXRT_EXPORT_KERNEL_OUTPUTS(FxBuildStridedTensor, CreateStridedTensor)
FXRT_EXPORT_KERNEL_RUN(FxBuildLayoutShape, BuildLayoutShape)
FXRT_EXPORT_KERNEL_OUTPUTS(FxBuildLayoutShape, CreateLayoutShape)
FXRT_EXPORT_KERNEL_RUN(FxBuildStorageCapacityShape, BuildStorageCapacityShape)
FXRT_EXPORT_KERNEL_OUTPUTS(FxBuildStorageCapacityShape, CreateLayoutShape)
