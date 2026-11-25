#include <cstdint>
#include <limits>
#include <new>

#include "common/logger.h"
#include "exe_graph/fxrt_kernel_export.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"

namespace {
// The output header is already default-constructed. Rewrite its address, size,
// and placement in place. The manager stays null, so the header does not own
// the storage and a later Free leaves the source allocation alone.
ge::graphStatus RebaseDataHeader(gert::GertTensorData *header, void *address, size_t size,
                                 gert::TensorPlacement placement) {
  if (header->MutableTensorData().SetAddr(address, nullptr) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  header->SetSize(size);
  header->SetPlacement(placement);
  return ge::GRAPH_SUCCESS;
}

// Inputs: GertTensorData*, source StorageShape*, runtime element offset,
// static byte offset, element size. Outputs are a rebased data header and a
// dense logical StorageShape. This is a host operation, not a device copy.
ge::graphStatus ContiguousView(gert::KernelContext *context) {
  if (context == nullptr || (context->GetInputNum() != 5U && context->GetInputNum() != 6U) ||
      context->GetOutputNum() != 2U) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const auto *source = context->GetInputPointer<gert::GertTensorData>(0U);
  const auto *source_shape = context->GetInputPointer<gert::StorageShape>(1U);
  auto *result = context->GetOutputPointer<gert::GertTensorData>(0U);
  auto *result_shape = context->GetOutputPointer<gert::StorageShape>(1U);
  int64_t runtime_offset = context->GetInputValue<int64_t>(2U);
  const size_t static_byte_offset = context->GetInputValue<size_t>(3U);
  const size_t item_size = context->GetInputValue<size_t>(4U);
  if (context->GetInputNum() == 6U) {
    const auto *layout = context->GetInputPointer<gert::Tensor>(5U);
    if (layout == nullptr || layout->GetDataType() != ge::DT_INT64 || layout->GetPlacement() != gert::kOnHost ||
        layout->GetShapeSize() != 1 || layout->GetData<int64_t>() == nullptr) {
      return ge::GRAPH_PARAM_INVALID;
    }
    const int64_t extra_offset = *layout->GetData<int64_t>();
    if (extra_offset < 0 || runtime_offset < 0 ||
        runtime_offset > std::numeric_limits<int64_t>::max() - extra_offset) {
      return ge::GRAPH_PARAM_INVALID;
    }
    runtime_offset += extra_offset;
  }
  if (source == nullptr || source_shape == nullptr || result == nullptr || result_shape == nullptr ||
      item_size == 0U || runtime_offset < 0 ||
      static_cast<uint64_t>(runtime_offset) > std::numeric_limits<size_t>::max() / item_size) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const size_t runtime_byte_offset = static_cast<size_t>(runtime_offset) * item_size;
  if (runtime_byte_offset > std::numeric_limits<size_t>::max() - static_byte_offset) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const size_t offset = runtime_byte_offset + static_byte_offset;
  const auto &dims = source_shape->GetOriginShape();
  result_shape->MutableStorageShape().SetDimNum(dims.GetDimNum());
  result_shape->MutableOriginShape().SetDimNum(dims.GetDimNum());
  for (size_t i = 0U; i < dims.GetDimNum(); ++i) {
    const int64_t dim = dims.GetDim(i);
    if (dim < 0) {
      return ge::GRAPH_PARAM_INVALID;
    }
    result_shape->MutableStorageShape().SetDim(i, dim);
    result_shape->MutableOriginShape().SetDim(i, dim);
  }
  const auto address = reinterpret_cast<uintptr_t>(source->GetAddr());
  // The shape describes the logical view, while the header's byte span must
  // describe the remaining storage after rebasing.  A non-contiguous view can
  // legally have holes, so requiring `numel(shape)` bytes here rejects valid
  // layouts such as a column slice before AsStrided performs its own span check.
  if (offset > source->GetSize() || address > std::numeric_limits<uintptr_t>::max() - offset ||
      (address == 0U && source->GetSize() - offset != 0U)) {
    RT_GLOG(ERROR) << "FxContiguousView: view exceeds source allocation (source_bytes=" << source->GetSize()
                   << ", byte_offset=" << offset << ")";
    return ge::GRAPH_PARAM_INVALID;
  }
  return RebaseDataHeader(result, reinterpret_cast<void *>(address + offset), source->GetSize() - offset,
                          source->GetPlacement());
}

ge::graphStatus CreateContiguousView(gert::KernelContext *context) {
  if (context == nullptr || context->GetOutputNum() != 2U) return ge::GRAPH_PARAM_INVALID;
  auto *data_output = context->GetOutput(0U);
  auto *shape_output = context->GetOutput(1U);
  auto *data = new (std::nothrow) gert::GertTensorData();
  auto *shape = new (std::nothrow) gert::StorageShape();
  if (data_output == nullptr || shape_output == nullptr || data == nullptr || shape == nullptr) {
    delete data;
    delete shape;
    return ge::GRAPH_FAILED;
  }
  data_output->SetWithDefaultDeleter(data);
  shape_output->SetWithDefaultDeleter(shape);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus StorageBaseView(gert::KernelContext *context) {
  if (context == nullptr || context->GetInputNum() != 4U || context->GetOutputNum() != 2U) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const auto *source = context->GetInputPointer<gert::GertTensorData>(0U);
  const auto *source_shape = context->GetInputPointer<gert::StorageShape>(1U);
  const int64_t storage_offset = context->GetInputValue<int64_t>(2U);
  const size_t item_size = context->GetInputValue<size_t>(3U);
  auto *result = context->GetOutputPointer<gert::GertTensorData>(0U);
  auto *result_shape = context->GetOutputPointer<gert::StorageShape>(1U);
  if (source == nullptr || source_shape == nullptr || result == nullptr || result_shape == nullptr ||
      storage_offset < 0 || item_size == 0U ||
      static_cast<uint64_t>(storage_offset) > std::numeric_limits<size_t>::max() / item_size) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const size_t byte_offset = static_cast<size_t>(storage_offset) * item_size;
  const auto address = reinterpret_cast<uintptr_t>(source->GetAddr());
  if (address < byte_offset || byte_offset > std::numeric_limits<size_t>::max() - source->GetSize()) {
    return ge::GRAPH_PARAM_INVALID;
  }
  if (RebaseDataHeader(result, reinterpret_cast<void *>(address - byte_offset), source->GetSize() + byte_offset,
                       source->GetPlacement()) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  const auto &storage = source_shape->GetStorageShape();
  const auto &origin = source_shape->GetOriginShape();
  result_shape->MutableStorageShape().SetDimNum(storage.GetDimNum());
  result_shape->MutableOriginShape().SetDimNum(origin.GetDimNum());
  for (size_t i = 0U; i < storage.GetDimNum(); ++i) result_shape->MutableStorageShape().SetDim(i, storage.GetDim(i));
  for (size_t i = 0U; i < origin.GetDimNum(); ++i) result_shape->MutableOriginShape().SetDim(i, origin.GetDim(i));
  return ge::GRAPH_SUCCESS;
}

}  // namespace

FXRT_EXPORT_KERNEL_RUN(FxContiguousView, ContiguousView)
FXRT_EXPORT_KERNEL_OUTPUTS(FxContiguousView, CreateContiguousView)
FXRT_EXPORT_KERNEL_RUN(FxStorageBaseView, StorageBaseView)
FXRT_EXPORT_KERNEL_OUTPUTS(FxStorageBaseView, CreateContiguousView)
