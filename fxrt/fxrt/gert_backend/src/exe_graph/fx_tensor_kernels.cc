#include <cstddef>
#include <cstdint>
#include <new>

#include "acl/acl_rt.h"
#include "exe_graph/fxrt_kernel_export.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "graph/types.h"

namespace {
constexpr size_t kShape = 0U;
constexpr size_t kData = 1U;
constexpr size_t kDtype = 2U;
constexpr size_t kPlacement = 3U;
constexpr size_t kStorageFormat = 4U;
constexpr size_t kOriginFormat = 5U;
constexpr size_t kBuildInputs = 6U;
constexpr size_t kStream = 6U;
constexpr size_t kOutput = 7U;
constexpr size_t kAllocator = 8U;
constexpr size_t kEnsureInputs = 9U;

struct TensorScalars {
  int32_t dtype = 0;
  int32_t placement = 0;
  int32_t storage_format = 0;
  int32_t origin_format = 0;
};

ge::graphStatus AttachAddress(gert::TensorData *header, const gert::GertTensorData *source) {
  if (header == nullptr || source == nullptr) {
    return ge::GRAPH_PARAM_INVALID;
  }
  if (header->SetAddr(source->GetAddr(), nullptr) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  header->SetSize(source->GetSize());
  header->SetPlacement(source->GetPlacement());
  return ge::GRAPH_SUCCESS;
}

void ApplyMetadata(gert::Tensor *tensor, const gert::StorageShape *shape, const TensorScalars &scalars,
                   bool set_placement) {
  tensor->MutableStorageShape() = shape->GetStorageShape();
  tensor->MutableOriginShape() = shape->GetOriginShape();
  tensor->SetDataType(static_cast<ge::DataType>(scalars.dtype));
  tensor->MutableFormat().SetStorageFormat(static_cast<ge::Format>(scalars.storage_format));
  tensor->MutableFormat().SetOriginFormat(static_cast<ge::Format>(scalars.origin_format));
  if (set_placement) {
    tensor->SetPlacement(static_cast<gert::TensorPlacement>(scalars.placement));
  }
}

bool ReadScalars(const gert::KernelContext *context, TensorScalars *scalars) {
  if (context == nullptr || scalars == nullptr || context->GetInputNum() < kBuildInputs) {
    return false;
  }
  scalars->dtype = context->GetInputValue<int32_t>(kDtype);
  scalars->placement = context->GetInputValue<int32_t>(kPlacement);
  scalars->storage_format = context->GetInputValue<int32_t>(kStorageFormat);
  scalars->origin_format = context->GetInputValue<int32_t>(kOriginFormat);
  return true;
}

ge::graphStatus CreateTensor(gert::KernelContext *context) {
  if (context == nullptr || context->GetOutputNum() != 1U || context->GetOutput(0U) == nullptr) {
    return ge::GRAPH_PARAM_INVALID;
  }
  auto *tensor = new (std::nothrow) gert::Tensor();
  if (tensor == nullptr) {
    return ge::GRAPH_FAILED;
  }
  context->GetOutput(0U)->SetWithDefaultDeleter(tensor);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus RunBuild(gert::KernelContext *context) {
  TensorScalars scalars;
  if (!ReadScalars(context, &scalars)) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const auto *shape = context->GetInputPointer<gert::StorageShape>(kShape);
  auto *data = context->MutableInputPointer<gert::GertTensorData>(kData);
  auto *tensor = context->GetOutputPointer<gert::Tensor>(0U);
  if (shape == nullptr || data == nullptr || tensor == nullptr) {
    return ge::GRAPH_PARAM_INVALID;
  }
  ApplyMetadata(tensor, shape, scalars, true);
  return AttachAddress(&tensor->MutableTensorData(), data);
}

bool FindCopyKind(gert::TensorPlacement source, gert::TensorPlacement dest, aclrtMemcpyKind *kind) {
  struct Pair {
    gert::TensorPlacement source;
    gert::TensorPlacement dest;
    aclrtMemcpyKind kind;
  };
  static const Pair kTable[] = {
      {gert::kOnHost, gert::kOnDeviceHbm, ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE},
      {gert::kOnHost, gert::kOnHost, ACL_MEMCPY_HOST_TO_HOST},
      {gert::kOnHost, gert::kFollowing, ACL_MEMCPY_HOST_TO_HOST},
      {gert::kOnHost, gert::kOnDeviceP2p, ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE},
      {gert::kFollowing, gert::kOnDeviceHbm, ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE},
      {gert::kFollowing, gert::kOnHost, ACL_MEMCPY_HOST_TO_HOST},
      {gert::kFollowing, gert::kFollowing, ACL_MEMCPY_HOST_TO_HOST},
      {gert::kFollowing, gert::kOnDeviceP2p, ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE},
      {gert::kOnDeviceHbm, gert::kOnDeviceHbm, ACL_MEMCPY_DEVICE_TO_DEVICE},
      {gert::kOnDeviceHbm, gert::kOnHost, ACL_MEMCPY_DEVICE_TO_HOST},
      {gert::kOnDeviceHbm, gert::kFollowing, ACL_MEMCPY_DEVICE_TO_HOST},
      {gert::kOnDeviceHbm, gert::kOnDeviceP2p, ACL_MEMCPY_DEVICE_TO_DEVICE},
      {gert::kOnDeviceP2p, gert::kOnDeviceHbm, ACL_MEMCPY_DEVICE_TO_DEVICE},
      {gert::kOnDeviceP2p, gert::kOnHost, ACL_MEMCPY_DEVICE_TO_HOST},
      {gert::kOnDeviceP2p, gert::kFollowing, ACL_MEMCPY_DEVICE_TO_HOST},
      {gert::kOnDeviceP2p, gert::kOnDeviceP2p, ACL_MEMCPY_DEVICE_TO_DEVICE},
  };
  for (const Pair &item : kTable) {
    if (item.source == source && item.dest == dest) {
      *kind = item.kind;
      return true;
    }
  }
  return false;
}

ge::graphStatus CopyToOutput(const gert::StorageShape &shape, gert::GertTensorData &source, const TensorScalars &scalars,
                             aclrtStream stream, gert::GertAllocator *allocator, gert::Tensor &output) {
  if (output.GetAddr() == nullptr) {
    output.SetPlacement(static_cast<gert::TensorPlacement>(scalars.placement));
    gert::GertMemBlock *block = source.GetGertMemBlock();
    if (block != nullptr && allocator != nullptr && allocator->MoveL2ToL1(block) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    return AttachAddress(&output.MutableTensorData(), &source);
  }

  const int64_t copy_size = ge::GetSizeInBytes(shape.GetStorageShape().GetShapeSize(),
                                               static_cast<ge::DataType>(scalars.dtype));
  if (copy_size < 0 || static_cast<size_t>(copy_size) > output.GetTensorData().GetSize() ||
      static_cast<size_t>(copy_size) > source.GetSize()) {
    return ge::GRAPH_FAILED;
  }
  void *dest_address = output.GetAddr();
  void *source_address = source.GetAddr();
  if (source_address == dest_address || copy_size == 0) {
    return ge::GRAPH_SUCCESS;
  }
  aclrtMemcpyKind kind = ACL_MEMCPY_DEFAULT;
  if (!FindCopyKind(source.GetPlacement(), output.GetPlacement(), &kind)) {
    return ge::GRAPH_FAILED;
  }
  if (kind == ACL_MEMCPY_HOST_TO_HOST) {
    return aclrtMemcpy(dest_address, output.GetTensorData().GetSize(), source_address, static_cast<size_t>(copy_size),
                       kind) == ACL_SUCCESS
               ? ge::GRAPH_SUCCESS
               : ge::GRAPH_FAILED;
  }
  return aclrtMemcpyAsync(dest_address, output.GetTensorData().GetSize(), source_address,
                          static_cast<size_t>(copy_size), kind, stream) == ACL_SUCCESS
             ? ge::GRAPH_SUCCESS
             : ge::GRAPH_FAILED;
}

ge::graphStatus RunEnsure(gert::KernelContext *context) {
  TensorScalars scalars;
  if (!ReadScalars(context, &scalars) || context->GetInputNum() < kEnsureInputs) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const auto *shape = context->GetInputPointer<gert::StorageShape>(kShape);
  auto *data = context->MutableInputPointer<gert::GertTensorData>(kData);
  auto *output_chain = context->MutableInput(kOutput);
  if (shape == nullptr || data == nullptr || output_chain == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto *output = output_chain->GetPointer<gert::Tensor>();
  if (output == nullptr) {
    output = new (std::nothrow) gert::Tensor();
    if (output == nullptr) {
      return ge::GRAPH_FAILED;
    }
    output_chain->SetWithDefaultDeleter(output);
  }
  ApplyMetadata(output, shape, scalars, false);
  return CopyToOutput(*shape, *data, scalars,
                      context->GetInputValue<aclrtStream>(kStream),
                      context->GetInputValue<gert::GertAllocator *>(kAllocator), *output);
}
}  // namespace

FXRT_EXPORT_KERNEL_RUN(FxBuildTensor, RunBuild);
FXRT_EXPORT_KERNEL_OUTPUTS(FxBuildTensor, CreateTensor);
FXRT_EXPORT_KERNEL_RUN(FxBuildRefTensor, RunBuild);
FXRT_EXPORT_KERNEL_OUTPUTS(FxBuildRefTensor, CreateTensor);
FXRT_EXPORT_KERNEL_RUN(FxEnsureTensorAtOutMemory, RunEnsure);
