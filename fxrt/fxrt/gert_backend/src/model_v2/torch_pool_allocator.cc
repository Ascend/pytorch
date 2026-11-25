// The one ABI=1 translation unit in this library.
//
// Everything else in mrt_fx_converter is compiled with _GLIBCXX_USE_CXX11_ABI=0 because
// GE's host libraries are ABI=0. torch_npu is ABI=1, and its caching allocator cannot be
// called across that boundary: NPUCachingAllocator.h pulls c10's inline helpers, whose
// failure paths take `const std::string&`, and an ABI=0 caller ends up referencing
// `std::string` symbols torch never exports ("undefined reference to
// c10::detail::torchInternalAssertFail(..., std::string const&)" at link time -- verified).
//
// So this file alone is compiled at ABI=1 and links torch_npu directly. It talks to the
// rest of the library through fx_allocator_bridge.h only, which carries PODs and raw C
// function pointers and therefore has the same layout under either ABI. No std type
// crosses: the callbacks below are extern "C" and take void*/size_t, and the descriptor
// holds nothing but pointers.
//
// Why here rather than in torch_npu: the consumer of the allocator is GE, driven by this
// library, and torch_npu is installed before it and must not depend on it. Keeping the
// callbacks here means torch_npu needs no knowledge of GE's allocator interface at all.

#include <torch/csrc/autograd/python_variable.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "api/fx_allocator_bridge.h"
#include "api/torch_input_bridge.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"

namespace {

// GE hands the block pointer back to free_fn unchanged, so `block` and the device address
// are the same value here -- raw_alloc returns the address itself.
extern "C" void* FxTorchPoolAlloc(void* /*obj*/, size_t size) {
  return c10_npu::NPUCachingAllocator::raw_alloc(size);
}

extern "C" void FxTorchPoolFree(void* /*obj*/, void* block) {
  if (block != nullptr) {
    c10_npu::NPUCachingAllocator::raw_delete(block);
  }
}

extern "C" void* FxTorchPoolGetAddr(void* block) {
  // raw_alloc hands back the address itself, so the block *is* the address.
  return block;
}

// Same base set as fxrt::ops::IsBaseFormat. These integers are the shared aclFormat /
// ge::FORMAT_ values: NCHW=0, NHWC=1, ND=2, NCDHW=30.
bool IsBaseAclFormat(int64_t format) {
  return format == 0 || format == 1 || format == 2 || format == 30;
}

int32_t WriteDims(c10::IntArrayRef sizes, int64_t* storage_dims, int32_t cap, int32_t* rank) {
  if (sizes.size() > static_cast<size_t>(cap)) {
    *rank = -1;
    return -1;
  }
  *rank = static_cast<int32_t>(sizes.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    storage_dims[i] = sizes[i];
  }
  return 0;
}

int32_t ReadFeedLayout(const at::Tensor& tensor, int64_t* dims, int32_t capacity, int32_t* rank) {
  constexpr int32_t kFormatNd = 2;
  if (!tensor.defined() || tensor.device().type() != at::DeviceType::PrivateUse1) {
    WriteDims(tensor.defined() ? tensor.sizes() : c10::IntArrayRef{}, dims, capacity, rank);
    return kFormatNd;
  }
  const int64_t format = at_npu::native::get_npu_format(tensor);
  if (IsBaseAclFormat(format)) {
    WriteDims(tensor.sizes(), dims, capacity, rank);
  } else {
    WriteDims(at_npu::native::get_npu_storage_sizes(tensor), dims, capacity, rank);
  }
  return format < 0 ? kFormatNd : static_cast<int32_t>(format);
}

void CollectPhysicalLayout(const at::Tensor& tensor, fx_converter::TorchInputDescriptor* desc) {
  int32_t rank = 0;
  int32_t format = 2;
  try {
    format = ReadFeedLayout(tensor, desc->storage_shape, static_cast<int32_t>(desc->capacity), &rank);
  } catch (const std::exception&) {
    // Preserve the legacy format query's logical-shape/ND fallback.
    rank = static_cast<int32_t>(desc->rank);
    std::copy_n(desc->shape, desc->rank, desc->storage_shape);
  }
  if (rank < 0) {
    throw std::invalid_argument("input storage rank exceeds descriptor capacity");
  }
  desc->storage_rank = static_cast<size_t>(rank);
  desc->format = format;
}

void ValidateTorchStorage(const at::Tensor& tensor, fx_converter::TorchInputDescriptor* desc) {
  if (desc->storage_offset < 0) {
    throw std::invalid_argument("negative input storage_offset");
  }
  if (!IsBaseAclFormat(desc->format)) {
    if (desc->storage_offset != 0) {
      throw std::invalid_argument("nonzero storage_offset is unsupported for private formats");
    }
    return;
  }
  const size_t item_size = tensor.element_size();
  const size_t nbytes = tensor.storage().nbytes();
  if (item_size == 0 || nbytes % item_size != 0) {
    throw std::invalid_argument("input storage size is not aligned to its element size");
  }
  const uint64_t capacity = nbytes / item_size;
  uint64_t last = static_cast<uint64_t>(desc->storage_offset);
  uint64_t logical_count = 1;
  const bool empty = std::find(desc->shape, desc->shape + desc->rank, 0) != desc->shape + desc->rank;
  for (size_t axis = 0; axis < desc->rank; ++axis) {
    const int64_t size = desc->shape[axis];
    const int64_t stride = desc->stride[axis];
    if (size < 0 || stride < 0) {
      throw std::invalid_argument("input shape and stride must be concrete and non-negative");
    }
    if (empty) {
      continue;
    }
    const uint64_t extent = static_cast<uint64_t>(size - 1);
    if (extent != 0 && static_cast<uint64_t>(stride) > (std::numeric_limits<uint64_t>::max() - last) / extent) {
      throw std::invalid_argument("input storage span overflows address range");
    }
    last += extent * static_cast<uint64_t>(stride);
    // A valid nonempty tensor's element count cannot exceed its storage span
    // unless it broadcasts. Saturate that product instead of overflowing it.
    const auto dim = static_cast<uint64_t>(size);
    logical_count = logical_count > std::numeric_limits<uint64_t>::max() / dim ? std::numeric_limits<uint64_t>::max()
                                                                               : logical_count * dim;
  }
  if (!empty && last >= capacity) {
    throw std::invalid_argument("input view span exceeds storage capacity");
  }
  if ((empty ? 0U : logical_count) != capacity) {
    if (capacity > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      throw std::invalid_argument("input storage capacity exceeds int64 range");
    }
    desc->storage_shape[0] = static_cast<int64_t>(capacity);
    desc->storage_rank = 1;
  }
}

void CollectTorchInput(PyObject* object, fx_converter::TorchInputDescriptor* desc) {
  if (!THPVariable_Check(object)) {
    throw std::invalid_argument("input must be a torch tensor");
  }
  const at::Tensor& tensor = THPVariable_Unpack(object);
  if (!tensor.defined() || tensor.layout() != at::kStrided) {
    throw std::invalid_argument("input must be a defined strided tensor");
  }
  const auto sizes = tensor.sizes();
  const auto strides = tensor.strides();
  if (sizes.size() > desc->capacity) {
    throw std::invalid_argument("input rank exceeds runtime descriptor capacity");
  }
  desc->rank = sizes.size();
  std::copy(sizes.begin(), sizes.end(), desc->shape);
  std::copy(strides.begin(), strides.end(), desc->stride);
  desc->storage_offset = tensor.storage_offset();
  CollectPhysicalLayout(tensor, desc);
  ValidateTorchStorage(tensor, desc);
  // TensorV2 offsets are relative to storage, while data_ptr is view-relative.
  desc->address = desc->storage_offset == 0 ? tensor.data_ptr() : tensor.storage().data_ptr().get();
}

} // namespace

// Outside the anonymous namespace so the ABI=0 nanobind TU can call it. One feed, one
// get_npu_format. Same read as fxrt UpdateTensorFromTorchTensor, done here because the
// rest of the converter is ABI=0 and cannot call at_npu. Storage shape follows
// StorageShapeFromTorch: a base format is fully described by the logical sizes, a private
// format uses get_npu_storage_sizes. Returns the acl/ge format (2 = ND). *rank is -1 when
// the shape does not fit or the query throws; the exception never leaves this TU.
extern "C" int32_t FxcQueryFeedLayout(PyObject* obj, int64_t* storage_dims, int32_t cap, int32_t* rank) {
  constexpr int32_t kFormatNd = 2;
  if (rank == nullptr) {
    return kFormatNd;
  }
  *rank = 0;
  if (obj == nullptr || storage_dims == nullptr || cap <= 0) {
    return kFormatNd;
  }
  try {
    return ReadFeedLayout(THPVariable_Unpack(obj), storage_dims, cap, rank);
  } catch (const std::exception&) {
    *rank = -1;
    return kFormatNd;
  } catch (...) {
    *rank = -1;
    return kFormatNd;
  }
}

namespace fx_converter {

extern "C" bool FxcCollectTensorInputs(
    void* py_sequence,
    TorchInputDescriptor* descriptors,
    size_t count,
    char* error,
    size_t error_capacity) {
  PyObject* sequence = PySequence_Fast(static_cast<PyObject*>(py_sequence), "inputs must be a tensor sequence");
  if (sequence == nullptr) {
    PyErr_Clear();
    if (error != nullptr && error_capacity != 0) {
      std::snprintf(error, error_capacity, "inputs must be a tensor sequence");
    }
    return false;
  }
  bool ok = false;
  try {
    if (static_cast<size_t>(PySequence_Fast_GET_SIZE(sequence)) != count) {
      throw std::invalid_argument("input count changed while collecting tensor descriptors");
    }
    for (size_t i = 0; i < count; ++i) {
      CollectTorchInput(PySequence_Fast_GET_ITEM(sequence, static_cast<Py_ssize_t>(i)), &descriptors[i]);
    }
    ok = true;
  } catch (const std::exception& exception) {
    if (error != nullptr && error_capacity != 0) {
      std::snprintf(error, error_capacity, "%s", exception.what());
    }
  } catch (...) {
    if (error != nullptr && error_capacity != 0) {
      std::snprintf(error, error_capacity, "failed to collect tensor inputs");
    }
  }
  Py_DECREF(sequence);
  return ok;
}

// Registers GE's device allocator with the torch_npu caching allocator. Idempotent and
// cheap after the first call; callers invoke it immediately before building a runner, so
// both RT2 and OM2 pick the descriptor up from the same place.
void FxRegisterTorchPoolAllocator() {
  static const bool registered = []() {
    FxExternalAllocatorDesc desc{};
    desc.alloc_fn = &FxTorchPoolAlloc;
    desc.free_fn = &FxTorchPoolFree;
    desc.get_addr_fn = &FxTorchPoolGetAddr;
    desc.obj = nullptr;
    FxSetExternalAllocator(&desc);
    return FxHasExternalAllocator();
  }();
  (void)registered;
}

} // namespace fx_converter
