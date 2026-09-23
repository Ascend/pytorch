#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <pybind11/pybind11.h> // Bridge
#include <torch/extension.h>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <utility>

#ifdef ENABLE_TORCH_NPU
#include "acl/acl.h"
#include "torch_npu/csrc/aten/common/from_blob.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#endif

#include "common/intrusive_ptr_caster.h"
#include "common/logger.h"
#include "ir/common/intrusive_ptr.h"
#include "ir/tensor/tensor.h"
#include "ir/value/value.h"
#include "ir/graph.h"
#include "ops/utils/async.h"
#include "ops/utils/utils.h"
#include "hardware/hardware_abstract/device_context.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"
#include "hardware/hardware_abstract/device_context_manager.h"
#include "runtime/executor/kernel_capture/kernel_capture_executor.h"
#include "runtime/utils/utils.h"

namespace nb = nanobind;
namespace ir = fxrt::ir;
namespace hardware = fxrt::hardware;

PYBIND11_DECLARE_HOLDER_TYPE(T, ir::IntrusivePtr<T>, true);

namespace {
using CaptureId_t = fxrt::runtime::CaptureId_t;
using MempoolId_t = fxrt::runtime::MempoolId_t;
using KernelCaptureExecutorManager = fxrt::runtime::KernelCaptureExecutorManager;
ir::DataType FromTorchDType(const at::ScalarType& type);
hardware::Device FromTorchDevice(const at::Device& device);
void UpdateFxrtValue(const ir::ValuePtr& fxrtValue, nb::handle h);

// Per-graph cache for AclGraph input staticization.
// During capture, runtime inputs are cloned into stable tensors so that the
// captured graph always sees the same device addresses.  During replay, new
// inputs are copied into those stable tensors in-place.
struct GraphInputStaticCache {
  std::mutex mutex;
  bool indices_initialized{false};
  std::vector<size_t> tensor_input_indices; // indices of non-parameter tensor inputs
  std::vector<size_t> other_input_indices; // indices of parameter / non-tensor inputs
  std::unordered_map<std::string, std::vector<at::Tensor>> static_tensors_by_shape;
};

// Global registry of per-graph input caches, keyed by graph identity.
std::mutex g_network_input_cache_mutex;
std::unordered_map<uintptr_t, std::shared_ptr<GraphInputStaticCache>> g_network_input_static_cache;

// Retrieve the IR input node at `index` from the Python-side param_nodes list.
ir::NodePtr GetInputNode(const nb::list& param_nodes, size_t index) {
  const auto& fxrt_node = nb::cast<ir::NodePtr>(param_nodes[index]);
  CHECK_IF_NULL(fxrt_node);
  CHECK_IF_NULL(fxrt_node->output);
  return fxrt_node;
}

// Update a single input value at `index` from new_inputs into the IR graph.
void UpdateInputValue(const nb::list& param_nodes, const nb::tuple& new_inputs, size_t index) {
  auto fxrt_node = GetInputNode(param_nodes, index);
  UpdateFxrtValue(fxrt_node->output, new_inputs[index]);
}

// Update all inputs directly (non-AclGraph path).
void UpdateInputsDirectly(const nb::list& param_nodes, const nb::tuple& new_inputs) {
  for (size_t i = 0; i < param_nodes.size(); ++i) {
    UpdateInputValue(param_nodes, new_inputs, i);
  }
}

// Update only the inputs at the specified indices.
void UpdateIndexedInputs(const nb::list& param_nodes, const nb::tuple& new_inputs, const std::vector<size_t>& indices) {
  for (const auto input_idx : indices) {
    UpdateInputValue(param_nodes, new_inputs, input_idx);
  }
}

// Try to cast a Python handle to at::Tensor; returns false on failure.
bool TryCastTorchTensor(nb::handle h, at::Tensor* tensor) {
  try {
    pybind11::handle ph(h.ptr());
    *tensor = ph.cast<at::Tensor>();
    return true;
  } catch (const pybind11::cast_error&) {
    return false;
  }
}

// Parse the per-input "is parameter" flags from a Python list/tuple, or return all-false if None.
std::vector<bool> ParseInputIsParameter(const nb::object& input_is_parameter, size_t expected_size) {
  std::vector<bool> flags(expected_size, false);
  if (input_is_parameter.is_none()) {
    return flags;
  }

  if (nb::isinstance<nb::list>(input_is_parameter)) {
    auto list = nb::cast<nb::list>(input_is_parameter);
    if (list.size() != expected_size) {
      RT_GLOG(EXCEPTION) << "Expected " << expected_size << " parameter flags, but received " << list.size();
    }
    for (size_t i = 0; i < expected_size; ++i) {
      flags[i] = nb::cast<bool>(list[i]);
    }
    return flags;
  }

  if (nb::isinstance<nb::tuple>(input_is_parameter)) {
    auto tuple = nb::cast<nb::tuple>(input_is_parameter);
    if (tuple.size() != expected_size) {
      RT_GLOG(EXCEPTION) << "Expected " << expected_size << " parameter flags, but received " << tuple.size();
    }
    for (size_t i = 0; i < expected_size; ++i) {
      flags[i] = nb::cast<bool>(tuple[i]);
    }
    return flags;
  }

  RT_GLOG(EXCEPTION) << "input_is_parameter must be a list, tuple, or None";
  return flags;
}

// Parse the non-parameter tensor input indices from a Python list/tuple, or return empty if None.
std::vector<size_t> ParseNonParameterTensorIndices(const nb::object& indices_object, size_t expected_size) {
  std::vector<size_t> indices;
  if (indices_object.is_none()) {
    return indices;
  }

  auto push_index = [&](size_t idx) {
    if (idx >= expected_size) {
      RT_GLOG(EXCEPTION) << "Non-parameter tensor input index out of range: " << idx
                         << ", input size: " << expected_size;
    }
    indices.emplace_back(idx);
  };

  if (nb::isinstance<nb::list>(indices_object)) {
    auto list = nb::cast<nb::list>(indices_object);
    indices.reserve(list.size());
    for (size_t i = 0; i < list.size(); ++i) {
      push_index(nb::cast<size_t>(list[i]));
    }
    return indices;
  }

  if (nb::isinstance<nb::tuple>(indices_object)) {
    auto tuple = nb::cast<nb::tuple>(indices_object);
    indices.reserve(tuple.size());
    for (size_t i = 0; i < tuple.size(); ++i) {
      push_index(nb::cast<size_t>(tuple[i]));
    }
    return indices;
  }

  RT_GLOG(EXCEPTION) << "non_parameter_tensor_indices must be a list, tuple, or None";
  return indices;
}

// Derive a graph-identity key: use the explicitly provided key, or fall back to the first node's address.
uintptr_t ParseGraphKey(const nb::object& graph_key_object, const nb::list& param_nodes) {
  if (!graph_key_object.is_none()) {
    return nb::cast<uintptr_t>(graph_key_object);
  }
  if (param_nodes.empty()) {
    return 0;
  }
  auto first_node = nb::cast<ir::NodePtr>(param_nodes[0]);
  return reinterpret_cast<uintptr_t>(first_node.get());
}

// Build a dash-separated shape key (e.g. "2-3-4-5") from tensor inputs for shape-based cache lookup.
std::string BuildGraphInputShapeKey(
    const nb::tuple& new_inputs,
    const std::vector<size_t>& tensor_input_indices,
    size_t* valid_tensor_count) {
  std::stringstream ss;
  bool first = true;
  size_t valid_count = 0;
  for (const auto input_idx : tensor_input_indices) {
    at::Tensor runtime_tensor;
    if (!TryCastTorchTensor(new_inputs[input_idx], &runtime_tensor)) {
      continue;
    }
    ++valid_count;
    const auto& shape = runtime_tensor.sizes();
    for (const auto& dim : shape) {
      if (!first) {
        ss << "-";
      }
      ss << dim;
      first = false;
    }
  }
  if (valid_tensor_count != nullptr) {
    *valid_tensor_count = valid_count;
  }
  return ss.str();
}

// One-time initialization of tensor/other input index partitioning for a graph cache.
// Tensor inputs will be staticized (cloned + copy_); other inputs are updated directly.
void InitializeGraphInputIndices(
    GraphInputStaticCache* graph_cache,
    size_t expected_size,
    const nb::object& input_is_parameter,
    const nb::object& non_parameter_tensor_indices,
    const nb::tuple& new_inputs) {
  CHECK_IF_NULL(graph_cache);
  if (graph_cache->indices_initialized) {
    return;
  }

  // Determine tensor input indices from explicit hint or by inspecting parameter flags + types.
  auto tensor_indices = ParseNonParameterTensorIndices(non_parameter_tensor_indices, expected_size);
  if (tensor_indices.empty()) {
    auto parameter_flags = ParseInputIsParameter(input_is_parameter, expected_size);
    tensor_indices.reserve(expected_size);
    for (size_t i = 0; i < expected_size; ++i) {
      if (parameter_flags[i]) {
        continue;
      }
      at::Tensor runtime_tensor;
      if (TryCastTorchTensor(new_inputs[i], &runtime_tensor)) {
        tensor_indices.emplace_back(i);
      }
    }
  }

  graph_cache->tensor_input_indices = std::move(tensor_indices);

  // Derive the complement: non-tensor inputs that are updated directly each call.
  std::vector<uint8_t> tensor_index_mask(expected_size, 0);
  for (const auto idx : graph_cache->tensor_input_indices) {
    if (idx >= expected_size) {
      RT_GLOG(EXCEPTION) << "Tensor input index out of range: " << idx << ", input size: " << expected_size;
    }
    tensor_index_mask[idx] = 1;
  }

  graph_cache->other_input_indices.clear();
  graph_cache->other_input_indices.reserve(expected_size - graph_cache->tensor_input_indices.size());
  for (size_t i = 0; i < expected_size; ++i) {
    if (tensor_index_mask[i] == 0) {
      graph_cache->other_input_indices.emplace_back(i);
    }
  }
  graph_cache->indices_initialized = true;
}

// Check whether a cached static tensor must be re-created (metadata mismatch) instead of copy-in-place.
bool NeedRecreateStaticTensor(const at::Tensor& cached_tensor, const at::Tensor& runtime_tensor) {
  if (!cached_tensor.defined()) {
    return true;
  }
  if (cached_tensor.scalar_type() != runtime_tensor.scalar_type() ||
      cached_tensor.device() != runtime_tensor.device()) {
    return true;
  }
  if (cached_tensor.sizes().vec() != runtime_tensor.sizes().vec()) {
    return true;
  }
  if (cached_tensor.strides().vec() != runtime_tensor.strides().vec()) {
    return true;
  }
  return false;
}

// Remove the static cache entry for a given graph key.
void ClearGraphInputStaticCache(uintptr_t graph_key) {
  std::lock_guard<std::mutex> lock(g_network_input_cache_mutex);
  (void)g_network_input_static_cache.erase(graph_key);
}

// Get or create the per-graph input cache.
std::shared_ptr<GraphInputStaticCache> GetOrCreateGraphCache(uintptr_t graph_key) {
  std::lock_guard<std::mutex> lock(g_network_input_cache_mutex);
  auto& graph_cache_slot = g_network_input_static_cache[graph_key];
  if (graph_cache_slot == nullptr) {
    graph_cache_slot = std::make_shared<GraphInputStaticCache>();
  }
  return graph_cache_slot;
}

// Update an IR tensor's metadata and data pointer from a PyTorch tensor.
void UpdateTensorFromTorchTensor(ir::Tensor* tensor, const at::Tensor& at_tensor) {
  ir::DataType type = FromTorchDType(at_tensor.scalar_type());
  std::vector<int64_t> shape(at_tensor.sizes().begin(), at_tensor.sizes().end());
  void* data = at_tensor.data_ptr();

  auto device = tensor->GetDevice();
  if (device != FromTorchDevice(at_tensor.device())) {
    RT_GLOG(EXCEPTION) << "Device mismatch in update_tensor";
  }

#ifdef ENABLE_TORCH_NPU
  if (device.type == hardware::DeviceType::NPU) {
    auto npuFormat = at_npu::native::get_npu_format(at_tensor);
    tensor->SetFormat(static_cast<ir::MemoryFormat>(npuFormat));
    tensor->SetStrides(at_tensor.strides().vec());
    // data_ptr() returns the offset-adjusted pointer, so set storage_offset to 0
    // to avoid double-counting the offset in Tensor::DataPtr()
    tensor->SetStorageOffset(0);
    tensor->SetStorageShape(at_npu::native::get_npu_storage_sizes(at_tensor));
    RT_GLOG(INFO) << "Update tensor, format=" << ir::FormatEnumToStr(tensor->Format())
                  << ", strides=" << tensor->Strides() << ", storageOffset=" << tensor->StorageOffset()
                  << ", storageShape=" << tensor->StorageShape() << ", isView=" << at_tensor.is_view()
                  << " at.tensor.shape: " << at_tensor.sizes();
  }
#endif

  tensor->SetOwnsStorage(false);
  tensor->SetDtype(type);
  tensor->SetShape(std::move(shape));
  tensor->Resize();
  // PyTorch may return nullptr from data_ptr() for 0-element tensors; that is valid.
  if (at_tensor.numel() > 0) {
    CHECK_IF_NULL(data);
  }
  tensor->UpdateData(data);
  // Track only the remaining accessible bytes from data_ptr() to the end of the underlying storage to keep view
  // bounds valid.
  const auto storageBytes = at_tensor.storage().nbytes();
  if (at_tensor.numel() > 0) {
    const auto offsetBytes = static_cast<size_t>(at_tensor.storage_offset()) * at_tensor.element_size();
    if (offsetBytes >= storageBytes) {
      RT_GLOG(EXCEPTION) << "storage_offset reaches or exceeds storage size: offsetBytes=" << offsetBytes
                         << ", storageBytes=" << storageBytes << ", shape=" << tensor->Shape()
                         << ", dtype=" << tensor->Dtype().ToString() << ", storageOffset=" << tensor->StorageOffset()
                         << ", storageShape=" << tensor->StorageShape();
    }
    tensor->GetStorage()->Resize(storageBytes - offsetBytes);
  } else {
    // Empty tensors may legally carry a storage_offset beyond the underlying storage. Since data_ptr() is never
    // dereferenced for zero elements, keep the original storage size and let Tensor::DataPtr() expose nullptr.
    tensor->GetStorage()->Resize(storageBytes);
  }
}

// Install a lazy updater that reads the current value of a Python-side tensor handle on each invocation.
void UpdateTensorWithHandle(const ir::TensorPtr& self, nb::handle h) {
  self->SetUpdater([h](ir::Tensor* tensor) {
    pybind11::handle ph(h.ptr());
    at::Tensor at_tensor = ph.cast<at::Tensor>();
    UpdateTensorFromTorchTensor(tensor, at_tensor);
  });
}

// Install a lazy updater that reads from a stable (cached) at::Tensor pointer.
void UpdateTensorWithTorchTensor(const ir::TensorPtr& self, const at::Tensor* at_tensor) {
  self->SetUpdater([at_tensor](ir::Tensor* tensor) {
    CHECK_IF_NULL(at_tensor);
    UpdateTensorFromTorchTensor(tensor, *at_tensor);
  });
}

// Look up or create a cached tensor vector for the given shape key.
// During capture, a new entry is always created; during replay, only existing entries are returned.
std::vector<at::Tensor>* GetCachedTensorInputs(
    std::unordered_map<std::string, std::vector<at::Tensor>>* shape_cache,
    const std::string& shape_key,
    size_t tensor_input_size,
    bool in_capture) {
  CHECK_IF_NULL(shape_cache);
  if (in_capture) {
    auto& capture_cached_inputs = (*shape_cache)[shape_key];
    if (capture_cached_inputs.size() < tensor_input_size) {
      capture_cached_inputs.resize(tensor_input_size);
    }
    return &capture_cached_inputs;
  }

  auto cache_it = shape_cache->find(shape_key);
  if (cache_it == shape_cache->end()) {
    return nullptr;
  }
  return &(cache_it->second);
}

// Update a single tensor input for AclGraph capture/replay.
// In capture: clone or copy_ the runtime tensor into the cached slot.
// In replay:  copy_ the runtime tensor into the already-allocated cached slot.
void UpdateTensorInputForCaptureOrReplay(
    const nb::list& param_nodes,
    const nb::tuple& new_inputs,
    size_t input_idx,
    at::Tensor* cached_tensor,
    bool in_capture) {
  auto fxrt_node = GetInputNode(param_nodes, input_idx);
  CHECK_IF_NULL(cached_tensor);

  at::Tensor runtime_tensor;
  if (!TryCastTorchTensor(new_inputs[input_idx], &runtime_tensor)) {
    // Non-tensor input (e.g. scalar): update directly.
    UpdateFxrtValue(fxrt_node->output, new_inputs[input_idx]);
    return;
  }

  if (in_capture) {
    if (!cached_tensor->defined() || NeedRecreateStaticTensor(*cached_tensor, runtime_tensor)) {
      *cached_tensor = runtime_tensor.clone();
    } else {
      cached_tensor->copy_(runtime_tensor, /*non_blocking=*/true);
    }
  } else {
    if (!cached_tensor->defined()) {
      RT_GLOG(EXCEPTION) << "The cache tensor must be valid in replay phase.";
    }
    cached_tensor->copy_(runtime_tensor, /*non_blocking=*/true);
  }

  if (!fxrt_node->output->IsTensor()) {
    RT_GLOG(EXCEPTION) << "Only support to staticize tensor input for copy, but got: " << fxrt_node->output;
  }
  // Bind the IR tensor to the stable cached tensor so the captured graph sees a fixed address.
  UpdateTensorWithTorchTensor(fxrt_node->output->ToTensor(), cached_tensor);
}

// Update all tensor inputs for AclGraph capture/replay.
void UpdateTensorInputsForCaptureOrReplay(
    const nb::list& param_nodes,
    const nb::tuple& new_inputs,
    const std::vector<size_t>& tensor_input_indices,
    std::vector<at::Tensor>* cached_inputs,
    bool in_capture) {
  CHECK_IF_NULL(cached_inputs);
  for (size_t pos = 0; pos < tensor_input_indices.size(); ++pos) {
    const auto input_idx = tensor_input_indices[pos];
    if (pos >= cached_inputs->size()) {
      // No cached slot available; fall back to direct update.
      UpdateInputValue(param_nodes, new_inputs, input_idx);
      continue;
    }
    UpdateTensorInputForCaptureOrReplay(param_nodes, new_inputs, input_idx, &((*cached_inputs)[pos]), in_capture);
  }
}

// DataType conversion utilities

static const std::map<at::ScalarType, ir::DataType> kAtScalarTypeToDataTypeMap = {
    {at::kHalf, ir::DataType::Type::Float16},
    {at::kBFloat16, ir::DataType::Type::BFloat16},
    {at::kFloat, ir::DataType::Type::Float32},
    {at::kDouble, ir::DataType::Type::Float64},
    {at::kComplexFloat, ir::DataType::Type::Complex64},
    {at::kChar, ir::DataType::Type::Int8},
    {at::kShort, ir::DataType::Type::Int16},
    {at::kInt, ir::DataType::Type::Int32},
    {at::kLong, ir::DataType::Type::Int64},
    {at::kByte, ir::DataType::Type::UInt8},
    {at::kBool, ir::DataType::Type::Bool},
};

static const std::map<ir::DataType, at::ScalarType> kDataTypeToAtScalarTypeMap = {
    {ir::DataType::Type::Float16, at::kHalf},
    {ir::DataType::Type::BFloat16, at::kBFloat16},
    {ir::DataType::Type::Float32, at::kFloat},
    {ir::DataType::Type::Float64, at::kDouble},
    {ir::DataType::Type::Complex64, at::kComplexFloat},
    {ir::DataType::Type::Int8, at::kChar},
    {ir::DataType::Type::Int16, at::kShort},
    {ir::DataType::Type::Int32, at::kInt},
    {ir::DataType::Type::Int64, at::kLong},
    {ir::DataType::Type::UInt8, at::kByte},
    {ir::DataType::Type::Bool, at::kBool},
};

#ifdef ENABLE_TORCH_NPU
inline aclFormat ConvertMemoryFormatToAclFormat(ir::MemoryFormat format) {
  static const std::map<ir::MemoryFormat, aclFormat> kMemoryFormatToAclFormatMap = {
      {ir::MemoryFormat::FORMAT_UNDEFINED, ACL_FORMAT_UNDEFINED},
      {ir::MemoryFormat::FORMAT_NCHW, ACL_FORMAT_NCHW},
      {ir::MemoryFormat::FORMAT_NHWC, ACL_FORMAT_NHWC},
      {ir::MemoryFormat::FORMAT_ND, ACL_FORMAT_ND},
      {ir::MemoryFormat::FORMAT_NC1HWC0, ACL_FORMAT_NC1HWC0},
      {ir::MemoryFormat::FORMAT_FRACTAL_Z, ACL_FORMAT_FRACTAL_Z},
      {ir::MemoryFormat::FORMAT_NC1HWC0_C04, ACL_FORMAT_NC1HWC0_C04},
      {ir::MemoryFormat::FORMAT_HWCN, ACL_FORMAT_HWCN},
      {ir::MemoryFormat::FORMAT_NDHWC, ACL_FORMAT_NDHWC},
      {ir::MemoryFormat::FORMAT_FRACTAL_NZ, ACL_FORMAT_FRACTAL_NZ},
      {ir::MemoryFormat::FORMAT_NCDHW, ACL_FORMAT_NCDHW},
      {ir::MemoryFormat::FORMAT_NDC1HWC0, ACL_FORMAT_NDC1HWC0},
      {ir::MemoryFormat::FORMAT_FRACTAL_Z_3D, ACL_FRACTAL_Z_3D},
      {ir::MemoryFormat::FORMAT_NC, ACL_FORMAT_NC},
      {ir::MemoryFormat::FORMAT_NCL, ACL_FORMAT_NCL},
  };

  auto iter = kMemoryFormatToAclFormatMap.find(format);
  if (iter == kMemoryFormatToAclFormatMap.end()) {
    RT_GLOG(EXCEPTION) << "Unsupported MemoryFormat " << format << " for conversion to aclFormat";
    return ACL_FORMAT_UNDEFINED;
  }

  return iter->second;
}
#endif

ir::DataType FromTorchDType(const at::ScalarType& type) {
  auto iter = kAtScalarTypeToDataTypeMap.find(type);
  if (iter == kAtScalarTypeToDataTypeMap.end()) {
    RT_GLOG(EXCEPTION) << "Unsupported at::ScalarType" << type << "for conversion to ir::DataType";
    return ir::DataType::Unknown;
  }

  return iter->second;
}

at::ScalarType ToTorchDType(ir::DataType type) {
  auto iter = kDataTypeToAtScalarTypeMap.find(type);
  if (iter == kDataTypeToAtScalarTypeMap.end()) {
    RT_GLOG(EXCEPTION) << "Unsupported ir::DataType " << type << " for conversion to at::ScalarType";
    return at::kFloat;
  }

  return iter->second;
}

// Device conversion utilities
hardware::Device FromTorchDevice(const at::Device& device) {
  hardware::DeviceType deviceType;
  switch (device.type()) {
    case at::DeviceType::CPU:
      deviceType = hardware::DeviceType::CPU;
      break;
    case at::DeviceType::PrivateUse1:
      deviceType = hardware::DeviceType::NPU;
      break;
    default:
      RT_GLOG(EXCEPTION) << "Unsupported torch::Device " << device.str() << " for conversion to hardware::Device";
  }
  return hardware::Device(deviceType, device.index());
}

at::Device ToTorchDevice(const hardware::Device device) {
  at::DeviceType deviceType;
  switch (device.type) {
    case hardware::DeviceType::CPU:
      deviceType = at::kCPU;
      break;
    case hardware::DeviceType::NPU:
      deviceType = at::kPrivateUse1;
      break;
    default:
      RT_GLOG(EXCEPTION) << "Unsupported hardware::DeviceType " << hardware::GetDeviceNameByType(device.type)
                         << " for conversion to torch::Device";
  }
  return at::Device(deviceType, device.index);
}

// Create a new fxrt Tensor with a weak ref to torch Tensor data
ir::TensorPtr FromTorchTensor(const at::Tensor& tensor, bool isFake = false) {
  ir::DataType type = FromTorchDType(tensor.scalar_type());
  std::vector<int64_t> shape;
  shape.reserve(tensor.dim());
  for (auto& dim : tensor.sym_sizes()) {
    if (dim.is_symbolic()) {
      (void)shape.emplace_back(-1);
    } else if (dim.maybe_as_int().has_value()) {
      (void)shape.emplace_back(dim.maybe_as_int().value());
    } else {
      RT_GLOG(EXCEPTION) << "Dynamic shape with non-int dimension is not supported";
    }
  }

  auto device = FromTorchDevice(tensor.device());
  if (isFake) {
    return ir::MakeIntrusive<ir::Tensor>(shape, type, device);
  } else {
    return ir::MakeIntrusive<ir::Tensor>(tensor.data_ptr(), shape, type, device);
  }
}

ir::StoragePtr CopyStorage(const ir::StoragePtr& srcStorage) {
  // Need wait all ops launch task finish before launch async copy task for graph output to stream.
  fxrt::WaitLaunchTaskFinish();

  RT_GLOG(INFO) << "Begin copy storage: " << srcStorage.get();
  auto device = srcStorage->GetDevice();
  auto storage = ir::MakeIntrusive<ir::Storage>(srcStorage->SizeBytes(), device);
  CHECK_IF_NULL(storage);
  if (srcStorage->SizeBytes() == 0) {
    RT_GLOG(INFO) << "Skip copy for empty storage";
    return storage;
  }
  storage->AllocateMemory();

  auto deviceId = fxrt::collective::CollectiveManager::Instance().local_rank_id();
  fxrt::device::DeviceContextKey deviceContextKey = {hardware::GetDeviceNameByType(device.type), deviceId};
  auto deviceContext = fxrt::device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(deviceContextKey);
  CHECK_IF_NULL(deviceContext);
  CHECK_IF_NULL(deviceContext->deviceResManager_);
  if (device.type == fxrt::hardware::DeviceType::CPU) {
    // CPU does not support async copy
    if (!deviceContext->deviceResManager_->SyncCopy(
            storage->Data(), srcStorage->Data(), srcStorage->SizeBytes(), fxrt::device::CopyType::D2D)) {
      RT_GLOG(EXCEPTION) << "Async copy for output storage failed";
    }
  } else {
    if (!deviceContext->deviceResManager_->AsyncCopy(
            storage->Data(),
            srcStorage->Data(),
            srcStorage->SizeBytes(),
            fxrt::device::CopyType::D2D,
            deviceContext->deviceResManager_->GetCurrentStream())) {
      RT_GLOG(EXCEPTION) << "Async copy for output storage failed";
    }
  }
  RT_GLOG(INFO) << "End copy storage: " << srcStorage.get();
  return storage;
}

// Create a new torch Tensor by moving ownership of data from fxrt Tensor
at::Tensor ToTorchTensor(const ir::TensorPtr& tensor) {
  CHECK_IF_NULL(tensor);
  if (!fxrt::ops::IsTensorBaseFormat(tensor)) {
    RT_GLOG(EXCEPTION) << "Network output does not support non-base memory format: "
                       << ir::FormatEnumToStr(tensor->Format());
  }
  // For input is used as output directly, should update tensor
  tensor->Update();
  auto storage = tensor->GetStorage();
  if (!storage->CheckOwnsData()) {
    // Parameter or tensor which references a parameter is graph output.
    storage = CopyStorage(storage);
  } else if (
      KernelCaptureExecutorManager::GetInstance().InCapture() ||
      KernelCaptureExecutorManager::GetInstance().InReplay()) {
    // Note: can refine by copy new output value of return node.
    auto newStorage = CopyStorage(storage);
    // Note: can not free output memory here after end alloc func(end capture.) or in peplay phase.
    // storage->FreeMemory();
    storage = newStorage;
  }

  auto atDevice = ToTorchDevice(tensor->GetDevice());
  auto options = at::TensorOptions().dtype(ToTorchDType(tensor->Dtype())).device(atDevice);
  if (tensor->Numel() == 0) {
    at::Tensor out;
    if (tensor->Strides().empty()) {
      out = at::empty(tensor->Shape(), options);
    } else {
      out = at::empty_strided(tensor->Shape(), tensor->Strides(), options);
    }
    if (tensor->StorageOffset() == 0) {
      return out;
    }
    return out.as_strided(tensor->Shape(), out.strides(), tensor->StorageOffset());
  }

  void* dataPtr = storage->Data();
  CHECK_IF_NULL(dataPtr);
  auto deleterFn = storage->GetDeleter();
  std::function<void(void*)> deleter;
  if (deleterFn == nullptr) {
    auto allocator = storage->GetAllocator();
    deleter = [allocator, dataPtr](void*) {
      if (dataPtr != nullptr) {
        allocator.Free(dataPtr);
      }
    };
  } else {
    deleter = [deleterFn, dataPtr](void*) {
      if (dataPtr != nullptr) {
        deleterFn(dataPtr);
      }
    };
  }
  storage->Release();

  switch (atDevice.type()) {
    case at::DeviceType::CPU: {
      if (tensor->Strides().empty()) {
        return at::from_blob(
            static_cast<char*>(dataPtr) + tensor->StorageOffset() * (tensor->Dtype().GetSize()),
            tensor->Shape(),
            std::move(deleter),
            options);
      }
      return at::from_blob(
          static_cast<char*>(dataPtr) + tensor->StorageOffset() * (tensor->Dtype().GetSize()),
          tensor->Shape(),
          tensor->Strides(),
          std::move(deleter),
          options);
    }
#ifdef ENABLE_TORCH_NPU
    case at::DeviceType::PrivateUse1: {
      at::Tensor out;
      if (tensor->Strides().empty()) {
        out = at_npu::native::from_blob(
            static_cast<char*>(dataPtr) + tensor->StorageOffset() * (tensor->Dtype().GetSize()),
            tensor->Shape(),
            std::move(deleter),
            options);
      } else {
        out = at_npu::native::from_blob(
            static_cast<char*>(dataPtr) + tensor->StorageOffset() * (tensor->Dtype().GetSize()),
            tensor->Shape(),
            tensor->Strides(),
            0,
            std::move(deleter),
            options);
      }
      auto& desc = static_cast<torch_npu::NPUStorageImpl*>(out.storage().unsafeGetStorageImpl())->npu_desc_;
      desc.npu_format_ = ConvertMemoryFormatToAclFormat(tensor->Format());
      return out;
    }
#endif
    default:
      RT_GLOG(EXCEPTION) << "Unsupported DeviceType " << atDevice.str();
  }
}

void UpdateTensor(const ir::TensorPtr& self, nb::handle h) {
  UpdateTensorWithHandle(self, h);
}

void UpdateFxrtValue(const ir::ValuePtr& fxrtValue, nb::handle h) {
  CHECK_IF_NULL(fxrtValue);

  switch (fxrtValue->GetTag()) {
    case ir::Value::Tag::Tensor: {
      UpdateTensor(fxrtValue->ToTensor(), h);
      return;
    }
    case ir::Value::Tag::Symbol: {
      const int64_t value = nb::cast<int64_t>(h);
      auto symbolicExpr = fxrtValue->ToSymbol();
      if (symbolicExpr->GetKind() == ir::SymbolicExpr::Kind::Variable) {
        auto symbolicVar = static_cast<ir::SymbolicVar*>(symbolicExpr.get());
        symbolicVar->SetValue(value);
      }
      return;
    }
    case ir::Value::Tag::Tuple: {
      auto fxrtTuple = fxrtValue->ToTuple();
      auto pyTuple = nb::cast<nb::tuple>(h);
      if (fxrtTuple->Size() != pyTuple.size()) {
        RT_GLOG(EXCEPTION) << "Expected " << fxrtTuple->Size() << " items in tuple, but received " << pyTuple.size();
      }
      auto it = fxrtTuple->begin();
      for (size_t i = 0; i < fxrtTuple->Size(); ++i, ++it) {
        UpdateFxrtValue(*it, pyTuple[i]);
      }
      return;
    }
    case ir::Value::Tag::Int: {
      *fxrtValue = ir::Value(nb::cast<int64_t>(h));
      return;
    }
    case ir::Value::Tag::Double: {
      *fxrtValue = ir::Value(nb::cast<double>(h));
      return;
    }
    case ir::Value::Tag::Bool: {
      *fxrtValue = ir::Value(nb::cast<bool>(h));
      return;
    }
    case ir::Value::Tag::String: {
      *fxrtValue = ir::Value(nb::cast<std::string>(h));
      return;
    }
    case ir::Value::Tag::None: {
      return;
    }
    default:
      RT_GLOG(EXCEPTION) << "Unsupported Value Tag";
      return;
  }
}

// Entry point called from Python to update all runtime inputs.
// When AclGraph is enabled, tensor inputs are staticized (cloned + copy_) so that
// the captured graph always reads from stable device addresses.  Otherwise, inputs
// are updated directly.
void BatchUpdateRuntimeInputs(
    const nb::list& paramNodes,
    const nb::tuple& newInputs,
    const nb::object& inputIsParameter = nb::none(),
    const nb::object& graphKey = nb::none(),
    const nb::object& nonParameterTensorIndices = nb::none()) {
  if (paramNodes.size() != newInputs.size()) {
    RT_GLOG(EXCEPTION) << "Expected " << paramNodes.size() << " inputs, but received " << newInputs.size();
  }

  // Fast path: without AclGraph, simply update all inputs directly.
  if (!fxrt::runtime::IsAclGraphEnabled()) {
    UpdateInputsDirectly(paramNodes, newInputs);
    return;
  }

  const auto graph_key = ParseGraphKey(graphKey, paramNodes);
  auto graph_cache = GetOrCreateGraphCache(graph_key);
  CHECK_IF_NULL(graph_cache);

  std::string shape_key;
  size_t valid_tensor_count = 0;
  {
    std::lock_guard<std::mutex> cache_lock(graph_cache->mutex);
    InitializeGraphInputIndices(
        graph_cache.get(), paramNodes.size(), inputIsParameter, nonParameterTensorIndices, newInputs);

    shape_key = BuildGraphInputShapeKey(newInputs, graph_cache->tensor_input_indices, &valid_tensor_count);
    KernelCaptureExecutorManager::GetInstance().SetShapeKey(shape_key);

    bool in_capture = KernelCaptureExecutorManager::GetInstance().InCapture();
    if (valid_tensor_count > 0) {
      auto* cached_inputs = GetCachedTensorInputs(
          &(graph_cache->static_tensors_by_shape), shape_key, graph_cache->tensor_input_indices.size(), in_capture);
      if (cached_inputs != nullptr) {
        UpdateTensorInputsForCaptureOrReplay(
            paramNodes, newInputs, graph_cache->tensor_input_indices, cached_inputs, in_capture);
      } else {
        // No cached entry for this shape (replay with unseen shape); fall back to direct update.
        UpdateIndexedInputs(paramNodes, newInputs, graph_cache->tensor_input_indices);
      }
    }

    // Non-tensor / parameter inputs are always updated directly.
    UpdateIndexedInputs(paramNodes, newInputs, graph_cache->other_input_indices);
  }
}

void SetAclgraphConf() {
#ifdef ENABLE_TORCH_NPU
  auto beginAllocFunc = [](MempoolId_t pool) {
    c10_npu::NPUCachingAllocator::beginAllocateToPool(
        c10_npu::current_device(), pool, [](aclrtStream stream) { return true; });
  };
  KernelCaptureExecutorManager::GetInstance().SetCaptureBeginFunc(beginAllocFunc);
  auto endAllocFunc = [](MempoolId_t pool_id) {
    auto stream = c10_npu::getCurrentNPUStream();
    c10_npu::NPUCachingAllocator::endAllocateToPool(c10_npu::current_device(), pool_id);
  };
  KernelCaptureExecutorManager::GetInstance().SetCaptureEndFunc(endAllocFunc);
#endif
}

void SetDeviceContext() {
#ifdef ENABLE_TORCH_NPU
  fxrt::device::DeviceContextKey deviceContextKey{
      "Ascend", fxrt::collective::CollectiveManager::Instance().local_rank_id()};
  auto deviceContext = fxrt::device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(deviceContextKey);
  CHECK_IF_NULL(deviceContext);
  CHECK_IF_NULL(deviceContext->deviceResManager_);

  auto currentNPUStream = c10_npu::getCurrentNPUStream();
  auto bindStreamFunc = [currentNPUStream]() { c10_npu::setCurrentNPUStream(currentNPUStream); };
  deviceContext->deviceResManager_->SetBindStreamFunc(bindStreamFunc);

  if (fxrt::ops::IsEnablePipeline()) {
    fxrt::ops::OpAsync::SetLaunchOpFunc(at_npu::native::OpCommand::RunOpApiV2);
    fxrt::ops::OpAsync::SetWaitLaunchFinishFunc([]() { (void)c10_npu::getCurrentNPUStream().stream(true); });
  }

  auto currentStream = currentNPUStream.stream(false);
  CHECK_IF_NULL(currentStream);
  deviceContext->deviceResManager_->SetCurrentStream(currentStream);
  auto ascend_allocator = [](size_t size) -> void* {
    void* cur_alloc = c10_npu::NPUCachingAllocator::raw_alloc(size);
    RT_GLOG(INFO) << "Memory allocated via PyTorch, new addr: " << cur_alloc;
    return cur_alloc;
  };
  deviceContext->deviceResManager_->SetAllocator(ascend_allocator);
  auto ascend_deleter = [](void* dataPtr) {
    if (dataPtr != nullptr) {
      c10_npu::NPUCachingAllocator::raw_delete(dataPtr);
    }
  };
  deviceContext->deviceResManager_->SetDeleter(ascend_deleter);
  SetAclgraphConf();
#endif
}

// Wrappers for nanobind
ir::TensorPtr FromTorchTensorWrapper(nb::handle h, bool isFake) {
  pybind11::handle ph(h.ptr());
  at::Tensor t = ph.cast<at::Tensor>();
  return FromTorchTensor(t, isFake);
}

nb::object ToTorchTensorWrapper(const ir::TensorPtr& tensor) {
  at::Tensor t = ToTorchTensor(tensor);
  pybind11::object po = pybind11::cast(t);
  return nb::steal(po.release().ptr());
}

} // namespace

NB_MODULE(_fxrt_torch, m) {
  m.doc() = "PyTorch extension for FXRT";
  m.def("from_torch", &FromTorchTensorWrapper, nb::arg("tensor"), nb::arg("is_fake") = false);
  m.def("to_torch", &ToTorchTensorWrapper, nb::rv_policy::reference);
  m.def("set_device_context", &SetDeviceContext);
  m.def("clear_graph_input_static_cache", &ClearGraphInputStaticCache, nb::arg("graph_key"));
  m.def(
      "batch_update_runtime_inputs",
      &BatchUpdateRuntimeInputs,
      nb::arg("param_nodes"),
      nb::arg("new_inputs"),
      nb::arg("input_is_parameter") = nb::none(),
      nb::arg("graph_key") = nb::none(),
      nb::arg("non_parameter_tensor_indices") = nb::none(),
      "Batch update runtime inputs for nodes");
}
