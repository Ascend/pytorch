#include "api/fx_converter_api.h"
#include "api/fx_allocator_bridge.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "acl/acl_rt.h"
#include "common/logger.h"
#include "exe_graph/execute_graph_builder.h"
#include "exe_graph/execute_graph_dumper.h"
#include "exe_graph/graph_builder_utils.h"
#include "exe_graph/ge_rt2_kernels.h"
#include "exe_graph/runtime/tensor.h"  // CANN
#include "graph/tensor.h"
#include "ge/ge_api_v2.h"
#include "lowering/common/wiring.h"
#include "model_v2/fx_external_allocator_internal.h"
#include "lowering/torch_fallback/torch_fallback_runtime.h"
#include "opp/aclnn_tensor_utils.h"
#include "opp/opp_registry.h"
#include "shape/shape_list_refresh.h"

namespace fx_converter {

namespace {

using InputDeviceTensor = std::variant<gert::Tensor, gert::TensorV2>;

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
  }
  return strides;
}

gert::StorageShape MakeStorageShape(const InputTensorLayout &layout) {
  const auto &logical = layout.logical_shape;
  const auto &storage = layout.storage_shape.empty() ? logical : layout.storage_shape;
  gert::StorageShape shape;
  shape.MutableStorageShape().SetDimNum(storage.size());
  shape.MutableOriginShape().SetDimNum(logical.size());
  for (size_t i = 0; i < storage.size(); ++i) {
    shape.MutableStorageShape().SetDim(i, storage[i]);
  }
  for (size_t i = 0; i < logical.size(); ++i) {
    shape.MutableOriginShape().SetDim(i, logical[i]);
  }
  return shape;
}

gert::StorageFormat MakeStorageFormat(int32_t format_code) {
  const auto storage = format_code < 0 ? ge::FORMAT_ND : static_cast<ge::Format>(format_code);
  gert::StorageFormat format;
  format.SetStorageFormat(storage);
  format.SetOriginFormat(OriginFormatOf(storage));
  return format;
}

gert::StorageFormat MakeNdStorageFormat() { return MakeStorageFormat(static_cast<int32_t>(ge::FORMAT_ND)); }

bool IsBaseStorageFormat(int32_t format_code) {
  const auto format = format_code < 0 ? ge::FORMAT_ND : static_cast<ge::Format>(format_code);
  return format == ge::FORMAT_ND || format == ge::FORMAT_NCHW || format == ge::FORMAT_NHWC ||
         format == ge::FORMAT_NCDHW;
}

bool ValidateInputLayout(const InputTensorLayout &layout, std::string *err) {
  if (layout.storage_offset < 0) {
    if (err != nullptr) *err = "input storage_offset must be non-negative";
    return false;
  }
  if (!layout.stride.empty() && layout.stride.size() != layout.logical_shape.size()) {
    if (err != nullptr) *err = "input logical shape/stride rank mismatch";
    return false;
  }
  if (layout.logical_shape.size() > gert::Stride::kMaxDimNum) {
    if (err != nullptr) *err = "input rank exceeds gert::Stride capacity";
    return false;
  }
  if (layout.storage_offset != 0 && !IsBaseStorageFormat(layout.format)) {
    if (err != nullptr) *err = "nonzero storage_offset is unsupported for private-format inputs";
    return false;
  }
  if (std::any_of(layout.logical_shape.begin(), layout.logical_shape.end(), [](int64_t dim) { return dim < 0; }) ||
      std::any_of(layout.stride.begin(), layout.stride.end(), [](int64_t stride) { return stride < 0; })) {
    if (err != nullptr) *err = "input shape and stride must be concrete and non-negative";
    return false;
  }
  if (std::find(layout.logical_shape.begin(), layout.logical_shape.end(), 0) != layout.logical_shape.end()) {
    return true;
  }
  uint64_t last = static_cast<uint64_t>(layout.storage_offset);
  const auto strides = layout.stride.empty() ? ContiguousStrides(layout.logical_shape) : layout.stride;
  for (size_t i = 0; i < layout.logical_shape.size(); ++i) {
    const uint64_t size = static_cast<uint64_t>(layout.logical_shape[i]);
    const uint64_t stride = static_cast<uint64_t>(strides[i]);
    if (size > 1U && stride > (std::numeric_limits<uint64_t>::max() - last) / (size - 1U)) {
      if (err != nullptr) *err = "input storage span overflows address range";
      return false;
    }
    last += (size - 1U) * stride;
  }
  if (layout.storage_shape.empty()) {
    if (layout.storage_offset != 0) {
      if (err != nullptr) *err = "input storage_shape is required for nonzero storage_offset";
      return false;
    }
    return true;
  }
  uint64_t capacity = 1U;
  for (const int64_t dim : layout.storage_shape) {
    if (dim < 0 || (dim != 0 && capacity > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim))) {
      if (err != nullptr) *err = "input storage_shape is invalid or overflows";
      return false;
    }
    capacity *= static_cast<uint64_t>(dim);
  }
  if (last >= capacity) {
    if (err != nullptr) *err = "input view exceeds its storage capacity";
    return false;
  }
  return true;
}

InputDeviceTensor MakeInputDeviceTensor(const InputTensorLayout &layout, void *device_addr, ge::DataType dtype) {
  const auto shape = MakeStorageShape(layout);
  const auto format = MakeStorageFormat(layout.format);
  const bool has_distinct_storage = !layout.storage_shape.empty() && layout.storage_shape != layout.logical_shape;
  if (layout.storage_offset == 0 &&
      (layout.stride.empty() || (!has_distinct_storage && layout.stride == ContiguousStrides(layout.logical_shape)))) {
    return gert::Tensor(shape, format, gert::kOnDeviceHbm, dtype, device_addr);
  }

  gert::Stride stride;
  const auto effective_stride = layout.stride.empty() ? ContiguousStrides(layout.logical_shape) : layout.stride;
  for (const int64_t value : effective_stride) {
    stride.AppendStride(value);
  }
  return gert::TensorV2(shape, format, gert::kOnDeviceHbm, dtype, device_addr, nullptr, stride,
                        layout.storage_offset);
}

gert::Tensor *GetBaseTensor(InputDeviceTensor &tensor) {
  if (auto *dense = std::get_if<gert::Tensor>(&tensor)) {
    return dense;
  }
  return reinterpret_cast<gert::Tensor *>(std::get_if<gert::TensorV2>(&tensor));
}

std::vector<int64_t> DimsOf(const gert::Shape &shape) {
  std::vector<int64_t> dims(shape.GetDimNum());
  for (size_t i = 0; i < dims.size(); ++i) {
    dims[i] = shape.GetDim(static_cast<int64_t>(i));
  }
  return dims;
}

bool MatchesModelShape(const std::vector<int64_t> &shape, const std::vector<int64_t> &model_shape) {
  if (shape.size() != model_shape.size()) {
    return false;
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    if (model_shape[i] >= 0 && shape[i] != model_shape[i]) {
      return false;
    }
  }
  return true;
}

bool ModelShapeFitsStorage(const std::vector<int64_t> &model_shape, const std::vector<int64_t> &storage_shape) {
  uint64_t required = 1U;
  for (const int64_t dim : model_shape) {
    if (dim < 0 || (dim != 0 && required > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim))) {
      return false;
    }
    required *= static_cast<uint64_t>(dim);
  }
  uint64_t capacity = 1U;
  for (const int64_t dim : storage_shape) {
    if (dim < 0 || (dim != 0 && capacity > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim))) {
      return false;
    }
    capacity *= static_cast<uint64_t>(dim);
  }
  return required <= capacity;
}

gert::StorageShape RuntimeInputShape(const BuiltExecuteGraph &built, size_t index, const gert::Tensor &input) {
  const auto origin_dims = DimsOf(input.GetOriginShape());
  const auto storage_dims = DimsOf(input.GetStorageShape());
  std::vector<int64_t> selected_dims = origin_dims;
  if (index < built.input_storage_shapes.size()) {
    const auto &model_shape = built.input_storage_shapes[index];
    // The ES runner validates StorageShape against the compiled model descriptor. Runtime
    // backing storage can be larger than the span captured by that descriptor (for example,
    // a strided slice whose final storage elements are unused), so use the model shape when
    // the actual storage has enough capacity. Keep the runtime logical shape as OriginShape.
    if (!MatchesModelShape(origin_dims, model_shape)) {
      if (MatchesModelShape(storage_dims, model_shape)) {
        selected_dims = storage_dims;
      } else if (ModelShapeFitsStorage(model_shape, storage_dims)) {
        selected_dims = model_shape;
      }
    }
  }
  InputTensorLayout layout;
  layout.logical_shape = origin_dims;
  layout.storage_shape = std::move(selected_dims);
  return MakeStorageShape(layout);
}

size_t LogicalBytes(const std::vector<int64_t> &dims, ge::DataType dtype);

size_t InputBytes(const gert::Tensor &tensor) {
  return LogicalBytes(DimsOf(tensor.GetStorageShape()), tensor.GetDataType());
}

ge::DataType DtypeOrFloat(const std::vector<int32_t> &dtypes, size_t i) {
  if (i < dtypes.size()) {
    return static_cast<ge::DataType>(dtypes[i]);
  }
  return ge::DT_FLOAT;
}

bool FillInputTensorVectors(const std::vector<void *> &input_addrs,
                            const std::vector<InputTensorLayout> &input_layouts,
                            const std::vector<int32_t> &input_dtypes, std::vector<InputDeviceTensor> *in_storage,
                            std::vector<gert::Tensor *> *in_ptrs, std::string *err) {
  if (input_addrs.size() != input_layouts.size() || in_storage == nullptr || in_ptrs == nullptr) {
    if (err != nullptr) {
      *err = "input addr/shape size mismatch";
    }
    return false;
  }
  in_storage->clear();
  in_ptrs->clear();
  in_storage->reserve(input_addrs.size());
  for (size_t i = 0; i < input_addrs.size(); ++i) {
    const auto &layout = input_layouts[i];
    if (!ValidateInputLayout(layout, err)) {
      return false;
    }
    in_storage->push_back(MakeInputDeviceTensor(layout, input_addrs[i], DtypeOrFloat(input_dtypes, i)));
  }
  std::transform(in_storage->begin(), in_storage->end(), std::back_inserter(*in_ptrs),
                 [](InputDeviceTensor &t) { return GetBaseTensor(t); });
  return true;
}

bool ValidateExecuteAllocArgs(const bool ready, const std::vector<void *> &input_addrs,
                              const std::vector<InputTensorLayout> &input_layouts,
                              std::vector<ExecutedOutput> *outputs, std::string *err) {
  if (!ready) {
    if (err != nullptr) {
      *err = "ConverterSession: not converted";
    }
    return false;
  }
  if (outputs == nullptr) {
    if (err != nullptr) {
      *err = "ExecuteAlloc: outputs is null";
    }
    return false;
  }
  if (input_addrs.size() != input_layouts.size()) {
    if (err != nullptr) {
      *err = "ExecuteAlloc: input addr/shape size mismatch";
    }
    return false;
  }
  return true;
}

// Logical payload size of a tensor: prod(shape) * sizeof(dtype). Distinct from
// gert::Tensor::GetSize(), which reports the (padded) allocated block size.
size_t LogicalBytes(const std::vector<int64_t> &dims, ge::DataType dtype) {
  size_t elem_count = 1;
  for (int64_t d : dims) {
    if (d <= 0) {
      return 0;
    }
    elem_count *= static_cast<size_t>(d);
  }
  const int32_t elem_size = ge::GetSizeByDataType(dtype);
  if (elem_size <= 0) {
    return 0;
  }
  return elem_count * static_cast<size_t>(elem_size);
}

}  // namespace

// Keeps the ES runner alive until the last
// transferred output buffer is freed. For torch-fallback adopted outputs there is
// no GE block (block/mgr stay null); `adopted` owns the bridge deferred result
// instead, so the torch storage dies with the wrapped tensor.
struct OwnedOutputCtx {
  void *block = nullptr;                  // ge::MemBlock* handle, NOT the device address
  gert::TensorAddrManager mgr = nullptr;
  std::shared_ptr<void> keepalive;
  std::shared_ptr<void> fallback_owner;
};

extern "C" void FxFreeOwnedOutput(void *ctx) {
  auto *owned = static_cast<OwnedOutputCtx *>(ctx);
  if (owned == nullptr) {
    return;
  }
  if (owned->mgr != nullptr && owned->block != nullptr) {
    (void)owned->mgr(owned->block, gert::kFreeTensor, nullptr);
  }
  delete owned;
}

namespace {

// A private-format fallback result is rejected before it is described as ND.
// A later view of this buffer keeps the same address and a different shape, so the
// recorded stride belongs only to the fallback result itself.
bool ApplyAdoptedOutputLayout(gert::Tensor *t, const std::shared_ptr<FallbackSessionState> &fallback_session,
                              ExecutedOutput *view, std::string *err) {
  int32_t adopted_format = view->format;
  std::vector<int64_t> adopted_storage;
  std::vector<int64_t> adopted_stride;
  std::vector<int64_t> adopted_logical;
  if (fallback_session == nullptr ||
      !fallback_session->LookupOutputLayout(t->GetAddr(), &adopted_format, &adopted_storage, &adopted_stride,
                                           &adopted_logical)) {
    return true;
  }
  if (!FallbackFeedIsBaseFormat(adopted_format)) {
    if (err != nullptr) {
      *err = "fx_converter: graph output does not support non-base memory format";
    }
    return false;
  }
  view->format = adopted_format;
  if (!adopted_storage.empty()) {
    view->storage_shape = std::move(adopted_storage);
    view->nbytes = LogicalBytes(view->storage_shape, t->GetDataType());
  }
  if (adopted_stride.size() == view->shape.size() && adopted_logical == view->shape) {
    view->stride = std::move(adopted_stride);
  }
  return true;
}

// Move one GE-allocated output out of its shell and into an ExecutedOutput the caller owns.
// Reads the device address *before* Release: while a manager is installed, TensorData::addr_
// is an opaque ge::MemBlock* and only GetAddr() (kGetTensorAddress) yields the real address.
bool TakeOwnershipOfOutput(gert::Tensor *t, const std::shared_ptr<void> &runner,
                           const std::shared_ptr<FallbackSessionState> &fallback_session, size_t index,
                           ExecutedOutput *view, std::string *err) {
  if (t->GetPlacement() != gert::kOnDeviceHbm && t->GetSize() != 0U) {
    if (err != nullptr) *err = "ExecuteAlloc: GE returned a non-device output " + std::to_string(index);
    return false;
  }
  const std::vector<int64_t> storage_dims = DimsOf(t->GetStorageShape());
  std::vector<int64_t> origin_dims = DimsOf(t->GetOriginShape());
  // A rank-0 tensor has an empty origin. An unfilled origin falls back to storage.
  if (origin_dims.empty() && !storage_dims.empty()) {
    origin_dims = storage_dims;
  }
  view->shape = std::move(origin_dims);
  view->storage_shape = storage_dims;
  view->format = static_cast<int32_t>(t->GetStorageFormat());
  view->dtype = static_cast<int32_t>(t->GetDataType());
  view->device_addr = reinterpret_cast<uintptr_t>(t->GetAddr());
  view->nbytes = LogicalBytes(storage_dims.empty() ? view->shape : storage_dims, t->GetDataType());
  const auto fallback_owner = fallback_session == nullptr ? std::shared_ptr<void>()
                                                           : fallback_session->LookupOutputOwner(t->GetAddr());
  if (!ApplyAdoptedOutputLayout(t, fallback_session, view, err)) {
    return false;
  }

  gert::TensorAddrManager mgr = nullptr;
  void *block = t->MutableTensorData().Release(mgr);

  // A zero-numel output (some shape dim is 0) is a legitimate empty tensor: GE fills
  // the shape but allocates no payload, so addr/nbytes stay 0. Only a shape that does
  // not explain the emptiness means GE never filled the output.
  const bool zero_dim =
      std::any_of(view->shape.begin(), view->shape.end(), [](int64_t d) { return d == 0; });
  const bool dims_filled =
      !view->shape.empty() && std::none_of(view->shape.begin(), view->shape.end(), [](int64_t d) { return d < 0; });
  const bool legit_empty = zero_dim && dims_filled && view->nbytes == 0;
  const bool invalid_shape = std::any_of(view->shape.begin(), view->shape.end(), [](int64_t dim) {
    return dim < 0;
  });
  const bool invalid_dtype = ge::GetSizeByDataType(t->GetDataType()) <= 0;
  if ((!legit_empty && (view->device_addr == 0 || view->nbytes == 0)) || invalid_shape || invalid_dtype) {
    if (mgr != nullptr && block != nullptr) {
      (void)mgr(block, gert::kFreeTensor, nullptr);
    }
    if (err != nullptr) {
      *err = "ExecuteAlloc: GE did not fill output " + std::to_string(index) + " addr/shape";
    }
    return false;
  }
  if (mgr == nullptr) {
    if (fallback_owner != nullptr) {
      view->owned_ctx = new OwnedOutputCtx{nullptr, nullptr, runner, fallback_owner};
      view->free_fn = &FxFreeOwnedOutput;
      return true;
    }
    // GE reused a caller-owned address without a deferred fallback owner.
    RT_GLOG(WARNING) << "ExecuteAlloc: output " << index << " is not GE-owned, returning a borrowed view";
    return true;
  }
  view->owned_ctx = new OwnedOutputCtx{block, mgr, runner};
  view->free_fn = &FxFreeOwnedOutput;
  return true;
}

bool ResolveLogicalOutputViews(ExecutedOutput *output, std::string *err) {
  try {
    for (auto &view : output->logical_views) {
      if (!view.preserve_layout) continue;
      for (size_t i = 0; i < view.shape_exprs.size(); ++i) {
        if (view.shape_exprs[i] != nullptr) view.shape[i] = view.shape_exprs[i]->Evaluate();
      }
      for (size_t i = 0; i < view.stride_exprs.size(); ++i) {
        if (view.stride_exprs[i] != nullptr) view.stride[i] = view.stride_exprs[i]->Evaluate();
      }
      if (view.offset_expr != nullptr) view.storage_offset = view.offset_expr->Evaluate();
      if (view.derive_dense_stride) {
        view.stride.assign(view.shape.size(), 1);
        int64_t stride = 1;
        for (size_t i = view.shape.size(); i-- > 0;) {
          if (view.shape[i] < 0 || stride > std::numeric_limits<int64_t>::max() /
                                                       std::max<int64_t>(view.shape[i], 1)) {
            if (err != nullptr) *err = "ExecuteAlloc: output view shape is invalid or overflows";
            return false;
          }
          view.stride[i] = stride;
          stride *= std::max<int64_t>(view.shape[i], 1);
        }
      }
      InputTensorLayout layout;
      layout.logical_shape = view.shape;
      layout.stride = view.stride;
      layout.storage_offset = view.storage_offset;
      layout.storage_shape = output->storage_shape.empty() ? output->shape : output->storage_shape;
      std::string reason;
      if (!ValidateInputLayout(layout, &reason)) {
        if (err != nullptr) *err = "ExecuteAlloc: invalid output view: " + reason;
        return false;
      }
    }
  } catch (const std::exception &e) {
    if (err != nullptr) *err = std::string("ExecuteAlloc: output layout evaluation failed: ") + e.what();
    return false;
  }
  return true;
}

}  // namespace

class EsGraphRunner {
 public:
  ~EsGraphRunner() {
    // Unload before the binary payloads, shape rules and fallback state in
    // built_ are released, including when compilation or loading failed.
    if (session_ != nullptr && graph_added_) {
      (void)aclrtSetDevice(device_id_);
      (void)session_->RemoveGraph(kGraphId);
    }
    session_.reset();
    if (owned_stream_ != nullptr) {
      (void)aclrtDestroyStream(owned_stream_);
    }
  }

  bool Init(BuiltExecuteGraph candidate, int32_t device_id, std::string *err) {
    built_ = std::move(candidate);
    auto &built = built_;
    if (built.logical_graph == nullptr) {
      if (err != nullptr) *err = "ES runner received an empty logical graph";
      return false;
    }
    device_id_ = device_id;
    std::vector<std::string> binary_ids;
    for (size_t i = 0; i < built.binary_kernels.size(); ++i) {
      const auto &binary = built.binary_kernels[i];
      const std::string id = "fx_binary_" + std::to_string(i);
      const std::string prefix = std::string("gert.binary.") + id + ".";
      std::vector<uint8_t> content(binary.bin_bytes.begin(), binary.bin_bytes.end());
      ge::TensorDesc content_desc(ge::Shape({static_cast<int64_t>(content.size())}), ge::FORMAT_ND, ge::DT_UINT8);
      ge::Tensor content_tensor(content_desc);
      ge::AttrValue name_attr;
      ge::AttrValue content_attr;
      ge::AscendString name_key((prefix + "name").c_str());
      ge::AscendString content_key((prefix + "content").c_str());
      ge::AscendString name_value(id.c_str());
      if (binary.bin_bytes.empty() || built.main.get() == nullptr ||
          content_tensor.SetData(content) != ge::GRAPH_SUCCESS ||
          name_attr.SetAttrValue(name_value) != ge::GRAPH_SUCCESS ||
          content_attr.SetAttrValue(content_tensor) != ge::GRAPH_SUCCESS ||
          built.main->SetAttr(name_key, name_attr) != ge::GRAPH_SUCCESS ||
          built.main->SetAttr(content_key, content_attr) != ge::GRAPH_SUCCESS) {
        if (err != nullptr) *err = "failed to register ES graph kernel binary";
        return false;
      }
      binary_ids.push_back(id);
    }
    std::vector<ge::AscendString> binary_id_attrs;
    binary_id_attrs.reserve(binary_ids.size());
    for (const auto &binary_id : binary_ids) {
      binary_id_attrs.emplace_back(binary_id.c_str());
    }
    ge::AttrValue ids_attr;
    ge::AscendString ids_key("gert.binary.ids");
    if (!binary_ids.empty() &&
        (built.main.get() == nullptr || ids_attr.SetAttrValue(binary_id_attrs) != ge::GRAPH_SUCCESS ||
         built.main->SetAttr(ids_key, ids_attr) != ge::GRAPH_SUCCESS)) {
      if (err != nullptr) *err = "failed to register ES graph binary list";
      return false;
    }
    if (built.fx_builder == nullptr) {
      if (err != nullptr) *err = "ES graph builder is missing";
      return false;
    }
    graph_ = built.fx_builder->BuildGraphAndReset();
    if (graph_ == nullptr) {
      if (err != nullptr) *err = "ES graph finalization failed";
      return false;
    }

    if (!EnsureGeRuntimeInitialized(err)) {
      return false;
    }
    std::map<ge::AscendString, ge::AscendString> options;
    const std::string device_id_text = std::to_string(device_id_);
    options.emplace(ge::AscendString("ge.session_device_id"), ge::AscendString(device_id_text.c_str()));
    session_ = std::make_unique<ge::GeSession>(options);
    std::map<ge::AscendString, ge::AscendString> graph_options;
    graph_options.emplace(ge::AscendString("ge.modelFormat"), ge::AscendString("7"));
    ge::Status status = session_->AddGraph(kGraphId, *graph_, graph_options);
    if (status != ge::SUCCESS) {
      if (err != nullptr) *err = "GeSession::AddGraph failed, status=" + std::to_string(status);
      return false;
    }
    graph_added_ = true;
    status = session_->CompileGraph(kGraphId);
    if (status != ge::SUCCESS) {
      if (err != nullptr) *err = "GeSession::CompileGraph failed, status=" + std::to_string(status);
      return false;
    }
    // OM2 currently requires a null load stream. The execution stream is
    // supplied separately to RunGraphWithStreamAsync.
    status = session_->LoadGraph(kGraphId, {}, nullptr);
    if (status != ge::SUCCESS) {
      if (err != nullptr) *err = "GeSession::LoadGraph failed, status=" + std::to_string(status);
      return false;
    }
    return true;
  }

  bool Execute(void *stream, gert::Tensor **inputs, size_t input_num, const std::vector<int64_t> &storage_offsets,
               std::vector<gert::Tensor> *outputs,
               const std::vector<std::pair<std::string, int64_t>> &scalar_syms, std::string *err) {
    if (session_ == nullptr || !graph_added_ || outputs == nullptr || (input_num != 0U && inputs == nullptr) ||
        storage_offsets.size() != input_num) {
      if (err != nullptr) *err = "GE runner received invalid execution arguments";
      return false;
    }
    if (stream == nullptr) {
      if (owned_stream_ == nullptr && aclrtCreateStream(&owned_stream_) != ACL_SUCCESS) {
        if (err != nullptr) *err = "failed to create a GE execution stream";
        return false;
      }
      stream = owned_stream_;
    }
    if (storage_offsets.size() != input_num || built_.input_storage_offsets == nullptr ||
        built_.input_storage_offsets->size() != input_num) {
      if (err != nullptr) *err = "GE runner input offset metadata size mismatch";
      return false;
    }
    // Const nodes hold pointers into these stable session-owned host slots.
    std::copy(storage_offsets.begin(), storage_offsets.end(), built_.input_storage_offsets->begin());
    if (!RefreshFeedLayouts(inputs, input_num, err)) return false;
    if (!RefreshShapeListInputs(built_, inputs, input_num, scalar_syms)) {
      if (err != nullptr) *err = "failed to refresh GE execution shape inputs";
      return false;
    }
    struct DeviceBuffers {
      std::vector<void *> buffers;
      ~DeviceBuffers() {
        for (void *buffer : buffers) (void)aclrtFree(buffer);
      }
      void *Allocate(size_t size) {
        void *addr = nullptr;
        if (aclrtMalloc(&addr, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || addr == nullptr) return nullptr;
        buffers.push_back(addr);
        return addr;
      }
    } temporary_buffers;
    std::vector<gert::Tensor> ge_inputs;
    ge_inputs.reserve(input_num);
    for (size_t i = 0; i < input_num; ++i) {
      const size_t input_bytes = InputBytes(*inputs[i]);
      void *input_addr = inputs[i]->GetAddr();
      if (input_bytes == 0U) {
        input_addr = temporary_buffers.Allocate(1U);
        if (input_addr == nullptr) {
          if (err != nullptr) *err = "failed to allocate an empty GE input buffer";
          return false;
        }
      }
      // The ES feed is V1. Live stride/offset metadata uses separate host
      // endpoints, without slicing TensorV2 and retaining its V2 version tag.
      const int32_t format_code = i < built_.input_formats.size()
                                      ? built_.input_formats[i]
                                      : static_cast<int32_t>(inputs[i]->GetStorageFormat());
      const gert::StorageFormat format = MakeStorageFormat(format_code);
      const auto shape = RuntimeInputShape(built_, i, *inputs[i]);
      gert::Tensor input(shape, format, gert::kOnDeviceHbm, inputs[i]->GetDataType(), input_addr);
      input.SetSize(std::max<size_t>(input_bytes, 1U));
      ge_inputs.emplace_back(std::move(input));
    }

    // RunGraph clears caller outputs and prepares static Host buffers. The
    // stream API preserves our empty device shells and their allocation policy.
    outputs->clear();
    outputs->reserve(built_.output_dtypes.size());
    for (size_t i = 0; i < built_.output_dtypes.size(); ++i) {
      gert::Tensor output({}, MakeStorageFormat(built_.output_formats.at(i)), gert::kOnDeviceHbm,
                          static_cast<ge::DataType>(built_.output_dtypes[i]), nullptr);
      output.SetSize(0U);
      outputs->emplace_back(std::move(output));
    }
    const bool use_external_allocator = FxHasExternalAllocator();
    if (use_external_allocator && !FxRegisterExternalAllocatorForStream(session_.get(), stream)) {
      if (err != nullptr) *err = "GE runner failed to register the external allocator for its execution stream";
      return false;
    }
    const ge::Status status = session_->RunGraphWithStreamAsync(kGraphId, stream, ge_inputs, *outputs);
    // Keep ExecuteAlloc synchronous, including failure paths that submitted
    // partial work, before releasing temporary feeds or transferring outputs.
    const aclError sync_status = aclrtSynchronizeStream(stream);
    const bool allocator_unregistered =
        !use_external_allocator || FxUnregisterExternalAllocatorForStream(session_.get(), stream);
    if (status != ge::SUCCESS) {
      if (err != nullptr) *err = "Execute failed: GeSession::RunGraphWithStreamAsync failed, status=" +
                                 std::to_string(status);
      return false;
    }
    if (sync_status != ACL_SUCCESS) {
      if (err != nullptr) *err = "GE execution stream synchronization failed, ret=" + std::to_string(sync_status);
      return false;
    }
    if (!allocator_unregistered) {
      if (err != nullptr) *err = "GE runner failed to unregister the external allocator for its execution stream";
      return false;
    }
    if (outputs->size() != built_.output_dtypes.size()) {
      if (err != nullptr) *err = "GeSession::RunGraphWithStreamAsync returned an unexpected output count";
      return false;
    }
    return true;
  }

  void SetExecuteScalarSyms(const std::vector<std::pair<std::string, int64_t>> &scalar_syms) {
    scalar_syms_ = scalar_syms;
    if (built_.symbol_env != nullptr) {
      for (const auto &kv : scalar_syms_) {
        built_.symbol_env->SetValue(kv.first, kv.second);
      }
    }
  }

  const std::shared_ptr<FallbackSessionState> &FallbackSession() const { return built_.fallback_session; }

  bool HasFallbackCalls() const { return !built_.torch_fallback_calls.empty(); }

  std::mutex &ExecuteMutex() { return execute_mutex_; }

 private:
  bool RefreshFeedLayouts(gert::Tensor *const *inputs, size_t input_num, std::string *err) {
    for (const auto &refresh : built_.feed_layout_refreshes) {
      if (refresh.feed_index >= input_num || inputs[refresh.feed_index] == nullptr || refresh.payload == nullptr) {
        if (err != nullptr) *err = "GE runner invalid feed layout binding";
        return false;
      }
      const auto *input = inputs[refresh.feed_index];
      const auto &shape = input->GetOriginShape();
      const size_t rank = shape.GetDimNum();
      const auto *view = input->GetVersion() == gert::kTensorV2
                             ? reinterpret_cast<const gert::TensorV2 *>(input) : nullptr;
      if (refresh.payload->size() != 2U * rank + 1U ||
          (view != nullptr && view->GetStride().GetDimNum() != rank)) {
        if (err != nullptr) *err = "GE runner feed shape/stride rank mismatch";
        return false;
      }
      auto &payload = *refresh.payload;
      int64_t dense_stride = 1;
      for (size_t i = rank; i > 0U; --i) {
        const size_t axis = i - 1U;
        payload[axis] = shape.GetDim(axis);
        payload[rank + axis] = view == nullptr ? dense_stride : view->GetStride().GetStride(axis);
        if (view == nullptr && payload[axis] > 0) {
          if (dense_stride > std::numeric_limits<int64_t>::max() / payload[axis]) {
            if (err != nullptr) *err = "GE runner feed dense stride overflow";
            return false;
          }
          dense_stride *= payload[axis];
        }
      }
      payload.back() = 0;  // Absolute offset uses its separate host endpoint.
    }
    return true;
  }

  static bool EnsureGeRuntimeInitialized(std::string *err) {
    static std::mutex mutex;
    static bool initialized = false;
    std::lock_guard<std::mutex> lock(mutex);
    if (initialized) return true;
    const ge::Status status = ge::GEInitializeV2({});
    if (status != ge::SUCCESS) {
      if (err != nullptr) *err = "GEInitializeV2 failed, status=" + std::to_string(status);
      return false;
    }
    // GE initialization is process-wide and may be shared with other clients.
    // Destroying one runner must not finalize GE underneath their sessions.
    initialized = true;
    return true;
  }

  int32_t device_id_ = 0;
  static constexpr uint32_t kGraphId = 0U;
  bool graph_added_ = false;
  aclrtStream owned_stream_ = nullptr;
  std::unique_ptr<ge::GeSession> session_;
  std::unique_ptr<ge::Graph> graph_;
  BuiltExecuteGraph built_;
  std::vector<std::pair<std::string, int64_t>> scalar_syms_;
  std::mutex execute_mutex_;
};

BuiltExecuteGraph ConvertFromMrtIr(const fxrt::ir::GraphPtr &graph, const std::string &name) {
  if (graph == nullptr) {
    RT_GLOG(ERROR) << "ConvertFromMrtIr: graph is null";
    BuiltExecuteGraph failed;
    failed.error = "graph is null";
    return failed;
  }
  RT_GLOG(INFO) << "ConvertFromMrtIr: start, name=" << name;
  EnsureGeRt2KernelsReady();
  BuiltExecuteGraph built = BuildExecuteGraphFromMrtIr(*graph, name);
  if (std::getenv("FX_CONVERTER_DUMP_GRAPH") != nullptr) {
    std::cout << DumpExecuteGraph(built) << std::flush;
    if (built.root != nullptr && built.error.empty()) {
      (void)built.root->DumpToFile(ge::Graph::DumpFormat::kReadable,
                                   ge::AscendString("FxConverterFinalExecuteGraph"));
    }
  }
  if (built.error.empty() && built.root != nullptr) {
    RT_GLOG(INFO) << "ConvertFromMrtIr: OK, inputs=" << built.input_dtypes.size()
                  << " outputs=" << built.output_dtypes.size();
  } else if (!built.error.empty()) {
    RT_GLOG(ERROR) << "ConvertFromMrtIr: failed - " << built.error;
  }
  return built;
}

ConverterSession::~ConverterSession() {
  release();
}

bool ConverterSession::has_fallback_calls() const {
  return es_runner_ != nullptr && es_runner_->HasFallbackCalls();
}

void ConverterSession::release() {
  // Outputs already handed to the caller keep their own strong ref to the runner, so
  // dropping ours here never invalidates a live buffer.
  es_runner_.reset();
}

bool ConverterSession::Convert(const fxrt::ir::GraphPtr &graph, const std::string &name, int32_t device_id,
                                void *stream, std::string *err) {
  RT_GLOG(INFO) << "ConverterSession::Convert: name=" << name << " device_id=" << device_id
                << " stream=" << stream;
  es_runner_.reset();
  device_id_ = device_id;
  if (!EnsureAclRuntimeReady(device_id_)) {
    RT_GLOG(ERROR) << "ConverterSession::Convert: EnsureAclRuntimeReady failed";
    if (err != nullptr) {
      *err = "EnsureAclRuntimeReady failed";
    }
    return false;
  }
  // Register GE's allocator bridge before ES graph compilation. Idempotent, and a no-op when
  // FXRT_GE_USE_TORCH_MEMPOOL=0.
  FxRegisterTorchPoolAllocator();
  BuiltExecuteGraph built = ConvertFromMrtIr(graph, name);
  if (!built.error.empty()) {
    if (err != nullptr) {
      *err = built.error;
    }
    return false;
  }
  if (built.root == nullptr || built.main == nullptr) {
    RT_GLOG(ERROR) << "ConverterSession::Convert: empty ExecuteGraph";
    if (err != nullptr) {
      *err = "Convert/Build failed: empty ExecuteGraph";
    }
    return false;
  }
  if (built.logical_graph == nullptr) {
    if (err != nullptr) *err = "Convert/Build produced no logical ES graph";
    return false;
  }

  // The public GERT facade owns the complete compile/load lifecycle. FXRT only
  // supplies the single logical execute Graph; CompileGraph performs staging,
  // preparation, codegen and OM2 assembly on GE's private work graph.
  auto es_runner = std::make_shared<EsGraphRunner>();
  const auto input_dtypes = built.input_dtypes;
  const auto input_requires_base_format = built.input_requires_base_format;
  const auto output_dtypes = built.output_dtypes;
  const auto output_views = built.output_views;
  if (!es_runner->Init(std::move(built), device_id_, err)) {
    return false;
  }
  es_runner_ = std::move(es_runner);
  input_dtypes_ = input_dtypes;
  input_requires_base_format_ = input_requires_base_format;
  output_dtypes_ = output_dtypes;
  output_views_ = output_views;
  RT_GLOG(INFO) << "ConverterSession::Convert: ES facade path ready, inputs=" << input_dtypes_.size()
                << " outputs=" << output_dtypes_.size();
  return true;
}

bool ConverterSession::ExecuteAlloc(const std::vector<void *> &input_addrs,
                                    const std::vector<InputTensorLayout> &input_layouts, void *stream,
                                    std::vector<ExecutedOutput> *outputs, std::string *err,
                                    const std::vector<std::pair<std::string, int64_t>> &scalar_syms) {
  // #lizard forgives
  if (!ValidateExecuteAllocArgs(ready(), input_addrs, input_layouts, outputs, err)) {
    return false;
  }
  // Scalar values, fallback feeds and output ownership belong to the same run.
  // The Python binding releases the GIL, so serialize the entire transaction.
  std::lock_guard<std::mutex> lock(es_runner_->ExecuteMutex());
  for (size_t i = 0; i < input_layouts.size() && i < input_requires_base_format_.size(); ++i) {
    if (input_requires_base_format_[i] && !IsBaseStorageFormat(input_layouts[i].format)) {
      if (err != nullptr) {
        *err = "special format is not supported by the selected AICore input ABI at feed " + std::to_string(i);
      }
      return false;
    }
  }
  es_runner_->SetExecuteScalarSyms(scalar_syms);
  std::shared_ptr<FallbackSessionState> fallback_session;
  bool has_fallback_calls = false;
  fallback_session = es_runner_->FallbackSession();
  has_fallback_calls = es_runner_->HasFallbackCalls();
  if (fallback_session != nullptr) {
    // Safe point to drop the previous Execute's adopted storages and deferred
    // output layouts: the caller already received that Execute's outputs, and
    // recycled device addresses must not tag this Execute's outputs.
    fallback_session->FlushAdoptedResults();
    fallback_session->ClearOutputLayouts();
    std::vector<FallbackFeed> feeds;
    feeds.reserve(input_layouts.size());
    for (size_t i = 0; i < input_layouts.size(); ++i) {
      const InputTensorLayout &layout = input_layouts[i];
      const bool stride_matches = layout.stride.empty() || layout.stride.size() == layout.logical_shape.size();
      const bool contiguous = stride_matches &&
                              FallbackStrideIsContiguous(layout.logical_shape.data(), layout.stride.data(),
                                                         layout.stride.empty() ? 0U : layout.logical_shape.size());
      if (has_fallback_calls && !FallbackFeedIsBaseFormat(layout.format) &&
          (layout.storage_offset != 0 || !contiguous)) {
        if (err != nullptr) {
          *err = layout.storage_offset != 0 ? "torch fallback: nonzero storage_offset is unsupported for private formats"
                                            : FallbackNoncontiguousPrivateReason();
        }
        return false;
      }
      FallbackFeed feed;
      feed.addr = i < input_addrs.size() ? reinterpret_cast<uintptr_t>(input_addrs[i]) : 0U;
      feed.format = layout.format;
      feed.logical_shape = layout.logical_shape;
      feed.storage_shape = layout.storage_shape;
      feed.stride = layout.stride;
      feed.storage_offset = layout.storage_offset;
      feed.contiguous = contiguous;
      feeds.push_back(std::move(feed));
    }
    fallback_session->PublishFeeds(std::move(feeds));
  }

  std::vector<InputDeviceTensor> in_storage;
  std::vector<gert::Tensor *> in_ptrs;
  if (!FillInputTensorVectors(input_addrs, input_layouts, input_dtypes_, &in_storage, &in_ptrs, err)) {
    return false;
  }

  std::vector<int64_t> storage_offsets;
  storage_offsets.reserve(input_layouts.size());
  for (const auto &layout : input_layouts) storage_offsets.push_back(layout.storage_offset);
  std::vector<gert::Tensor> run_outputs;
  if (!es_runner_->Execute(stream, in_ptrs.data(), in_ptrs.size(), storage_offsets, &run_outputs, scalar_syms, err)) {
    return false;
  }

  // Hand every buffer to the caller. On failure, release what we already took so the
  // partial batch cannot leak.
  outputs->clear();
  outputs->reserve(run_outputs.size());
  for (size_t i = 0; i < run_outputs.size(); ++i) {
    ExecutedOutput view;
    view.logical_views = output_views_.at(i);
    std::shared_ptr<void> keepalive = es_runner_;
    if (!TakeOwnershipOfOutput(&run_outputs[i], keepalive, fallback_session, i, &view, err) ||
        !ResolveLogicalOutputViews(&view, err)) {
      if (view.free_fn != nullptr && view.owned_ctx != nullptr) view.free_fn(view.owned_ctx);
      for (auto &taken : *outputs) {
        if (taken.free_fn != nullptr && taken.owned_ctx != nullptr) {
          taken.free_fn(taken.owned_ctx);
        }
      }
      outputs->clear();
      return false;
    }
    outputs->push_back(std::move(view));
  }
  return true;
}

bool ConverterSession::MemcpyDeviceToDevice(void *dst, const void *src, size_t nbytes, std::string *err) {
  if (dst == nullptr || src == nullptr || nbytes == 0) {
    if (err != nullptr) {
      *err = "MemcpyDeviceToDevice: invalid args";
    }
    return false;
  }
  const aclError ret = aclrtMemcpy(dst, nbytes, src, nbytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
  if (ret != ACL_SUCCESS) {
    if (err != nullptr) {
      *err = "aclrtMemcpy D2D failed, ret=" + std::to_string(static_cast<int>(ret));
    }
    return false;
  }
  return true;
}

}  // namespace fx_converter
