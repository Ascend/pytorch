#include "lowering/torch_fallback/torch_fallback_runtime.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "acl/acl_rt.h"
#include "common/logger.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "graph/operator_factory.h"
#include "ir/tensor/tensor.h"
#include "shape/fx_sym_expr_parser.h"

namespace fx_converter {

FallbackSessionState::~FallbackSessionState() { FlushAdoptedResults(); }

void FallbackSessionState::RecordOutputLayout(void *addr, int32_t format,
                                              std::vector<int64_t> storage_shape,
                                              std::vector<int64_t> stride,
                                              std::vector<int64_t> logical_shape) {
  if (addr == nullptr) {
    return;
  }
  FallbackFeed feed;
  feed.addr = reinterpret_cast<uintptr_t>(addr);
  feed.format = format;
  feed.storage_shape = std::move(storage_shape);
  feed.stride = std::move(stride);
  feed.logical_shape = std::move(logical_shape);
  const auto existing = std::find_if(output_layouts_.begin(), output_layouts_.end(),
                                     [&feed](const std::pair<uintptr_t, FallbackFeed> &entry) {
                                       return entry.first == feed.addr;
                                     });
  if (existing != output_layouts_.end()) {
    existing->second = std::move(feed);
    return;
  }
  output_layouts_.emplace_back(feed.addr, std::move(feed));
}

bool FallbackSessionState::LookupOutputLayout(const void *addr, int32_t *format,
                                              std::vector<int64_t> *storage_shape,
                                              std::vector<int64_t> *stride,
                                              std::vector<int64_t> *logical_shape) const {
  if (addr == nullptr || format == nullptr || storage_shape == nullptr || stride == nullptr ||
      logical_shape == nullptr) {
    return false;
  }
  const auto key = reinterpret_cast<uintptr_t>(addr);
  const auto existing = std::find_if(output_layouts_.begin(), output_layouts_.end(),
                                     [key](const std::pair<uintptr_t, FallbackFeed> &entry) {
                                       return entry.first == key;
                                     });
  if (existing == output_layouts_.end()) {
    return false;
  }
  *format = existing->second.format;
  *storage_shape = existing->second.storage_shape;
  *stride = existing->second.stride;
  *logical_shape = existing->second.logical_shape;
  return true;
}

std::shared_ptr<void> FallbackSessionState::KeepAdoptedResult(const FxTorchFallbackApi *api, uintptr_t handle) {
  if (api == nullptr || handle == 0) {
    return {};
  }
  auto owner = std::shared_ptr<void>(reinterpret_cast<void *>(handle), [api](void *value) {
    if (value != nullptr && api->destroy_result != nullptr) {
      api->destroy_result(reinterpret_cast<uintptr_t>(value));
    }
  });
  adopted_results_.push_back(owner);
  return owner;
}

void FallbackSessionState::RecordOutputOwner(const void *addr, const std::shared_ptr<void> &owner) {
  if (addr == nullptr || owner == nullptr) {
    return;
  }
  const auto key = reinterpret_cast<uintptr_t>(addr);
  const auto existing = std::find_if(output_owners_.begin(), output_owners_.end(),
                                     [key](const auto &entry) { return entry.first == key; });
  if (existing == output_owners_.end()) {
    output_owners_.emplace_back(key, owner);
  } else {
    existing->second = owner;
  }
}

std::shared_ptr<void> FallbackSessionState::LookupOutputOwner(const void *addr) const {
  if (addr == nullptr) {
    return {};
  }
  const auto key = reinterpret_cast<uintptr_t>(addr);
  const auto existing = std::find_if(output_owners_.begin(), output_owners_.end(),
                                     [key](const auto &entry) { return entry.first == key; });
  return existing == output_owners_.end() ? std::shared_ptr<void>() : existing->second.lock();
}

void FallbackSessionState::FlushAdoptedResults() {
  output_owners_.clear();
  adopted_results_.clear();
}

struct TorchFallbackRuntimeCache {
  std::vector<FxTorchValue> arguments;
  std::deque<std::vector<FxTorchValue>> list_elements;
  std::vector<std::vector<int64_t>> sizes;
  std::vector<std::vector<int64_t>> strides;
  std::vector<std::vector<int64_t>> storage_sizes;
  std::vector<std::vector<int64_t>> output_sizes;
  std::vector<std::vector<int64_t>> output_strides;
};

namespace {

constexpr size_t kErrorCapacity = 1024U;
constexpr size_t kFallbackMaxRank =
    sizeof(((FxTorchFallbackResultMetadata *)nullptr)->sizes) / sizeof(int64_t);
constexpr size_t kFallbackMaxStorageRank = kFallbackMaxRank;
constexpr size_t kFallbackApiBaseSize = offsetof(FxTorchFallbackApi, export_storage) +
                                         sizeof(((FxTorchFallbackApi *)nullptr)->export_storage);

bool HasStructField(uint32_t struct_size, size_t offset, size_t field_size) {
  return static_cast<size_t>(struct_size) >= offset &&
         static_cast<size_t>(struct_size) - offset >= field_size;
}

std::atomic<const FxTorchFallbackApi *> g_torch_fallback_api{nullptr};
std::atomic<const FxTorchFallbackApi *> g_python_call_api{nullptr};

bool HasFallbackExportLayout(const TorchFallbackCall *call) {
  return call != nullptr && call->api != nullptr &&
         HasStructField(call->api->struct_size, offsetof(FxTorchFallbackApi, export_layout),
                        sizeof(call->api->export_layout)) && call->api->export_layout != nullptr;
}

bool HasFallbackExportStrides(const TorchFallbackCall *call) {
  return call != nullptr && call->api != nullptr &&
         HasStructField(call->api->struct_size, offsetof(FxTorchFallbackApi, export_strides),
                        sizeof(call->api->export_strides)) && call->api->export_strides != nullptr;
}

uint32_t ValueKind(const fxrt::ir::ValuePtr &value, uint32_t *element_kind, std::string *error);

uint32_t AtomValueKind(const fxrt::ir::ValuePtr &value) {
  if (value == nullptr || value->IsNone()) {
    return FX_TORCH_VALUE_NONE;
  }
  if (value->IsTensor()) {
    return FX_TORCH_VALUE_TENSOR;
  }
  if (value->IsDouble()) {
    return FX_TORCH_VALUE_DOUBLE;
  }
  if (value->IsInt() || value->IsSymbol()) {
    return FX_TORCH_VALUE_INT;
  }
  if (value->IsBool()) {
    return FX_TORCH_VALUE_BOOL;
  }
  if (value->IsString()) {
    return FX_TORCH_VALUE_STRING;
  }
  return std::numeric_limits<uint32_t>::max();
}

uint32_t ListValueKind(const fxrt::ir::ValuePtr &value, uint32_t *element_kind, std::string *error) {
  const auto &tuple = value->ToTuple();
  uint32_t common_kind = FX_TORCH_VALUE_INT;
  for (size_t i = 0; i < tuple->Size(); ++i) {
    uint32_t nested_kind = FX_TORCH_VALUE_NONE;
    const uint32_t kind = ValueKind((*tuple)[i], &nested_kind, error);
    if (kind == FX_TORCH_VALUE_LIST || kind == std::numeric_limits<uint32_t>::max()) {
      if (error != nullptr && error->empty()) {
        *error = "torch fallback: nested lists are not supported yet";
      }
      return std::numeric_limits<uint32_t>::max();
    }
    if (i == 0U) {
      common_kind = kind;
    } else if (kind != common_kind) {
      if (error != nullptr) {
        *error = "torch fallback: list arguments must be homogeneous";
      }
      return std::numeric_limits<uint32_t>::max();
    }
  }
  if (element_kind != nullptr) {
    *element_kind = common_kind;
  }
  return FX_TORCH_VALUE_LIST;
}

uint32_t ValueKind(const fxrt::ir::ValuePtr &value, uint32_t *element_kind, std::string *error) {
  if (element_kind != nullptr) {
    *element_kind = FX_TORCH_VALUE_NONE;
  }
  if (value != nullptr && value->IsTuple()) {
    return ListValueKind(value, element_kind, error);
  }
  const uint32_t kind = AtomValueKind(value);
  if (kind == std::numeric_limits<uint32_t>::max() && error != nullptr) {
    *error = "torch fallback: unsupported argument kind";
  }
  return kind;
}

fxrt::ir::ValuePtr NormalizeValue(const fxrt::ir::ValuePtr &value, FxSymbolEnv *env) {
  if (value == nullptr || env == nullptr) {
    return value;
  }
  if (value->IsSymbol()) {
    const auto parsed = ParseFxSymExpr(value->ToSymbol(), *env);
    return parsed == nullptr ? nullptr : fxrt::ir::MakeIntrusive<fxrt::ir::Value>(parsed);
  }
  if (!value->IsTuple()) {
    return value;
  }
  std::vector<fxrt::ir::ValuePtr> elements;
  const auto &tuple = value->ToTuple();
  elements.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); ++i) {
    auto element = NormalizeValue((*tuple)[i], env);
    if (element == nullptr) {
      return nullptr;
    }
    elements.push_back(std::move(element));
  }
  auto normalized = fxrt::ir::MakeIntrusive<fxrt::ir::Tuple>(std::move(elements));
  return fxrt::ir::MakeIntrusive<fxrt::ir::Value>(normalized);
}

std::shared_ptr<FallbackShapeExpr> BuildFallbackShapeExpr(const fxrt::ir::SymbolicExprPtr &expr,
                                                          const FxSymbolEnv &env) {
  if (expr == nullptr) {
    return nullptr;
  }
  using Kind = fxrt::ir::SymbolicExpr::Kind;
  auto result = std::make_shared<FallbackShapeExpr>();
  if (expr->GetKind() == Kind::Constant) {
    result->kind = FallbackShapeExprKind::kConstant;
    result->value = static_cast<const fxrt::ir::SymbolicConst *>(expr.get())->GetValue();
    return result;
  }
  if (expr->GetKind() == Kind::Variable) {
    const auto &name = static_cast<const fxrt::ir::SymbolicVar *>(expr.get())->GetName();
    const auto seed = std::find_if(env.seeds.begin(), env.seeds.end(), [&](const FxSymbolEnv::Seed &candidate) {
      return candidate.name == name;
    });
    if (seed == env.seeds.end() || seed->feed_index < 0 || seed->dim_index < 0) {
      return nullptr;
    }
    result->kind = FallbackShapeExprKind::kDimension;
    result->feed_index = static_cast<size_t>(seed->feed_index);
    result->dim_index = static_cast<size_t>(seed->dim_index);
    return result;
  }
  const auto *binary = dynamic_cast<const fxrt::ir::SymbolicBinaryOp *>(expr.get());
  if (binary == nullptr) {
    return nullptr;
  }
  switch (expr->GetKind()) {
    case Kind::Add: result->kind = FallbackShapeExprKind::kAdd; break;
    case Kind::Mul: result->kind = FallbackShapeExprKind::kMul; break;
    case Kind::TrueDiv: result->kind = FallbackShapeExprKind::kTrueDiv; break;
    case Kind::FloorDiv: result->kind = FallbackShapeExprKind::kFloorDiv; break;
    case Kind::CeilDiv: result->kind = FallbackShapeExprKind::kCeilDiv; break;
    case Kind::Mod: result->kind = FallbackShapeExprKind::kMod; break;
    case Kind::Min: result->kind = FallbackShapeExprKind::kMin; break;
    default: return nullptr;
  }
  result->lhs = BuildFallbackShapeExpr(binary->getLHS(), env);
  result->rhs = BuildFallbackShapeExpr(binary->getRHS(), env);
  return result->lhs == nullptr || result->rhs == nullptr ? nullptr : result;
}

void SetSimpleValue(const fxrt::ir::ValuePtr &value, FxTorchValue *output) {
  if (value == nullptr || output == nullptr) {
    return;
  }
  if (value->IsInt()) {
    output->kind = FX_TORCH_VALUE_INT;
    output->int_value = value->ToInt();
  } else if (value->IsDouble()) {
    output->kind = FX_TORCH_VALUE_DOUBLE;
    output->double_value = value->ToDouble();
  } else if (value->IsBool()) {
    output->kind = FX_TORCH_VALUE_BOOL;
    output->int_value = value->ToBool() ? 1 : 0;
  } else if (value->IsString()) {
    const std::string &text = value->ToString();
    output->kind = FX_TORCH_VALUE_STRING;
    output->string_value = {text.data(), text.size()};
  } else if (value->IsSymbol()) {
    // Plan creation runs before feeds bind symbols. Leave a placeholder; Refresh
    // evaluates the expression on every Execute.
    output->kind = FX_TORCH_VALUE_INT;
    output->int_value = 0;
  }
}

void FillContiguousStrides(const std::vector<int64_t> &sizes, std::vector<int64_t> *strides) {
  strides->resize(sizes.size());
  int64_t next = 1;
  for (size_t i = sizes.size(); i > 0U; --i) {
    (*strides)[i - 1U] = next;
    next *= sizes[i - 1U];
  }
}

constexpr uint32_t kFormatFractalNz =
    static_cast<uint32_t>(fxrt::ir::MemoryFormat::FORMAT_FRACTAL_NZ);
constexpr int64_t kNzCube = 16;
constexpr int64_t kNzBlockBytes = 32;

bool IsNzStorageRank(size_t rank) { return rank == 4U || rank == 5U; }

void InferNzStorageSizes(const std::vector<int64_t> &origin, int64_t itemsize,
                         std::vector<int64_t> *out) {
  std::vector<int64_t> dim = origin;
  if (dim.empty()) {
    dim.push_back(1);
  }
  if (dim.size() == 1U) {
    dim.push_back(1);
  }
  const int64_t item = itemsize > 2 ? 2 : (itemsize < 1 ? 1 : itemsize);
  const int64_t c0 = kNzBlockBytes / item;
  out->clear();
  for (size_t i = 0; i + 2U < dim.size(); ++i) {
    out->push_back(dim[i]);
  }
  const int64_t h = dim[dim.size() - 2U];
  const int64_t w = dim[dim.size() - 1U];
  out->push_back((w + c0 - 1) / c0);
  out->push_back((h + kNzCube - 1) / kNzCube);
  out->push_back(kNzCube);
  out->push_back(c0);
}

void CopyGeStorageDims(const gert::Shape &storage_dims, std::vector<int64_t> *out) {
  out->resize(storage_dims.GetDimNum());
  for (size_t i = 0; i < storage_dims.GetDimNum(); ++i) {
    (*out)[i] = storage_dims.GetDim(i);
  }
}

bool IsNzLayoutRank(size_t storage_rank, size_t origin_rank) {
  return IsNzStorageRank(storage_rank) && storage_rank == origin_rank + 2U;
}

bool NeedsNzMetadata(const TorchFallbackArgument &argument, size_t origin_rank, uint32_t *format) {
  *format = argument.tensor_format;
  if (IsNzLayoutRank(argument.ir_storage_sizes.size(), origin_rank)) {
    *format = kFormatFractalNz;
    return true;
  }
  return *format == kFormatFractalNz;
}

// Called only when NeedsNzMetadata accepted: the IR storage rank or the IR
// format already says FRACTAL_NZ. Derive the physical storage sizes.
void SelectNzStorageSizes(const TorchFallbackArgument &argument, const gert::Shape &origin,
                          const gert::Shape &ge_storage, int32_t ge_dtype, uint32_t *format,
                          std::vector<int64_t> *storage_sizes) {
  const size_t origin_rank = origin.GetDimNum();
  const bool ir_nz = IsNzLayoutRank(argument.ir_storage_sizes.size(), origin_rank);
  if (ir_nz) {
    *storage_sizes = argument.ir_storage_sizes;
    *format = kFormatFractalNz;
    return;
  }
  *format = kFormatFractalNz;
  if (IsNzLayoutRank(ge_storage.GetDimNum(), origin_rank)) {
    CopyGeStorageDims(ge_storage, storage_sizes);
    return;
  }
  std::vector<int64_t> origin_sizes(origin_rank);
  for (size_t i = 0; i < origin_rank; ++i) {
    origin_sizes[i] = origin.GetDim(i);
  }
  if (origin_sizes.empty()) {
    CopyGeStorageDims(ge_storage, storage_sizes);
    origin_sizes = *storage_sizes;
  }
  const int32_t item = ge::GetSizeByDataType(static_cast<ge::DataType>(ge_dtype));
  InferNzStorageSizes(origin_sizes, item, storage_sizes);
}

bool FillConstantValue(const fxrt::ir::ValuePtr &value, TorchFallbackRuntimeCache *storage,
                       FxTorchValue *output) {
  if (value == nullptr || value->IsNone()) {
    output->kind = FX_TORCH_VALUE_NONE;
    return true;
  }
  if (!value->IsTuple()) {
    SetSimpleValue(value, output);
    return output->kind != FX_TORCH_VALUE_NONE;
  }
  const auto &tuple = value->ToTuple();
  storage->list_elements.emplace_back(tuple->Size());
  auto &elements = storage->list_elements.back();
  for (size_t i = 0; i < elements.size(); ++i) {
    if (!FillConstantValue((*tuple)[i], storage, &elements[i]) ||
        elements[i].kind == FX_TORCH_VALUE_LIST) {
      return false;
    }
  }
  output->kind = FX_TORCH_VALUE_LIST;
  output->elements = elements.data();
  output->element_count = elements.size();
  return true;
}

bool CopyShapeDims(const gert::Shape &shape, std::vector<int64_t> *out, bool *has_zero) {
  const size_t rank = shape.GetDimNum();
  out->resize(rank);
  bool zero = false;
  for (size_t i = 0; i < rank; ++i) {
    const int64_t dim = shape.GetDim(i);
    if (dim < 0) {
      return false;
    }
    zero = zero || (dim == 0);
    (*out)[i] = dim;
  }
  if (has_zero != nullptr) {
    *has_zero = zero;
  }
  return true;
}

bool FillEagerTensor(const gert::Tensor *tensor, int32_t dtype, std::vector<int64_t> *sizes,
                     std::vector<int64_t> *strides, FxTorchTensorView *output) {
  if (tensor == nullptr || output == nullptr || sizes == nullptr || strides == nullptr) {
    return false;
  }
  // Logical sizes must come from OriginShape. FRACTAL_NZ StorageShape is a 4D/5D
  // physical layout (or a 1-D nbytes blob); feeding it to Torch as sizes makes
  // npu_quant_matmul see a 4D x2 against a 2D scale.
  const auto &origin = tensor->GetOriginShape();
  const auto &storage_shape = tensor->GetStorageShape();
  const auto &shape = origin.GetDimNum() > 0U ? origin : storage_shape;
  bool is_empty = false;
  if (!CopyShapeDims(shape, sizes, &is_empty)) {
    return false;
  }
  // Torch NPU represents a zero-element tensor with a null data pointer and
  // zero storage. A null address is invalid only when the tensor has data.
  if (tensor->GetAddr() == nullptr && !is_empty) {
    return false;
  }
  if (tensor->GetVersion() == gert::kTensorV2) {
    const auto *view = reinterpret_cast<const gert::TensorV2 *>(tensor);
    strides->resize(view->GetStride().GetDimNum());
    for (size_t i = 0; i < strides->size(); ++i) {
      (*strides)[i] = view->GetStride().GetStride(i);
    }
    output->storage_offset = view->GetOffset();
  } else {
    FillContiguousStrides(*sizes, strides);
    output->storage_offset = 0;
  }
  output->data = const_cast<void *>(tensor->GetAddr());
  output->sizes = sizes->data();
  output->strides = strides->data();
  output->rank = sizes->size();
  output->dtype = dtype;
  output->storage_format = static_cast<uint32_t>(ge::FORMAT_ND);
  output->nbytes = tensor->GetSize();
  output->storage_sizes = nullptr;
  output->storage_rank = 0U;
  return true;
}

// The live feed stride describes the graph input. An in-graph view shares that
// address, but its GE tensor keeps the parent storage shape. When the IR stride
// disagrees, rebuild sizes from the IR logical shape and use the IR stride.
bool ConcreteShape(const std::vector<int64_t> &shape) {
  return !shape.empty() && std::all_of(shape.begin(), shape.end(), [](int64_t dim) { return dim >= 0; });
}

void ApplyLiveOrIrStride(const TorchFallbackArgument &argument, const FallbackFeed *feed,
                         std::vector<int64_t> *sizes, std::vector<int64_t> *strides,
                         FxTorchTensorView *output) {
  const bool ir_shape = ConcreteShape(argument.ir_logical_shape);
  const size_t stride_rank = ir_shape ? argument.ir_logical_shape.size() : sizes->size();
  const bool ir_matches = argument.ir_stride.size() == stride_rank;
  const bool ir_stride_concrete = ir_matches &&
      std::all_of(argument.ir_stride.begin(), argument.ir_stride.end(), [](int64_t step) { return step >= 0; });
  const bool feed_matches = feed != nullptr && feed->logical_shape == *sizes &&
                            feed->stride.size() == sizes->size();
  const bool ir_differs = ir_stride_concrete && feed_matches && argument.ir_stride != feed->stride;
  const std::vector<int64_t> *chosen = nullptr;
  if (feed_matches && !ir_differs) {
    chosen = &feed->stride;
    output->storage_offset = feed->storage_offset;
  } else if (ir_stride_concrete) {
    chosen = &argument.ir_stride;
    if (ir_shape) {
      *sizes = argument.ir_logical_shape;
      output->sizes = sizes->data();
      output->rank = sizes->size();
    }
  }
  if (chosen == nullptr) {
    return;
  }
  *strides = *chosen;
  output->strides = strides->data();
}

void ApplyInputNzMetadata(const TorchFallbackArgument &argument, const gert::Tensor *tensor,
                          std::vector<int64_t> *storage_sizes, FxTorchTensorView *output) {
  if (tensor == nullptr || output == nullptr || storage_sizes == nullptr) {
    return;
  }
  uint32_t format = argument.tensor_format;
  if (!NeedsNzMetadata(argument, output->rank, &format)) {
    return;
  }
  SelectNzStorageSizes(argument, tensor->GetOriginShape(), tensor->GetStorageShape(), output->dtype,
                       &format, storage_sizes);
  output->storage_format = format;
  output->storage_sizes = storage_sizes->data();
  output->storage_rank = storage_sizes->size();
}

bool ReleaseEagerOutputTensorData(gert::EagerOpExecutionContext *ctx, size_t index) {
  if (ctx == nullptr) {
    return false;
  }
  auto *tensor = const_cast<gert::Tensor *>(ctx->GetOutputTensor(index));
  if (tensor == nullptr) {
    return false;
  }
  gert::TensorAddrManager manager = nullptr;
  void *block = tensor->MutableTensorData().Release(manager);
  if (manager != nullptr && block != nullptr && manager(block, gert::kFreeTensor, nullptr) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "TorchFallbackEagerOp: failed to release previous output storage " << index;
    return false;
  }
  return true;
}

gert::Tensor *MallocEagerOutputTensor(gert::EagerOpExecutionContext *ctx, size_t index,
                                      const gert::StorageShape &shape,
                                      const gert::StorageFormat &format, ge::DataType dtype) {
  gert::StorageShape allocation_shape;
  allocation_shape.MutableOriginShape() = shape.GetOriginShape();
  allocation_shape.MutableStorageShape() = shape.GetStorageShape();
  bool is_empty = false;
  auto &allocation_dims = allocation_shape.MutableStorageShape();
  for (size_t i = 0; i < allocation_dims.GetDimNum(); ++i) {
    if (allocation_dims.GetDim(i) == 0) {
      allocation_dims.SetDim(i, 1);
      is_empty = true;
    }
  }

  auto *tensor = ctx->MallocOutputTensor(index, allocation_shape, format, dtype);
  // ExecuteCustomOp output slots can inherit a compile-time nbytes from the
  // OpDesc (indexer [T,1,512] int32). GE then skips L1 malloc because
  // TensorData::GetSize()>0, but the address is still 0. FillEagerTensor
  // rejects a null address on a non-empty tensor — DSV4 "output 0 malloc".
  if (tensor != nullptr && tensor->GetAddr() == nullptr && !is_empty) {
    tensor->SetSize(0);
    tensor = ctx->MallocOutputTensor(index, allocation_shape, format, dtype);
  }
  if (tensor != nullptr) {
    tensor->SetPlacement(gert::kOnDeviceHbm);
  }
  if (tensor != nullptr && is_empty) {
    // GE 9.2 rejects zero-byte allocations. Retain the padded allocation as a
    // valid sentinel while publishing the real empty shape to all consumers.
    tensor->MutableOriginShape() = shape.GetOriginShape();
    tensor->MutableStorageShape() = shape.GetStorageShape();
  }
  return tensor;
}

bool PrepareArgumentTree(TorchFallbackArgument *argument, TorchFallbackRuntimeCache *cache,
                         FxTorchValue *output) {
  if (argument->tensor_index != kNoTorchFallbackTensor) {
    argument->runtime_shape_slot = cache->sizes.size();
    cache->sizes.emplace_back();
    cache->strides.emplace_back();
    cache->storage_sizes.emplace_back();
    output->kind = FX_TORCH_VALUE_TENSOR;
    return true;
  }
  if (!argument->elements.empty()) {
    argument->runtime_list_slot = cache->list_elements.size();
    cache->list_elements.emplace_back(argument->elements.size());
    auto &elems = cache->list_elements.back();
    for (size_t i = 0; i < elems.size(); ++i) {
      if (!PrepareArgumentTree(&argument->elements[i], cache, &elems[i])) {
        return false;
      }
    }
    output->kind = FX_TORCH_VALUE_LIST;
    output->elements = elems.data();
    output->element_count = elems.size();
    return true;
  }
  return FillConstantValue(argument->value, cache, output);
}

bool PrepareRuntimeCache(TorchFallbackCall *call, std::string *error) {
  call->cache = std::make_unique<TorchFallbackRuntimeCache>();
  auto *cache = call->cache.get();
  cache->arguments.resize(call->arguments.size());
  cache->output_sizes.resize(call->output_dtypes.size());
  cache->output_strides.resize(call->output_dtypes.size());
  for (size_t i = 0; i < call->arguments.size(); ++i) {
    if (!PrepareArgumentTree(&call->arguments[i], cache, &cache->arguments[i])) {
      if (error != nullptr) {
        *error = "torch fallback: failed to materialize constant arguments";
      }
      return false;
    }
  }
  return true;
}

bool EvaluateFallbackShapeExpr(const FallbackShapeExpr *expr, const gert::EagerOpExecutionContext *context,
                               size_t shape_input_base, int64_t *value) {
  if (expr == nullptr || context == nullptr || value == nullptr) {
    return false;
  }
  if (expr->kind == FallbackShapeExprKind::kConstant) {
    *value = expr->value;
    return true;
  }
  if (expr->kind == FallbackShapeExprKind::kDimension) {
    const auto *tensor = context->GetInputTensor(shape_input_base + expr->feed_index);
    if (tensor == nullptr || expr->dim_index >= tensor->GetOriginShape().GetDimNum()) {
      return false;
    }
    *value = tensor->GetOriginShape().GetDim(expr->dim_index);
    return true;
  }
  int64_t lhs = 0;
  int64_t rhs = 0;
  if (!EvaluateFallbackShapeExpr(expr->lhs.get(), context, shape_input_base, &lhs) ||
      !EvaluateFallbackShapeExpr(expr->rhs.get(), context, shape_input_base, &rhs)) {
    return false;
  }
  __int128 wide = 0;
  switch (expr->kind) {
    case FallbackShapeExprKind::kAdd:
      wide = static_cast<__int128>(lhs) + rhs;
      break;
    case FallbackShapeExprKind::kMul:
      wide = static_cast<__int128>(lhs) * rhs;
      break;
    case FallbackShapeExprKind::kTrueDiv:
      if (rhs == 0 || (lhs == std::numeric_limits<int64_t>::min() && rhs == -1)) return false;
      *value = lhs / rhs;
      return true;
    case FallbackShapeExprKind::kFloorDiv: {
      if (rhs == 0 || (lhs == std::numeric_limits<int64_t>::min() && rhs == -1)) return false;
      int64_t quotient = lhs / rhs;
      const int64_t remainder = lhs % rhs;
      if (remainder != 0 && ((remainder < 0) != (rhs < 0))) --quotient;
      *value = quotient;
      return true;
    }
    case FallbackShapeExprKind::kCeilDiv: {
      if (rhs == 0 || (lhs == std::numeric_limits<int64_t>::min() && rhs == -1)) return false;
      int64_t quotient = lhs / rhs;
      const int64_t remainder = lhs % rhs;
      if (remainder != 0 && ((remainder < 0) == (rhs < 0))) ++quotient;
      *value = quotient;
      return true;
    }
    case FallbackShapeExprKind::kMod:
      if (rhs == 0 || (lhs == std::numeric_limits<int64_t>::min() && rhs == -1)) return false;
      *value = lhs % rhs;
      return true;
    case FallbackShapeExprKind::kMin:
      *value = std::min(lhs, rhs);
      return true;
    default:
      return false;
  }
  if (wide < std::numeric_limits<int64_t>::min() || wide > std::numeric_limits<int64_t>::max()) {
    return false;
  }
  *value = static_cast<int64_t>(wide);
  return true;
}

bool RefreshSymbolicScalar(const TorchFallbackArgument *argument, const TorchFallbackCall *call,
                           const gert::EagerOpExecutionContext *context, FxTorchValue *output) {
  if (argument == nullptr || call == nullptr || output == nullptr || argument->value == nullptr ||
      !argument->value->IsSymbol()) {
    return true;
  }
  output->kind = FX_TORCH_VALUE_INT;
  return EvaluateFallbackShapeExpr(argument->shape_expr.get(), context, call->tensor_count,
                                   &output->int_value);
}

const FallbackFeed *FindFeedByAddr(const std::vector<FallbackFeed> &feeds, const void *addr) {
  if (addr == nullptr) {
    return nullptr;
  }
  const auto key = reinterpret_cast<uintptr_t>(addr);
  const auto existing = std::find_if(feeds.begin(), feeds.end(),
                                     [key](const FallbackFeed &feed) { return feed.addr == key; });
  return existing == feeds.end() ? nullptr : &*existing;
}

bool ApplyLiveFeedFormat(const FallbackFeed &feed, std::vector<int64_t> *storage_sizes,
                         FxTorchTensorView *output) {
  if (!feed.contiguous && !FallbackFeedIsBaseFormat(feed.format)) {
    RT_GLOG(ERROR) << "TorchFallbackEagerOp: " << FallbackNoncontiguousPrivateReason();
    return false;
  }
  if (FallbackFeedIsBaseFormat(feed.format)) {
    return true;
  }
  *storage_sizes = feed.storage_shape;
  output->storage_format = static_cast<uint32_t>(feed.format);
  if (storage_sizes->empty()) {
    return true;
  }
  output->storage_sizes = storage_sizes->data();
  output->storage_rank = storage_sizes->size();
  return true;
}

bool RefreshArgumentTree(TorchFallbackArgument *argument, gert::EagerOpExecutionContext *ctx,
                         size_t *input_index, const std::vector<FallbackFeed> &feeds,
                         TorchFallbackRuntimeCache *cache, const TorchFallbackCall *call, FxTorchValue *output) {
  if (argument->tensor_index != kNoTorchFallbackTensor) {
    const auto *tensor = ctx->GetInputTensor((*input_index)++);
    const size_t slot = argument->runtime_shape_slot;
    if (tensor == nullptr ||
        !FillEagerTensor(tensor, static_cast<int32_t>(tensor->GetDataType()), &cache->sizes[slot],
                         &cache->strides[slot], &output->tensor)) {
      return false;
    }
    const FallbackFeed *feed = FindFeedByAddr(feeds, tensor->GetAddr());
    if (feed != nullptr) {
      if (!ApplyLiveFeedFormat(*feed, &cache->storage_sizes[slot], &output->tensor)) {
        return false;
      }
    } else {
      ApplyInputNzMetadata(*argument, tensor, &cache->storage_sizes[slot], &output->tensor);
    }
    // A boxed view already holds this Execute's logical shape, stride and
    // offset. The same allocation can also be a feed (including square
    // transposes); its original layout must not overwrite the derived view.
    if (tensor->GetVersion() != gert::kTensorV2) {
      ApplyLiveOrIrStride(*argument, feed, &cache->sizes[slot], &cache->strides[slot], &output->tensor);
    }
    output->kind = FX_TORCH_VALUE_TENSOR;
    return true;
  }
  if (argument->elements.empty()) {
    return RefreshSymbolicScalar(argument, call, ctx, output);
  }
  auto &elems = cache->list_elements[argument->runtime_list_slot];
  for (size_t i = 0; i < argument->elements.size(); ++i) {
    if (!RefreshArgumentTree(&argument->elements[i], ctx, input_index, feeds, cache, call,
                             &elems[i])) {
      return false;
    }
  }
  return true;
}

bool RefreshAllArguments(TorchFallbackCall *call, gert::EagerOpExecutionContext *ctx) {
  if (call->cache == nullptr || call->session == nullptr) {
    return false;
  }
  size_t input_index = 0;
  auto *cache = call->cache.get();
  for (size_t i = 0; i < call->arguments.size(); ++i) {
    if (!RefreshArgumentTree(&call->arguments[i], ctx, &input_index,
                             call->session->Feeds(), cache, call, &cache->arguments[i])) {
      return false;
    }
  }
  return true;
}

bool FetchDeviceId(TorchFallbackCall *call, int32_t *device_id) {
  if (call->cached_device_id >= 0) {
    *device_id = call->cached_device_id;
    return true;
  }
  int32_t id = 0;
  if (aclrtGetDevice(&id) != ACL_SUCCESS) {
    return false;
  }
  call->cached_device_id = id;
  *device_id = id;
  return true;
}

// Ref-paired outputs (mutating out-param ops, e.g. vllm.dsa_forward) share
// the paired input's buffer: MakeOutputRefInput marks the output as an alias
// of the input (validated by the eager op execution context against the
// registered op proto's same-name input/output declaration). Paired outputs
// must never go through MallocOutputTensor: once the op proto is registered,
// its "output name must not exist in input" validation rejects the same-name
// ref pair.
size_t RefInputSlotForOutput(const TorchFallbackCall *call, size_t output_index) {
  for (const auto *pairs : {&call->ref_pairs, &call->read_alias_pairs}) {
    for (const auto &pair : *pairs) {
      if (pair.second != output_index || pair.first >= call->arguments.size()) continue;
      const size_t slot = call->arguments[pair.first].tensor_index;
      if (slot != kNoTorchFallbackTensor) return slot;
    }
  }
  return kNoTorchFallbackTensor;
}

const FxTorchValue *RefInputValue(const TorchFallbackCall *call, const TorchFallbackRuntimeCache *cache,
                                  size_t output_index) {
  if (cache == nullptr) {
    return nullptr;
  }
  for (const auto *pairs : {&call->ref_pairs, &call->read_alias_pairs}) {
    for (const auto &pair : *pairs) {
      if (pair.second != output_index || pair.first >= cache->arguments.size()) continue;
      return &cache->arguments[pair.first];
    }
  }
  return nullptr;
}

bool SameLogicalShape(const FxTorchTensorView &tensor, const FxTorchFallbackResultMetadata &meta) {
  if (tensor.rank != meta.rank || (tensor.rank != 0U && tensor.sizes == nullptr)) {
    return false;
  }
  for (size_t i = 0; i < meta.rank; ++i) {
    if (tensor.sizes[i] != meta.sizes[i]) {
      return false;
    }
  }
  return true;
}

bool IsReadAliasOutput(const TorchFallbackCall &call, size_t index) {
  return std::any_of(call.read_alias_pairs.begin(), call.read_alias_pairs.end(),
                     [index](const auto &pair) { return pair.second == index; });
}

bool LogicalDataAddress(const FxTorchTensorView &tensor, uintptr_t *address) {
  if (address == nullptr || tensor.data == nullptr || tensor.storage_offset < 0) {
    return false;
  }
  const int32_t item_size = ge::GetSizeByDataType(static_cast<ge::DataType>(tensor.dtype));
  if (item_size <= 0 || static_cast<uint64_t>(tensor.storage_offset) >
                            std::numeric_limits<uintptr_t>::max() / static_cast<uint64_t>(item_size)) {
    return false;
  }
  const uintptr_t base = reinterpret_cast<uintptr_t>(tensor.data);
  const uintptr_t byte_offset = static_cast<uintptr_t>(tensor.storage_offset) *
                                static_cast<uintptr_t>(item_size);
  if (byte_offset > std::numeric_limits<uintptr_t>::max() - base) {
    return false;
  }
  *address = base + byte_offset;
  return true;
}

bool SameDenseElementCount(const FxTorchTensorView &input, const FxTorchFallbackResultMetadata &meta) {
  if (input.storage_offset < 0 || !FallbackFeedIsBaseFormat(static_cast<int32_t>(input.storage_format)) ||
      !FallbackStrideIsContiguous(input.sizes, input.strides, input.rank)) return false;
  const auto count = [](const int64_t *sizes, size_t rank, uint64_t *value) {
    *value = 1U;
    if (rank != 0U && sizes == nullptr) return false;
    for (size_t i = 0; i < rank; ++i) {
      if (sizes[i] < 0 || (sizes[i] != 0 && *value >
          std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(sizes[i]))) return false;
      *value *= static_cast<uint64_t>(sizes[i]);
    }
    return true;
  };
  uint64_t input_count = 0;
  uint64_t output_count = 0;
  return input.dtype == meta.dtype && count(input.sizes, input.rank, &input_count) &&
         count(meta.sizes, meta.rank, &output_count) && input_count == output_count;
}

// Half-open device byte range. python_call has no schema, so an output that
// reuses an input is only visible here, after run_deferred exports the result
// pointer. Adopting that pointer would free the input's buffer twice.
struct ByteRange {
  uintptr_t begin = 0;
  uintptr_t end = 0;
};

void CollectTensorByteRanges(const FxTorchValue &value, std::vector<ByteRange> *ranges) {
  if (ranges == nullptr) {
    return;
  }
  if (value.kind == FX_TORCH_VALUE_TENSOR) {
    if (value.tensor.data == nullptr || value.tensor.nbytes == 0U) {
      return;
    }
    const auto begin = reinterpret_cast<uintptr_t>(value.tensor.data);
    if (value.tensor.nbytes > std::numeric_limits<uintptr_t>::max() - begin) {
      return;
    }
    ranges->push_back(ByteRange{begin, begin + static_cast<uintptr_t>(value.tensor.nbytes)});
    return;
  }
  if (value.kind != FX_TORCH_VALUE_LIST || value.elements == nullptr) {
    return;
  }
  for (size_t i = 0; i < value.element_count; ++i) {
    CollectTensorByteRanges(value.elements[i], ranges);
  }
}

bool ByteRangesOverlap(const ByteRange &left, const ByteRange &right) {
  return left.begin < right.end && right.begin < left.end;
}

bool AdoptedStorageAliasesInput(void *addr, size_t nbytes, const std::vector<ByteRange> &inputs) {
  if (addr == nullptr || nbytes == 0U || inputs.empty()) {
    return false;
  }
  const auto begin = reinterpret_cast<uintptr_t>(addr);
  if (nbytes > std::numeric_limits<uintptr_t>::max() - begin) {
    return false;
  }
  const ByteRange output{begin, begin + static_cast<uintptr_t>(nbytes)};
  return std::any_of(inputs.begin(), inputs.end(),
                     [&output](const ByteRange &input) { return ByteRangesOverlap(output, input); });
}

bool BindMallocedOutput(gert::EagerOpExecutionContext *ctx, size_t index,
                        const gert::StorageShape &storage_shape, const gert::StorageFormat &format,
                        ge::DataType dtype, std::vector<int64_t> *sizes,
                        std::vector<int64_t> *strides, FxTorchTensorView *output);

bool BindAdoptedOutput(gert::EagerOpExecutionContext *ctx, size_t index,
                       const gert::StorageShape &storage_shape, const gert::StorageFormat &format,
                       ge::DataType dtype, void *addr, size_t nbytes, std::vector<int64_t> *sizes,
                       std::vector<int64_t> *strides, FxTorchTensorView *output) {
  // Torch empty({0}) has a null data pointer. GE 9.2 and ExecuteAlloc/wrap_ge_outputs
  // require a non-null sentinel when this slot is a model output (indexer values).
  if (nbytes == 0U) {
    return BindMallocedOutput(ctx, index, storage_shape, format, dtype, sizes, strides, output);
  }
  auto *tensor = const_cast<gert::Tensor *>(ctx->GetOutputTensor(index));
  if (tensor == nullptr) {
    return false;
  }
  tensor->MutableOriginShape() = storage_shape.GetOriginShape();
  tensor->MutableStorageShape() = storage_shape.GetStorageShape();
  tensor->SetOriginFormat(format.GetOriginFormat());
  tensor->SetStorageFormat(format.GetStorageFormat());
  tensor->SetDataType(dtype);
  tensor->SetPlacement(gert::kOnDeviceHbm);
  gert::TensorAddrManager mgr = nullptr;
  void *block = tensor->MutableTensorData().Release(mgr);
  if (mgr != nullptr && block != nullptr) {
    (void)mgr(block, gert::kFreeTensor, nullptr);
  }
  if (addr == nullptr) {
    return false;
  }
  tensor->MutableTensorData().SetAddr(addr, nullptr);
  tensor->SetSize(nbytes);
  tensor->SetPlacement(gert::kOnDeviceHbm);
  return FillEagerTensor(tensor, static_cast<int32_t>(dtype), sizes, strides, output);
}

bool BindMallocedOutput(gert::EagerOpExecutionContext *ctx, size_t index,
                        const gert::StorageShape &storage_shape, const gert::StorageFormat &format,
                        ge::DataType dtype, std::vector<int64_t> *sizes,
                        std::vector<int64_t> *strides, FxTorchTensorView *output) {
  if (!ReleaseEagerOutputTensorData(ctx, index)) {
    RT_GLOG(ERROR) << "TorchFallbackEagerOp: failed to reset previous output storage " << index;
    return false;
  }
  auto *tensor = MallocEagerOutputTensor(ctx, index, storage_shape, format, dtype);
  if (tensor == nullptr) {
    return false;
  }
  if (!FillEagerTensor(tensor, static_cast<int32_t>(dtype), sizes, strides, output)) {
    RT_GLOG(ERROR) << "TorchFallbackEagerOp: FillEagerTensor failed output " << index
                   << " dtype=" << static_cast<int32_t>(dtype) << " rank=" << tensor->GetOriginShape().GetDimNum()
                   << " addr=" << tensor->GetAddr() << " nbytes=" << tensor->GetSize();
    return false;
  }
  tensor->SetPlacement(gert::kOnDeviceHbm);
  return true;
}

bool ShapeFromDeferredMeta(const FxTorchFallbackResultMetadata &meta, int32_t expected_dtype,
                           gert::Shape *shape) {
  if (meta.rank > kFallbackMaxRank || meta.dtype != expected_dtype) {
    return false;
  }
  shape->SetDimNum(meta.rank);
  for (size_t j = 0; j < meta.rank; ++j) {
    if (meta.sizes[j] < 0) {
      return false;
    }
    shape->SetDim(j, meta.sizes[j]);
  }
  return true;
}

bool IsWholePythonInputAlias(const FxTorchValue &value, const void *addr,
                             const FxTorchFallbackResultMetadata &meta) {
  if (value.kind == FX_TORCH_VALUE_TENSOR) {
    return value.tensor.data == addr && value.tensor.dtype == meta.dtype && SameLogicalShape(value.tensor, meta);
  }
  if (value.kind != FX_TORCH_VALUE_LIST || value.elements == nullptr) {
    return false;
  }
  for (size_t i = 0; i < value.element_count; ++i) {
    if (IsWholePythonInputAlias(value.elements[i], addr, meta)) {
      return true;
    }
  }
  return false;
}

bool IsWholePythonInputAlias(const std::vector<FxTorchValue> &arguments, const void *addr,
                             const FxTorchFallbackResultMetadata &meta) {
  return std::any_of(arguments.begin(), arguments.end(), [&](const FxTorchValue &argument) {
    return IsWholePythonInputAlias(argument, addr, meta);
  });
}

bool CopyPythonAliasOutput(gert::EagerOpExecutionContext *ctx, size_t index,
                           const gert::StorageShape &storage_shape, const gert::StorageFormat &format,
                           ge::DataType dtype, void *src, size_t nbytes, std::vector<int64_t> *sizes,
                           std::vector<int64_t> *strides, FxTorchTensorView *output) {
  if (!BindMallocedOutput(ctx, index, storage_shape, format, dtype, sizes, strides, output)) {
    return false;
  }
  auto *tensor = const_cast<gert::Tensor *>(ctx->GetOutputTensor(index));
  if (tensor == nullptr || tensor->GetAddr() == nullptr || src == nullptr) {
    return false;
  }
  const aclError ret = aclrtMemcpyAsync(tensor->GetAddr(), tensor->GetSize(), src, nbytes,
                                        ACL_MEMCPY_DEVICE_TO_DEVICE,
                                        reinterpret_cast<aclrtStream>(ctx->GetStream()));
  if (ret != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "TorchFallbackEagerOp: python call alias copy failed, ret="
                   << static_cast<int>(ret);
    return false;
  }
  return true;
}

struct DeferredExport {
  std::vector<void *> addrs;
  std::vector<size_t> nbytes;
  std::vector<int32_t> formats;
  std::vector<int64_t> storage_ranks;
  std::vector<int64_t> storage;
  std::vector<int64_t> stride_ranks;
  std::vector<int64_t> strides;
  std::vector<int64_t> stride_offsets;
  bool has_layout_metadata = false;
  bool has_stride_metadata = false;
};

struct DeferredResultGuard {
  const FxTorchFallbackApi *api;
  uintptr_t handle;
  // Move the handle to the caller and leave zero, so the destructor does not
  // destroy a result the session now owns.
  uintptr_t Release() {
    uintptr_t kept = 0;
    std::swap(kept, handle);
    return kept;
  }
  ~DeferredResultGuard() {
    if (api != nullptr && handle != 0) {
      api->destroy_result(handle);
    }
  }
};

ge::graphStatus ExportDeferredPayload(TorchFallbackCall *call, uintptr_t handle, size_t output_count,
                                      FxTorchError *error, const char *message, DeferredExport *out) {
  if (call->api->export_storage == nullptr) {
    RT_GLOG(ERROR) << "TorchFallbackEagerOp: fallback ABI missing export_storage";
    return ge::GRAPH_FAILED;
  }
  constexpr size_t kStorageStride = kFallbackMaxStorageRank;
  out->addrs.assign(output_count, nullptr);
  out->nbytes.assign(output_count, 0U);
  out->formats.assign(output_count, static_cast<int32_t>(ge::FORMAT_ND));
  out->storage_ranks.assign(output_count, 0);
  out->storage.assign(output_count * kStorageStride, 0);
  if (call->api->export_storage(handle, out->addrs.data(), out->nbytes.data(), output_count, error) != 0) {
    RT_GLOG(ERROR) << "TorchFallbackEagerOp: export_storage " << call->qualified_name << " failed: " << message;
    return ge::GRAPH_FAILED;
  }
  if (HasFallbackExportLayout(call)) {
    if (call->api->export_layout(handle, out->formats.data(), out->storage_ranks.data(), out->storage.data(),
                                 kStorageStride, output_count, error) != 0) {
      RT_GLOG(ERROR) << "TorchFallbackEagerOp: export_layout " << call->qualified_name << " failed: " << message;
      return ge::GRAPH_FAILED;
    }
    out->has_layout_metadata = true;
  }
  constexpr size_t kStrideCap = kFallbackMaxRank;
  out->stride_ranks.assign(output_count, -1);
  out->strides.assign(output_count * kStrideCap, 0);
  out->stride_offsets.assign(output_count, 0);
  if (HasFallbackExportStrides(call)) {
    if (call->api->export_strides(handle, out->stride_ranks.data(), out->strides.data(),
                                  out->stride_offsets.data(), kStrideCap, output_count, error) != 0) {
      RT_GLOG(ERROR) << "TorchFallbackEagerOp: export_strides " << call->qualified_name << " failed: " << message;
      return ge::GRAPH_FAILED;
    }
    out->has_stride_metadata = true;
  }
  return ge::GRAPH_SUCCESS;
}

bool OutputUsesRefInput(const TorchFallbackCall *call, const TorchFallbackRuntimeCache *cache, size_t index,
                        void *addr, const FxTorchFallbackResultMetadata &meta, const DeferredExport &exported,
                        size_t *ref_slot) {
  *ref_slot = RefInputSlotForOutput(call, index);
  const FxTorchValue *ref_input = RefInputValue(call, cache, index);
  bool same_storage = false;
  if (ref_input != nullptr && ref_input->kind == FX_TORCH_VALUE_TENSOR && addr != nullptr) {
    uintptr_t expected_addr = 0;
    const bool address_matches = LogicalDataAddress(ref_input->tensor, &expected_addr) &&
                                 reinterpret_cast<uintptr_t>(addr) == expected_addr;
    const bool read_alias = IsReadAliasOutput(*call, index);
    const bool offset_matches = !read_alias ||
                                (exported.has_stride_metadata && index < exported.stride_offsets.size() &&
                                 exported.stride_offsets[index] == ref_input->tensor.storage_offset);
    const bool layout_matches = read_alias ? SameDenseElementCount(ref_input->tensor, meta) && offset_matches
                                           : SameLogicalShape(ref_input->tensor, meta);
    same_storage = address_matches && layout_matches;
  }
  return *ref_slot != kNoTorchFallbackTensor && (addr == nullptr || same_storage);
}

void CopyRankPrefix(const int64_t *row, int64_t rank, size_t cap, std::vector<int64_t> *out) {
  if (rank > 0 && static_cast<size_t>(rank) <= cap && row != nullptr) {
    out->assign(row, row + rank);
  }
}

void RecordDeferredOutputLayout(FallbackSessionState *session, void *addr, int32_t format,
                                const DeferredExport &exported, size_t index,
                                const FxTorchFallbackResultMetadata &meta) {
  if (!exported.has_layout_metadata && !exported.has_stride_metadata) {
    return;
  }
  std::vector<int64_t> storage;
  const auto storage_row =
      exported.storage.begin() + static_cast<std::ptrdiff_t>(index * kFallbackMaxStorageRank);
  CopyRankPrefix(&*storage_row, exported.storage_ranks[index], kFallbackMaxStorageRank, &storage);
  std::vector<int64_t> stride;
  const auto stride_row = exported.strides.begin() + static_cast<std::ptrdiff_t>(index * kFallbackMaxRank);
  CopyRankPrefix(&*stride_row, exported.stride_ranks[index], kFallbackMaxRank, &stride);
  std::vector<int64_t> logical;
  if (meta.rank > 0U && meta.rank <= kFallbackMaxRank) {
    logical.assign(meta.sizes, meta.sizes + meta.rank);
  }
  session->RecordOutputLayout(addr, format, std::move(storage), std::move(stride), std::move(logical));
}

bool DeferredOutputHasDenseStride(const DeferredExport &exported, size_t index,
                                  const FxTorchFallbackResultMetadata &meta) {
  if (!exported.has_stride_metadata) {
    return true;
  }
  if (index >= exported.stride_ranks.size() || index >= exported.stride_offsets.size() ||
      exported.stride_ranks[index] != static_cast<int64_t>(meta.rank)) {
    return false;
  }
  const size_t stride_capacity = kFallbackMaxRank;
  const auto *stride = exported.strides.data() + index * stride_capacity;
  int64_t expected = 1;
  for (size_t i = meta.rank; i > 0U; --i) {
    const size_t axis = i - 1U;
    const int64_t size = meta.sizes[axis];
    if (size < 0 || stride[axis] < 0 || (size > 1 && stride[axis] != expected)) {
      return false;
    }
    if (size > 0) {
      if (expected > std::numeric_limits<int64_t>::max() / size) {
        return false;
      }
      expected *= size;
    }
  }
  return true;
}

void LogDeferredBindFailure(const TorchFallbackCall *call, size_t index, const FxTorchFallbackResultMetadata &meta,
                            size_t nbytes, void *addr, size_t ref_slot) {
  std::ostringstream dims;
  for (size_t d = 0; d < meta.rank; ++d) {
    if (d != 0U) {
      dims << ",";
    }
    dims << meta.sizes[d];
  }
  RT_GLOG(ERROR) << "TorchFallbackEagerOp: failed to bind deferred output " << index
                 << " op=" << call->qualified_name << " dtype=" << meta.dtype << " shape=[" << dims.str()
                 << "] nbytes=" << nbytes << " addr=" << addr << " ref_slot=" << ref_slot;
}

bool BindDeferredOutput(TorchFallbackCall *call, TorchFallbackRuntimeCache *cache, gert::EagerOpExecutionContext *ctx,
                        size_t index, const FxTorchFallbackResultMetadata &meta, const DeferredExport &exported,
                        const std::vector<ByteRange> &input_ranges, bool copy_dynamic_ref, FxTorchTensorView *view) {
  gert::Shape shape;
  if (!ShapeFromDeferredMeta(meta, call->output_dtypes[index], &shape)) {
    RT_GLOG(ERROR) << "TorchFallbackEagerOp: deferred metadata mismatch for output " << index;
    return false;
  }
  gert::StorageShape storage_shape;
  storage_shape.MutableOriginShape() = shape;
  storage_shape.MutableStorageShape() = shape;
  gert::StorageFormat format;
  format.SetOriginFormat(ge::FORMAT_ND);
  format.SetStorageFormat(ge::FORMAT_ND);
  size_t ref_slot = kNoTorchFallbackTensor;
  void *addr = exported.addrs[index];
  if (OutputUsesRefInput(call, cache, index, addr, meta, exported, &ref_slot)) {
    auto *tensor = ctx->MakeOutputRefInput(index, ref_slot);
    if (tensor != nullptr && IsReadAliasOutput(*call, index)) {
      tensor->MutableStorageShape() = shape;
      tensor->MutableOriginShape() = shape;
    }
    RT_GLOG(INFO) << "TorchFallbackEagerOp: deferred MakeOutputRefInput index=" << index << " input=" << ref_slot;
    return tensor != nullptr && FillEagerTensor(tensor, call->output_dtypes[index], &cache->output_sizes[index],
                                                &cache->output_strides[index], view);
  }
  if (IsReadAliasOutput(*call, index)) {
    RT_GLOG(ERROR) << "TorchFallbackEagerOp: " << call->qualified_name << " output " << index
                   << " violates the declared dense read-alias contract";
    return false;
  }
  if (copy_dynamic_ref &&
      (index >= call->python_output_dense.size() || call->python_output_dense[index] != 1)) {
    RT_GLOG(ERROR) << "TorchFallbackEagerOp: " << call->qualified_name << " output " << index
                   << " is not proven dense with storage_offset=0 by the FX output metadata";
    return false;
  }
  if (ref_slot == kNoTorchFallbackTensor &&
      AdoptedStorageAliasesInput(addr, exported.nbytes[index], input_ranges)) {
    const bool fresh_dense = index < call->fresh_dense_outputs.size() && call->fresh_dense_outputs[index] == 1;
    // A whole-input alias is a plain view of one argument even when the schema
    // does not declare a ref pair (for example _npu_moe_token_unpermute returns
    // its permuted_tokens input unchanged for backward). Hand FXRT its own dense
    // copy so the output owns its storage instead of double-owning the input.
    if (IsWholePythonInputAlias(cache->arguments, addr, meta)) {
      return CopyPythonAliasOutput(ctx, index, storage_shape, format, static_cast<ge::DataType>(meta.dtype), addr,
                                   exported.nbytes[index], &cache->output_sizes[index], &cache->output_strides[index],
                                   view);
    }
    if (!copy_dynamic_ref && !fresh_dense) {
      RT_GLOG(ERROR) << "TorchFallbackEagerOp: " << call->qualified_name << " output " << index
                     << " shares storage with an input but declares no ref pairs";
      return false;
    }
    RT_GLOG(ERROR) << "TorchFallbackEagerOp: " << call->qualified_name << " output " << index
                   << " aliases only part of an input; the output contract cannot preserve this alias";
    return false;
  }
  if (!FallbackFeedIsBaseFormat(exported.formats[index])) {
    RT_GLOG(ERROR) << "TorchFallbackEagerOp: " << call->qualified_name << " output " << index
                   << " does not support non-base memory format";
    return false;
  }
  if (!DeferredOutputHasDenseStride(exported, index, meta)) {
    RT_GLOG(ERROR) << "TorchFallbackEagerOp: " << call->qualified_name << " output " << index
                   << " has a non-dense stride layout that the deferred output ABI cannot bind as TensorV1";
    return false;
  }
  const bool bound =
      BindAdoptedOutput(ctx, index, storage_shape, format, static_cast<ge::DataType>(meta.dtype), addr,
                        exported.nbytes[index], &cache->output_sizes[index], &cache->output_strides[index], view);
  if (bound && addr != nullptr && call->session != nullptr) {
    RecordDeferredOutputLayout(call->session.get(), addr, exported.formats[index], exported, index, meta);
  }
  if (!bound) {
    LogDeferredBindFailure(call, index, meta, exported.nbytes[index], addr, ref_slot);
  }
  return bound;
}

ge::graphStatus ExecuteDeferred(TorchFallbackCall *call, gert::EagerOpExecutionContext *ctx) {
  int32_t device_id = 0;
  if (!FetchDeviceId(call, &device_id)) {
    return ge::GRAPH_FAILED;
  }
  auto *cache = call->cache.get();
  const size_t output_count = call->output_dtypes.size();
  std::vector<FxTorchFallbackResultMetadata> metadata(output_count);
  uintptr_t result_handle = 0;
  char message[kErrorCapacity] = {};
  FxTorchError error{message, sizeof(message)};
  const int32_t run_result = call->api->run_deferred(
      call->plan, cache->arguments.data(), cache->arguments.size(),
      reinterpret_cast<uintptr_t>(ctx->GetStream()), device_id, &result_handle, metadata.data(),
      metadata.size(), &error);
  if (run_result != 0 || result_handle == 0) {
    RT_GLOG(ERROR) << "TorchFallbackEagerOp: deferred " << call->qualified_name
                   << " failed: " << message;
    if (result_handle != 0) {
      call->api->destroy_result(result_handle);
    }
    return ge::GRAPH_FAILED;
  }
  DeferredResultGuard guard{call->api, result_handle};
  DeferredExport exported;
  if (ExportDeferredPayload(call, result_handle, output_count, &error, message, &exported) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  // Undeclared input aliases need a proven whole-input copy contract; schema
  // aliases retain their input allocation instead of being adopted as owners.
  std::vector<ByteRange> input_ranges;
  const bool copy_dynamic_ref = call->dynamic_ref_policy == FallbackDynamicRefPolicy::kCopyIfStandard;
  for (const auto &argument : cache->arguments) {
    CollectTensorByteRanges(argument, &input_ranges);
  }
  std::vector<FxTorchTensorView> outputs(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    if (!BindDeferredOutput(call, cache, ctx, i, metadata[i], exported, input_ranges, copy_dynamic_ref,
                            &outputs[i])) {
      return ge::GRAPH_FAILED;
    }
  }
  const auto result_owner = call->session->KeepAdoptedResult(call->api, guard.Release());
  for (const void *addr : exported.addrs) {
    call->session->RecordOutputOwner(addr, result_owner);
  }
  return ge::GRAPH_SUCCESS;
}

void OutputStorageFromTemplate(gert::EagerOpExecutionContext *ctx, size_t index,
                               const std::vector<int64_t> &fallback_dims,
                               gert::StorageShape *storage_shape, gert::StorageFormat *format) {
  const auto *template_tensor = ctx->GetOutputTensor(index);
  if (template_tensor != nullptr) {
    storage_shape->MutableOriginShape() = template_tensor->GetOriginShape();
    storage_shape->MutableStorageShape() = template_tensor->GetStorageShape();
    format->SetOriginFormat(template_tensor->GetOriginFormat());
    format->SetStorageFormat(template_tensor->GetStorageFormat());
    return;
  }
  gert::Shape shape;
  shape.SetDimNum(fallback_dims.size());
  for (size_t j = 0; j < fallback_dims.size(); ++j) {
    shape.SetDim(j, fallback_dims[j]);
  }
  storage_shape->MutableOriginShape() = shape;
  storage_shape->MutableStorageShape() = shape;
  format->SetStorageFormat(ge::FORMAT_ND);
  format->SetOriginFormat(ge::FORMAT_ND);
}

ge::graphStatus ExecuteWithKnownShape(TorchFallbackCall *call, gert::EagerOpExecutionContext *ctx) {
  auto *cache = call->cache.get();
  const size_t output_count = call->output_dtypes.size();
  std::vector<FxTorchTensorView> outputs(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    // ExecuteCustomOpWithInferShape copies the runtime shape from the
    // template Tensor input before invoking this adapter.  Use that shape
    // here; call->output_shapes only contains compile-time metadata and may
    // intentionally contain symbolic -1 dimensions.
    gert::StorageShape storage_shape;
    gert::StorageFormat format;
    OutputStorageFromTemplate(ctx, i, call->output_shapes[i], &storage_shape, &format);
    const size_t ref_slot = RefInputSlotForOutput(call, i);
    bool bound = false;
    if (ref_slot != kNoTorchFallbackTensor) {
      auto *tensor = ctx->MakeOutputRefInput(i, ref_slot);
      RT_GLOG(INFO) << "TorchFallbackEagerOp: MakeOutputRefInput index=" << i
                    << " input=" << ref_slot;
      bound = tensor != nullptr &&
              FillEagerTensor(tensor, call->output_dtypes[i], &cache->output_sizes[i],
                              &cache->output_strides[i], &outputs[i]);
    } else {
      bound = BindMallocedOutput(ctx, i, storage_shape, format,
                                 static_cast<ge::DataType>(call->output_dtypes[i]),
                                 &cache->output_sizes[i], &cache->output_strides[i], &outputs[i]);
    }
    if (!bound) {
      RT_GLOG(ERROR) << "TorchFallbackEagerOp: failed to allocate output " << i;
      return ge::GRAPH_FAILED;
    }
  }
  int32_t device_id = 0;
  if (!FetchDeviceId(call, &device_id)) {
    return ge::GRAPH_FAILED;
  }
  char message[kErrorCapacity] = {};
  FxTorchError error{message, sizeof(message)};
  const int32_t result = call->api->run_into(
      call->plan, cache->arguments.data(), cache->arguments.size(), outputs.data(), outputs.size(),
      reinterpret_cast<uintptr_t>(ctx->GetStream()), device_id, &error);
  if (result != 0) {
    RT_GLOG(ERROR) << "TorchFallbackEagerOp: " << call->qualified_name << " failed: " << message;
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0; i < output_count; ++i) {
    auto *tensor = const_cast<gert::Tensor *>(ctx->GetOutputTensor(i));
    if (tensor != nullptr) {
      tensor->SetPlacement(gert::kOnDeviceHbm);
    }
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

ge::graphStatus RunTorchFallback(TorchFallbackCall *call, gert::EagerOpExecutionContext *ctx) {
  if (ctx == nullptr || call == nullptr || call->api == nullptr || call->plan == nullptr ||
      call->cache == nullptr || call->session == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto &feeds = call->session->Feeds();
  const bool private_feed = std::any_of(feeds.begin(), feeds.end(), [](const FallbackFeed &feed) {
    return !FallbackFeedIsBaseFormat(feed.format);
  });
  if (private_feed && !HasFallbackExportLayout(call)) {
    RT_GLOG(ERROR) << "TorchFallbackEagerOp: cannot validate the result format for a private-format input; "
                      "the fallback bridge does not export layout metadata";
    return ge::GRAPH_FAILED;
  }
  if (!RefreshAllArguments(call, ctx)) {
    RT_GLOG(ERROR) << "TorchFallbackEagerOp: failed to materialize arguments for "
                   << call->qualified_name;
    return ge::GRAPH_FAILED;
  }
  // Private-format feeds cannot be written into the preallocated ND out shell.
  // Adopt the functional result, then validate its exported format. Base-format
  // out schemas write the GE buffer directly.
  const bool adopt_torch_result = call->deferred_shape || private_feed;
  return adopt_torch_result ? ExecuteDeferred(call, ctx) : ExecuteWithKnownShape(call, ctx);
}

bool NormalizeArgumentTree(TorchFallbackArgument *argument, FxSymbolEnv *symbol_env, std::string *error) {
  if (argument == nullptr) {
    return false;
  }
  if (argument->tensor_index != kNoTorchFallbackTensor) {
    return true;
  }
  if (!argument->elements.empty()) {
    return std::all_of(argument->elements.begin(), argument->elements.end(),
                       [&](TorchFallbackArgument &element) {
                         return NormalizeArgumentTree(&element, symbol_env, error);
                       });
  }
  argument->value = NormalizeValue(argument->value, symbol_env);
  if (argument->value == nullptr) {
    if (error != nullptr) {
      *error = "torch fallback: failed to parse symbolic argument";
    }
    return false;
  }
  if (argument->value->IsSymbol()) {
    argument->shape_expr = symbol_env == nullptr ? nullptr :
        BuildFallbackShapeExpr(argument->value->ToSymbol(), *symbol_env);
    if (argument->shape_expr == nullptr) {
      if (error != nullptr) {
        *error = "torch fallback: symbolic argument cannot be resolved from input shape tensors";
      }
      return false;
    }
  }
  return true;
}

size_t ShapeInputCount(const FxSymbolEnv *symbol_env) {
  if (symbol_env == nullptr) {
    return 0U;
  }
  size_t count = 0U;
  for (const auto &seed : symbol_env->seeds) {
    if (seed.feed_index >= 0) {
      count = std::max(count, static_cast<size_t>(seed.feed_index) + 1U);
    }
  }
  return count;
}

TorchFallbackCall::~TorchFallbackCall() {
  if (api != nullptr && plan != nullptr) {
    api->destroy_plan(plan);
  }
}

bool FallbackFeedIsBaseFormat(int32_t storage_format) {
  return storage_format == static_cast<int32_t>(ge::FORMAT_NCHW) ||
         storage_format == static_cast<int32_t>(ge::FORMAT_NHWC) ||
         storage_format == static_cast<int32_t>(ge::FORMAT_ND) ||
         storage_format == static_cast<int32_t>(ge::FORMAT_NCDHW);
}

bool FallbackStrideIsContiguous(const int64_t *shape, const int64_t *stride, size_t rank) {
  if (rank == 0U || stride == nullptr || shape == nullptr) {
    return true;
  }
  int64_t expect = 1;
  for (size_t i = rank; i > 0U; --i) {
    const int64_t dim = shape[i - 1U];
    if (dim == 0) {
      return true;
    }
    if (dim != 1 && stride[i - 1U] != expect) {
      return false;
    }
    if (dim > 1 && expect > std::numeric_limits<int64_t>::max() / dim) {
      return false;
    }
    if (dim > 0) {
      expect *= dim;
    }
  }
  return true;
}

const char *FallbackNoncontiguousPrivateReason() {
  static const char kReason[] =
      "torch fallback: non-contiguous input with non-base format is not supported";
  return kReason;
}

bool TorchFallbackPlanUsesOutSchema(const TorchFallbackCall &call) {
  return call.api != nullptr &&
         HasStructField(call.api->struct_size, offsetof(FxTorchFallbackApi, uses_out_schema),
                        sizeof(call.api->uses_out_schema)) &&
         call.api->uses_out_schema != nullptr && call.plan != nullptr &&
         call.api->uses_out_schema(call.plan) != 0;
}

bool RegisterTorchFallbackApi(uintptr_t api_address, std::string *error) {
  const auto *api = reinterpret_cast<const FxTorchFallbackApi *>(api_address);
  if (api == nullptr || api->abi_version != FX_TORCH_FALLBACK_ABI_VERSION ||
      api->struct_size < kFallbackApiBaseSize || api->create_plan == nullptr ||
      api->run_into == nullptr || api->run_deferred == nullptr || api->materialize == nullptr ||
      api->destroy_result == nullptr || api->destroy_plan == nullptr ||
      api->export_storage == nullptr) {
    if (error != nullptr) {
      *error = "torch fallback bridge reports an incompatible ABI";
    }
    return false;
  }
  g_torch_fallback_api.store(api, std::memory_order_release);
  return true;
}

// Operator's Input/OutputRegister are protected (the REG_OP subclass
// pattern); a local derived class is the sanctioned way to declare a proto
// programmatically.
class FallbackOpProto final : public ge::Operator {
 public:
  FallbackOpProto(const std::string &type, const std::vector<std::string> &inputs,
                  const std::vector<std::string> &outputs)
      : ge::Operator(type.c_str()) {
    for (const auto &name : inputs) {
      InputRegister(name.c_str());
    }
    for (const auto &name : outputs) {
      OutputRegister(name.c_str());
    }
  }
};

// OperatorFactory proto registrations for fallback op types. The eager op
// execution context resolves the CNI node type through OperatorFactory to
// obtain the op proto (input/output names) used by output-ref-input reuse
// marking, so every fallback op type must be registered. Ref-paired outputs
// reuse their paired input's name so MakeOutputRefInput's same-name check
// passes; other outputs keep collision-free positional names.
std::mutex g_fallback_proto_mu;
std::vector<std::unique_ptr<ge::OperatorCreatorRegister>> g_fallback_proto_regs;

std::string RegisterFallbackOpProto(const TorchFallbackCall &call) {
  std::vector<std::string> input_names;
  input_names.reserve(call.tensor_count + call.shape_input_count);
  for (size_t i = 0; i < call.tensor_count; ++i) {
    input_names.push_back("arg_" + std::to_string(i));
  }
  for (size_t i = 0; i < call.shape_input_count; ++i) {
    input_names.push_back("shape_" + std::to_string(i));
  }
  std::vector<std::string> output_names(call.output_dtypes.size());
  for (size_t j = 0; j < output_names.size(); ++j) {
    output_names[j] = "out_" + std::to_string(j);
  }
  for (const auto &pair : call.ref_pairs) {
    if (pair.second >= output_names.size() || pair.first >= call.arguments.size()) {
      continue;
    }
    const size_t slot = call.arguments[pair.first].tensor_index;
    if (slot == kNoTorchFallbackTensor || slot >= input_names.size()) {
      continue;
    }
    output_names[pair.second] = input_names[slot];
  }
  for (const auto &pair : call.read_alias_pairs) {
    if (pair.second >= output_names.size() || pair.first >= call.arguments.size()) continue;
    const size_t slot = call.arguments[pair.first].tensor_index;
    if (slot != kNoTorchFallbackTensor && slot < input_names.size()) {
      output_names[pair.second] = input_names[slot];
    }
  }
  // One dispatcher schema can return a fresh allocation or an input alias.
  // GE registrations are process-wide, so key the proto by its actual port
  // names instead of letting the first compiled graph fix later alias rules.
  std::string op_type = call.qualified_name + "__fxrt_" + std::to_string(call.tensor_count) + "_" +
                        std::to_string(call.shape_input_count);
  for (const auto &name : output_names) {
    op_type += "__" + name;
  }
  const std::lock_guard<std::mutex> lock(g_fallback_proto_mu);
  if (ge::OperatorFactory::IsExistOp(op_type.c_str())) {
    return op_type;
  }
  auto creator = [op_type, input_names, output_names](const ge::AscendString &) {
    return FallbackOpProto(op_type, input_names, output_names);
  };
  g_fallback_proto_regs.push_back(std::make_unique<ge::OperatorCreatorRegister>(op_type.c_str(), creator));
  return op_type;
}

void AssignFallbackTensorSlots(std::vector<TorchFallbackArgument> *arguments, size_t *tensor_count) {
  std::function<void(TorchFallbackArgument *)> assign_slots;
  assign_slots = [&](TorchFallbackArgument *argument) {
    if (argument->tensor_index != kNoTorchFallbackTensor) {
      argument->tensor_index = (*tensor_count)++;
      return;
    }
    for (auto &element : argument->elements) {
      assign_slots(&element);
    }
  };
  for (auto &argument : *arguments) {
    assign_slots(&argument);
  }
}

bool SpecForFallbackArgument(TorchFallbackArgument *argument, FxSymbolEnv *symbol_env, std::string *error,
                             FxTorchArgSpec *spec) {
  if (argument->tensor_index != kNoTorchFallbackTensor) {
    *spec = {FX_TORCH_VALUE_TENSOR, FX_TORCH_VALUE_NONE};
    return true;
  }
  if (!argument->elements.empty()) {
    const bool has_tensor_leaf =
        std::any_of(argument->elements.begin(), argument->elements.end(), [](const TorchFallbackArgument &element) {
          return element.tensor_index != kNoTorchFallbackTensor;
        });
    if (has_tensor_leaf) {
      *spec = {FX_TORCH_VALUE_LIST, FX_TORCH_VALUE_TENSOR};
      return true;
    }
  }
  if (!NormalizeArgumentTree(argument, symbol_env, error)) {
    return false;
  }
  uint32_t element_kind = FX_TORCH_VALUE_NONE;
  const uint32_t kind = ValueKind(argument->value, &element_kind, error);
  if (kind == std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  *spec = {kind, element_kind};
  return true;
}

bool BuildFallbackArgumentSpecs(std::vector<TorchFallbackArgument> *arguments, FxSymbolEnv *symbol_env,
                                std::string *error, std::vector<FxTorchArgSpec> *specs, size_t *tensor_count) {
  // Tensor leaves inside a Tensor[] argument each own a GE slot, so count them
  // all, not only top-level tensors.
  *tensor_count = 0U;
  AssignFallbackTensorSlots(arguments, tensor_count);
  specs->assign(arguments->size(), FxTorchArgSpec{});
  for (size_t i = 0; i < arguments->size(); ++i) {
    if (!SpecForFallbackArgument(&(*arguments)[i], symbol_env, error, &(*specs)[i])) {
      return false;
    }
  }
  return true;
}

void CollectFallbackRefPairs(const FxTorchFallbackApi *api, void *plan, TorchFallbackCall *call) {
  if (!HasStructField(api->struct_size, offsetof(FxTorchFallbackApi, ref_pairs), sizeof(api->ref_pairs)) ||
      api->ref_pairs == nullptr) {
    return;
  }
  const size_t pair_count = api->ref_pairs(plan, nullptr, nullptr, 0U);
  if (pair_count == 0U) {
    return;
  }
  std::vector<size_t> arg_indices(pair_count);
  std::vector<size_t> output_indices(pair_count);
  const size_t filled = api->ref_pairs(plan, arg_indices.data(), output_indices.data(), pair_count);
  call->ref_pairs.reserve(filled);
  for (size_t i = 0; i < filled; ++i) {
    // New bridges report schema may-alias pairs as well as write refs. A
    // conditional alias (for example to.dtype) can produce fresh storage;
    // treating it as a write ref also overwrites the output dtype/layout.
    const auto pair = std::make_pair(arg_indices[i], output_indices[i]);
    const bool fresh_dense = pair.second < call->fresh_dense_outputs.size() &&
                             call->fresh_dense_outputs[pair.second] == 1;
    const bool read_alias = std::find(call->read_alias_pairs.begin(), call->read_alias_pairs.end(), pair) !=
                            call->read_alias_pairs.end();
    if (fresh_dense || read_alias) continue;
    call->ref_pairs.emplace_back(arg_indices[i], output_indices[i]);
  }
}

std::shared_ptr<TorchFallbackCall> CreateTorchFallbackCall(
    const std::string &qualified_name, const std::string &overload_name,
    std::vector<TorchFallbackArgument> arguments, FxSymbolEnv *symbol_env,
    std::vector<int32_t> output_dtypes, std::vector<std::pair<size_t, size_t>> read_alias_pairs,
    std::vector<int8_t> fresh_dense_outputs, std::shared_ptr<FallbackSessionState> session,
    std::string *error) {
  const auto *api = g_torch_fallback_api.load(std::memory_order_acquire);
  if (api == nullptr) {
    if (error != nullptr) {
      *error = "torch fallback bridge is not registered";
    }
    return nullptr;
  }
  std::vector<FxTorchArgSpec> specs;
  size_t tensor_count = 0U;
  if (!BuildFallbackArgumentSpecs(&arguments, symbol_env, error, &specs, &tensor_count)) {
    return nullptr;
  }
  if (output_dtypes.empty()) {
    if (error != nullptr) {
      *error = "torch fallback: output list is empty";
    }
    return nullptr;
  }
  FxTorchOpSpec spec{{qualified_name.data(), qualified_name.size()},
                     {overload_name.data(), overload_name.size()}, specs.data(), specs.size(),
                     output_dtypes.size()};
  char message[kErrorCapacity] = {};
  FxTorchError bridge_error{message, sizeof(message)};
  void *plan = api->create_plan(&spec, &bridge_error);
  if (plan == nullptr) {
    if (error != nullptr) {
      *error = std::string("torch fallback plan creation failed: ") + message;
    }
    return nullptr;
  }
  auto call = std::make_shared<TorchFallbackCall>();
  call->api = api;
  call->plan = plan;
  call->qualified_name = qualified_name;
  call->overload_name = overload_name;
  call->arguments = std::move(arguments);
  call->tensor_count = tensor_count;
  call->shape_input_count = ShapeInputCount(symbol_env);
  call->symbol_env = symbol_env;
  call->output_dtypes = std::move(output_dtypes);
  call->read_alias_pairs = std::move(read_alias_pairs);
  call->fresh_dense_outputs = std::move(fresh_dense_outputs);
  call->dynamic_ref_policy = FallbackDynamicRefPolicy::kReject;
  call->session = std::move(session);
  CollectFallbackRefPairs(api, plan, call.get());
  call->proto_type = RegisterFallbackOpProto(*call);
  if (!PrepareRuntimeCache(call.get(), error)) {
    return nullptr;
  }
  return call;
}

bool RegisterPythonCallApi(uintptr_t api_address, std::string *error) {
  // python_call publishes the same V3 table as torch fallback.
  const auto *api = reinterpret_cast<const FxTorchFallbackApi *>(api_address);
  if (api == nullptr || api->abi_version != FX_TORCH_FALLBACK_ABI_VERSION ||
      api->struct_size < sizeof(FxTorchFallbackApi) || api->create_plan == nullptr ||
      api->run_into == nullptr || api->run_deferred == nullptr || api->materialize == nullptr ||
      api->destroy_result == nullptr || api->destroy_plan == nullptr ||
      api->export_storage == nullptr) {
    if (error != nullptr) {
      *error = "python call bridge reports an incompatible ABI";
    }
    return false;
  }
  g_python_call_api.store(api, std::memory_order_release);
  return true;
}

std::shared_ptr<TorchFallbackCall> CreatePythonCall(
    const std::string &module_name, const std::string &func_name,
    std::vector<TorchFallbackArgument> arguments, FxSymbolEnv *symbol_env,
    std::vector<int32_t> output_dtypes, std::shared_ptr<FallbackSessionState> session,
    std::string *error) {
  const auto *api = g_python_call_api.load(std::memory_order_acquire);
  if (api == nullptr) {
    if (error != nullptr) {
      *error = "python call bridge is not registered";
    }
    return nullptr;
  }
  std::vector<FxTorchArgSpec> specs;
  size_t tensor_count = 0U;
  if (!BuildFallbackArgumentSpecs(&arguments, symbol_env, error, &specs, &tensor_count)) {
    return nullptr;
  }
  if (output_dtypes.empty()) {
    if (error != nullptr) {
      *error = "python call: output list is empty";
    }
    return nullptr;
  }
  // The internal adapter maps the operation-shaped spec to PythonCall's
  // separate module/function ABI.
  FxTorchOpSpec spec{{module_name.data(), module_name.size()},
                     {func_name.data(), func_name.size()},
                     specs.data(),
                     specs.size(),
                     output_dtypes.size()};
  char message[kErrorCapacity] = {};
  FxTorchError bridge_error{message, sizeof(message)};
  void *plan = api->create_plan(&spec, &bridge_error);
  if (plan == nullptr) {
    if (error != nullptr) {
      *error = std::string("python call plan creation failed: ") + message;
    }
    return nullptr;
  }
  auto call = std::make_shared<TorchFallbackCall>();
  call->api = api;
  call->plan = plan;
  call->qualified_name = module_name + "." + func_name;
  call->arguments = std::move(arguments);
  call->tensor_count = tensor_count;
  call->shape_input_count = ShapeInputCount(symbol_env);
  call->symbol_env = symbol_env;
  call->output_dtypes = std::move(output_dtypes);
  call->dynamic_ref_policy = FallbackDynamicRefPolicy::kCopyIfStandard;
  call->session = std::move(session);
  call->proto_type = RegisterFallbackOpProto(*call);
  if (!PrepareRuntimeCache(call.get(), error)) {
    return nullptr;
  }
  return call;
}

}  // namespace fx_converter
