// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * ABI=1 island: wraps GE-allocated device buffers as owning torch NPU tensors.
 *
 * Why a separate module: GE host libs force _GLIBCXX_USE_CXX11_ABI=0, torch/torch_npu
 * are ABI=1. Anything with std::string / std::function / IntArrayRef in its signature
 * cannot cross that line, so this file must not include a single GE header. The GE side
 * (_fxrt_fx_converter) hands over only integers: a device address, an opaque context
 * pointer and a `void(*)(void*)` deleter -- all ABI-neutral.
 *
 * The deleter returns the block to GE's L1 allocator and drops a strong reference to the
 * ModelV2Runner that owns that allocator, so the buffer can safely outlive the session.
 *
 * Layering note: this module IS the ABI boundary, mirrored on the converter side by
 * torch_fallback_abi_client.cc. Schema matching, stream joining and NZ metadata handling
 * below are the fallback business logic; the POD marshalling around them exists only
 * because of the ABI split. In a single-ABI build this module would be deleted and the
 * business logic would move into the converter.
 */

#include <torch/extension.h>

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/aten/common/from_blob.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_fallback/torch_fallback_abi.h"

namespace {

using FxFreeOwnedFn = void (*)(void *);

struct TorchFallbackPlan {
  enum class ReturnKind { kTensorValues, kTensorList };

  c10::OperatorHandle handle;
  std::string qualified_name;
  std::string overload_name;
  std::vector<c10::TypePtr> argument_types;
  std::vector<size_t> out_arg_indices;
  // (schema argument index, output index) for mutating (out-param) ops on
  // non-out overloads: a write-alias Tensor argument paired with the return
  // that aliases it (e.g. vllm.dsa_forward's "Tensor(a!) output" argument
  // and its "Tensor(a!)" return). Drives GE's output-ref-input reuse marking.
  // For mutation-only schemas (no returns) the pairs map each write-alias
  // argument to its synthetic output (see synthetic_outputs).
  std::vector<std::pair<size_t, size_t>> ref_pairs;
  // Mutation-only schema (returns nothing, e.g. "dsa_forward(...) -> ()"):
  // the declared outputs are the write-alias input arguments themselves (ref
  // semantics). The op writes the shared buffers through its inputs; results
  // extraction and writeback are skipped.
  bool synthetic_outputs = false;
  ReturnKind return_kind = ReturnKind::kTensorValues;
  size_t output_count = 0U;

  TorchFallbackPlan(c10::OperatorHandle selected_handle, std::string selected_name,
                    std::string selected_overload, std::vector<c10::TypePtr> selected_types,
                    std::vector<size_t> selected_out_arg_indices,
                    std::vector<std::pair<size_t, size_t>> selected_ref_pairs,
                    bool selected_synthetic, ReturnKind selected_return_kind,
                    size_t selected_output_count)
      : handle(std::move(selected_handle)),
        qualified_name(std::move(selected_name)),
        overload_name(std::move(selected_overload)),
        argument_types(std::move(selected_types)),
        out_arg_indices(std::move(selected_out_arg_indices)),
        ref_pairs(std::move(selected_ref_pairs)),
        synthetic_outputs(selected_synthetic),
        return_kind(selected_return_kind),
        output_count(selected_output_count) {}
};

void SetError(FxTorchError *error, const std::string &message) {
  if (error == nullptr || error->message == nullptr || error->capacity == 0U) {
    return;
  }
  const size_t size = std::min(message.size(), error->capacity - 1U);
  std::memcpy(error->message, message.data(), size);
  error->message[size] = '\0';
}

std::string QualifiedName(const FxTorchStringView &view) {
  return view.data == nullptr ? std::string() : std::string(view.data, view.size);
}

std::string RequestedOverload(const FxTorchOpSpec &spec) {
  return QualifiedName(spec.overload_name);
}

at::ScalarType MapDtype(int32_t dtype) {
  switch (dtype) {
    case 0: return at::kFloat;
    case 1: return at::kHalf;
    case 2: return at::kChar;
    case 3: return at::kInt;
    case 4: return at::kByte;
    case 6: return at::kShort;
    case 9: return at::kLong;
    case 11: return at::kDouble;
    case 12: return at::kBool;
    case 27: return at::kBFloat16;
    default: throw std::runtime_error("unsupported GE dtype " + std::to_string(dtype));
  }
}

int32_t ToGeDtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat: return 0;
    case at::kHalf: return 1;
    case at::kChar: return 2;
    case at::kInt: return 3;
    case at::kByte: return 4;
    case at::kShort: return 6;
    case at::kLong: return 9;
    case at::kDouble: return 11;
    case at::kBool: return 12;
    case at::kBFloat16: return 27;
    default: throw std::runtime_error("unsupported Torch dtype");
  }
}

bool IsTypeCompatible(const c10::TypePtr &type, uint32_t kind, uint32_t element_kind) {
  if (type == nullptr) {
    return false;
  }
  // JIT `Any` (e.g. _c10d_functional::all_to_all_single's group_name) imposes no
  // constraint on the argument; the boxed call carries whatever IValue we build.
  if (type->kind() == c10::TypeKind::AnyType) {
    return true;
  }
  if (type->kind() == c10::TypeKind::OptionalType) {
    if (kind == FX_TORCH_VALUE_NONE) {
      return true;
    }
    return IsTypeCompatible(type->cast<c10::OptionalType>()->getElementType(), kind, element_kind);
  }
  switch (kind) {
    case FX_TORCH_VALUE_TENSOR: return type->kind() == c10::TypeKind::TensorType;
    case FX_TORCH_VALUE_DOUBLE:
      return type->kind() == c10::TypeKind::FloatType || type->kind() == c10::TypeKind::NumberType;
    case FX_TORCH_VALUE_INT: {
      const auto type_kind = type->kind();
      if (type_kind == c10::TypeKind::IntType || type_kind == c10::TypeKind::NumberType ||
          type_kind == c10::TypeKind::BoolType || type_kind == c10::TypeKind::SymIntType) {
        return true;
      }
      const std::string type_name = type->str();
      return type_name == "ScalarType" || type_name == "Layout" || type_name == "MemoryFormat";
    }
    case FX_TORCH_VALUE_BOOL: return type->kind() == c10::TypeKind::BoolType;
    case FX_TORCH_VALUE_STRING:
      return type->kind() == c10::TypeKind::StringType || type->kind() == c10::TypeKind::DeviceObjType;
    case FX_TORCH_VALUE_LIST: {
      if (type->kind() != c10::TypeKind::ListType) {
        return false;
      }
      const auto elem_type = type->cast<c10::ListType>()->getElementType();
      return IsTypeCompatible(elem_type, element_kind, FX_TORCH_VALUE_NONE);
    }
    case FX_TORCH_VALUE_NONE: return type->kind() == c10::TypeKind::NoneType;
    default: return false;
  }
}

bool CollectOutArguments(const c10::FunctionSchema &schema, std::vector<size_t> *indices,
                         bool *is_out_overload) {
  // Only dispatcher out overloads are eligible for the preallocated-output
  // path.  Mutable self/in-place and view aliases remain rejected until their
  // GE alias/lifetime contract is implemented separately.
  const auto &overload = schema.overload_name();
  *is_out_overload = overload == "out" ||
                     (overload.size() >= 4U && overload.substr(overload.size() - 4U) == "_out");
  if (!*is_out_overload) {
    return true;
  }
  for (size_t i = 0; i < schema.arguments().size(); ++i) {
    const auto &argument = schema.arguments()[i];
    if (argument.alias_info() != nullptr && argument.alias_info()->isWrite()) {
      if (argument.real_type()->kind() != c10::TypeKind::TensorType) {
        return false;
      }
      indices->push_back(i);
    }
  }
  return !indices->empty();
}

bool IsOutArgument(size_t index, const std::vector<size_t> &out_arg_indices) {
  return std::find(out_arg_indices.begin(), out_arg_indices.end(), index) != out_arg_indices.end();
}

// wait_tensor is declared `Tensor -> Tensor` with no alias set, but the kernel
// returns that same tensor once the collective work has finished.
void AddWaitTensorRefPair(const std::string &dispatcher_name,
                          std::vector<std::pair<size_t, size_t>> *pairs) {
  if (pairs->empty() && dispatcher_name == "_c10d_functional::wait_tensor") {
    pairs->emplace_back(0U, 0U);
  }
}

// Pair Tensor arguments with the returns that share their alias symbol.
// Write aliases (Tensor(a!)) and view aliases (Tensor(a)) are both schema
// refs. A wildcard set does not name one input. out= overloads receive a
// preallocated destination and are not paired here.
void CollectAliasRefPairs(const c10::FunctionSchema &schema,
                          std::vector<std::pair<size_t, size_t>> *pairs) {
  pairs->clear();
  for (size_t i = 0; i < schema.arguments().size(); ++i) {
    const auto &argument = schema.arguments()[i];
    if (argument.real_type()->kind() != c10::TypeKind::TensorType) {
      continue;
    }
    const auto *arg_alias = argument.alias_info();
    if (arg_alias == nullptr) {
      continue;
    }
    const c10::Symbol arg_symbol = arg_alias->beforeSet();
    if (arg_symbol == c10::AliasInfo::wildcardSet()) {
      continue;
    }
    for (size_t r = 0; r < schema.returns().size(); ++r) {
      const auto *ret_alias = schema.returns()[r].alias_info();
      if (ret_alias == nullptr || ret_alias->beforeSet() == c10::AliasInfo::wildcardSet()) {
        continue;
      }
      if (ret_alias->beforeSet() == arg_symbol) {
        pairs->emplace_back(i, r);
        break;
      }
    }
  }
}

// Mutation-only schema (no returns): every write-alias Tensor argument IS a
// synthetic output, paired in declaration order.
void CollectSyntheticRefPairs(const c10::FunctionSchema &schema,
                              std::vector<std::pair<size_t, size_t>> *pairs) {
  pairs->clear();
  for (size_t i = 0; i < schema.arguments().size(); ++i) {
    const auto &argument = schema.arguments()[i];
    if (argument.real_type()->kind() != c10::TypeKind::TensorType) {
      continue;
    }
    const auto *arg_alias = argument.alias_info();
    if (arg_alias == nullptr || !arg_alias->isWrite()) {
      continue;
    }
    pairs->emplace_back(i, pairs->size());
  }
}

bool MatchSchema(const c10::FunctionSchema &schema, const FxTorchOpSpec &spec,
                 const std::vector<size_t> &out_arg_indices) {
  size_t spec_index = 0;
  for (size_t schema_index = 0; schema_index < schema.arguments().size(); ++schema_index) {
    if (IsOutArgument(schema_index, out_arg_indices)) {
      continue;
    }
    const auto &argument = schema.arguments()[schema_index];
    if (spec_index < spec.argument_count) {
      if (!IsTypeCompatible(argument.real_type(), spec.arguments[spec_index].kind,
                            spec.arguments[spec_index].element_kind)) {
        return false;
      }
      ++spec_index;
      continue;
    }
    if (!argument.default_value().has_value()) {
      return false;
    }
  }
  return spec_index == spec.argument_count;
}

bool MatchReturns(const c10::FunctionSchema &schema, size_t output_count,
                  TorchFallbackPlan::ReturnKind *return_kind) {
  if (schema.returns().size() == output_count &&
      std::all_of(schema.returns().begin(), schema.returns().end(), [](const auto &result) {
        return result.type()->kind() == c10::TypeKind::TensorType;
      })) {
    *return_kind = TorchFallbackPlan::ReturnKind::kTensorValues;
    return true;
  }
  if (schema.returns().size() != 1U || schema.returns()[0].type()->kind() != c10::TypeKind::ListType) {
    return false;
  }
  const auto element_type = schema.returns()[0].type()->cast<c10::ListType>()->getElementType();
  if (element_type->kind() != c10::TypeKind::TensorType) {
    return false;
  }
  *return_kind = TorchFallbackPlan::ReturnKind::kTensorList;
  return true;
}

std::shared_ptr<TorchFallbackPlan> SelectPlan(const FxTorchOpSpec &spec, FxTorchError *error) {
  const std::string exported_name = QualifiedName(spec.qualified_name);
  if (exported_name.empty()) {
    SetError(error, "empty qualified operator name");
    return nullptr;
  }
  std::string dispatcher_name = exported_name;
  const auto requested_overload = RequestedOverload(spec);
  const size_t dot = dispatcher_name.rfind('.');
  const std::string suffix = dot == std::string::npos ? std::string() : dispatcher_name.substr(dot + 1U);
  if (dot != std::string::npos &&
      (suffix == "default" || (!requested_overload.empty() && suffix == requested_overload))) {
    // The exporter uses either `namespace::op.overload` or the legacy
    // `namespace.op.overload` spelling.  Strip the overload first; only the
    // latter needs its namespace separator rewritten.
    dispatcher_name.erase(dot);
  }
  if (dispatcher_name.find("::") == std::string::npos) {
    const size_t ns_dot = dispatcher_name.find('.');
    if (ns_dot != std::string::npos) {
      dispatcher_name.replace(ns_dot, 1U, "::");
    }
  }
  std::vector<c10::OperatorHandle> dispatcher_handles;
  auto append_schema = [&dispatcher_handles, &dispatcher_name](const std::string &overload) {
    if (const auto found = c10::Dispatcher::singleton().findSchema({dispatcher_name, overload})) {
      dispatcher_handles.push_back(*found);
    }
  };
  if (requested_overload.empty() || requested_overload == "default") {
    // A functional call can use the dispatcher out overload when it has the
    // same logical inputs and a matching number of GE-owned outputs. Prefer
    // this path so the preallocated GE buffers are written directly; fall
    // back to the functional/default schema when no compatible out schema
    // exists (or when its signature does not match below). A private-format
    // feed does not use that out shell: RunDeferredPlan calls the functional
    // schema so the kernel can allocate a result in the same format.
    append_schema("out");
    append_schema("");
    // aten::to has no nameless default; the FX exporter often emits
    // aten::to / aten::to.default for the dtype overload. Collect named
    // overloads so dtype/device/other can match.
    if (dispatcher_name.find("::") != std::string::npos) {
      const auto symbol = torch::jit::Symbol::fromQualString(dispatcher_name);
      for (const auto &op : torch::jit::getAllOperatorsFor(symbol)) {
        append_schema(op->schema().overload_name());
      }
    }
  } else {
    append_schema(requested_overload);
  }
  if (dispatcher_handles.empty()) {
    if (dispatcher_name.find("::") == std::string::npos) {
      SetError(error, "invalid Torch qualified operator name " + dispatcher_name);
      return nullptr;
    }
    const auto symbol = torch::jit::Symbol::fromQualString(dispatcher_name);
    for (const auto &op : torch::jit::getAllOperatorsFor(symbol)) {
      try {
        dispatcher_handles.push_back(c10::Dispatcher::singleton().findSchemaOrThrow(
            dispatcher_name.c_str(), op->schema().overload_name().c_str()));
      } catch (const c10::Error &) {
        // JIT-only schemas are not callable through the runtime dispatcher.
      }
    }
  }
  if (dispatcher_handles.empty()) {
    SetError(error, "no Torch operator registered for " + dispatcher_name);
    return nullptr;
  }
  for (const auto &dispatcher_handle : dispatcher_handles) {
    const auto &schema = dispatcher_handle.schema();
    const std::string schema_overload = schema.overload_name().empty() ? "default" : schema.overload_name();
    std::vector<size_t> out_arg_indices;
    bool is_out_overload = false;
    if (!CollectOutArguments(schema, &out_arg_indices, &is_out_overload)) {
      continue;
    }
    const bool allow_default_out = requested_overload == "default" && is_out_overload;
    if (!requested_overload.empty() && schema_overload != requested_overload && !allow_default_out) {
      continue;
    }
    if (is_out_overload && out_arg_indices.size() != spec.output_count) {
      continue;
    }
    TorchFallbackPlan::ReturnKind return_kind = TorchFallbackPlan::ReturnKind::kTensorValues;
    bool synthetic_outputs = false;
    std::vector<std::pair<size_t, size_t>> ref_pairs;
    if (!is_out_overload && schema.returns().empty()) {
      // Mutation-only schema: outputs are the write-alias arguments (ref).
      CollectSyntheticRefPairs(schema, &ref_pairs);
      synthetic_outputs = !ref_pairs.empty();
      if (!synthetic_outputs || spec.output_count != ref_pairs.size()) {
        continue;
      }
    } else {
      if (!is_out_overload) {
        CollectAliasRefPairs(schema, &ref_pairs);
        AddWaitTensorRefPair(dispatcher_name, &ref_pairs);
      }
      if (!MatchReturns(schema, spec.output_count, &return_kind)) {
        continue;
      }
    }
    if (!MatchSchema(schema, spec, out_arg_indices)) {
      continue;
    }
    try {
      std::vector<c10::TypePtr> argument_types;
      argument_types.reserve(schema.arguments().size() - out_arg_indices.size());
      for (size_t i = 0; i < schema.arguments().size(); ++i) {
        if (!IsOutArgument(i, out_arg_indices)) {
          argument_types.push_back(schema.arguments()[i].real_type());
        }
      }
      return std::make_shared<TorchFallbackPlan>(dispatcher_handle, dispatcher_name, schema_overload,
                                                 std::move(argument_types), std::move(out_arg_indices),
                                                 std::move(ref_pairs), synthetic_outputs, return_kind,
                                                 spec.output_count);
    } catch (const c10::Error &e) {
      SetError(error, e.what_without_backtrace());
      return nullptr;
    }
  }
  std::ostringstream message;
  message << "no matching Torch schema for " << dispatcher_name;
  SetError(error, message.str());
  return nullptr;
}

std::vector<int64_t> InferNzStorageSizes(const std::vector<int64_t> &origin, int64_t itemsize) {
  std::vector<int64_t> dim = origin;
  if (dim.empty()) {
    dim.push_back(1);
  }
  if (dim.size() == 1U) {
    dim.push_back(1);
  }
  constexpr int64_t kNzCube = 16;
  constexpr int64_t kNzBlockBytes = 32;
  const int64_t item = itemsize > 2 ? 2 : (itemsize < 1 ? 1 : itemsize);
  const int64_t c0 = kNzBlockBytes / item;
  std::vector<int64_t> out;
  for (size_t i = 0; i + 2U < dim.size(); ++i) {
    out.push_back(dim[i]);
  }
  const int64_t h = dim[dim.size() - 2U];
  const int64_t w = dim[dim.size() - 1U];
  out.push_back((w + c0 - 1) / c0);
  out.push_back((h + kNzCube - 1) / kNzCube);
  out.push_back(kNzCube);
  out.push_back(c0);
  return out;
}

bool IsBaseAclFormat(uint32_t format) {
  return format == static_cast<uint32_t>(ACL_FORMAT_NCHW) || format == static_cast<uint32_t>(ACL_FORMAT_NHWC) ||
         format == static_cast<uint32_t>(ACL_FORMAT_ND) || format == static_cast<uint32_t>(ACL_FORMAT_NCDHW) ||
         format == 0U;
}

aclFormat BaseAclFormat(uint32_t format) {
  switch (format) {
    case ACL_FORMAT_NC1HWC0:
    case ACL_FORMAT_FRACTAL_Z:
    case ACL_FORMAT_NCHW:
      return ACL_FORMAT_NCHW;
    case ACL_FORMAT_NHWC:
      return ACL_FORMAT_NHWC;
    case ACL_FORMAT_NCDHW:
      return ACL_FORMAT_NCDHW;
    default:
      return ACL_FORMAT_ND;
  }
}

void ApplyPrivateStorageDesc(at::Tensor *tensor, const FxTorchTensorView &view, const std::vector<int64_t> &sizes) {
  auto *impl =
      static_cast<torch_npu::NPUStorageImpl *>(tensor->unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
  if (impl == nullptr) {
    return;
  }
  // format_contiguous densifies a private-format tensor whose base sizes or
  // strides do not match the logical view. Fill the same fields a cast tensor
  // carries so the kernel keeps the live format.
  auto &desc = impl->npu_desc_;
  // Match the tensor from_blob actually built. View strides can differ after
  // that call, and a mismatch makes format_contiguous drop the private format.
  desc.base_sizes_.assign(tensor->sizes().begin(), tensor->sizes().end());
  desc.base_strides_.assign(tensor->strides().begin(), tensor->strides().end());
  desc.base_offset_ = 0;
  desc.data_type_ = tensor->dtype();
  desc.npu_format_ = static_cast<aclFormat>(view.storage_format);
  desc.origin_format_ = BaseAclFormat(view.storage_format);
  if (view.storage_sizes != nullptr && view.storage_rank > 0U) {
    desc.storage_sizes_.assign(view.storage_sizes, view.storage_sizes + view.storage_rank);
  } else if (view.storage_format == static_cast<uint32_t>(ACL_FORMAT_FRACTAL_NZ)) {
    const auto inferred = InferNzStorageSizes(sizes, static_cast<int64_t>(tensor->element_size()));
    desc.storage_sizes_.assign(inferred.begin(), inferred.end());
  }
  int64_t storage_numel = 1;
  for (const int64_t dim : desc.storage_sizes_) {
    storage_numel *= dim;
  }
  if (storage_numel > 0) {
    tensor->storage().set_nbytes(static_cast<size_t>(storage_numel) * tensor->element_size());
  }
}

at::Tensor MakeTensor(const FxTorchTensorView &view, int32_t device_id) {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  if (view.rank != 0U) {
    sizes.assign(view.sizes, view.sizes + view.rank);
    strides.assign(view.strides, view.strides + view.rank);
  }
  const auto device = c10::Device(c10::DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(device_id));
  const auto options = at::TensorOptions().dtype(MapDtype(view.dtype)).device(device);
  auto tensor = at_npu::native::from_blob(view.data, sizes, strides, view.storage_offset, options, device);
  if (!IsBaseAclFormat(view.storage_format)) {
    ApplyPrivateStorageDesc(&tensor, view, sizes);
  }
  return tensor;
}

bool BuildIValue(const FxTorchValue &value, const c10::TypePtr &expected_type, int32_t device_id,
                 c10::IValue *result) {
  c10::TypePtr type = expected_type;
  if (type != nullptr && type->kind() == c10::TypeKind::OptionalType && value.kind != FX_TORCH_VALUE_NONE) {
    type = type->cast<c10::OptionalType>()->getElementType();
  }
  switch (value.kind) {
    case FX_TORCH_VALUE_NONE:
      *result = c10::IValue();
      return true;
    case FX_TORCH_VALUE_TENSOR:
      *result = MakeTensor(value.tensor, device_id);
      return true;
    case FX_TORCH_VALUE_DOUBLE:
      *result = value.double_value;
      return true;
    case FX_TORCH_VALUE_INT:
      if (type != nullptr && type->kind() == c10::TypeKind::BoolType) {
        *result = value.int_value != 0;
        return true;
      }
      if (type != nullptr && type->kind() == c10::TypeKind::SymIntType) {
        *result = c10::SymInt(value.int_value);
        return true;
      }
      if (type != nullptr && type->str() == "ScalarType") {
        *result = static_cast<at::ScalarType>(value.int_value);
        return true;
      }
      *result = value.int_value;
      return true;
    case FX_TORCH_VALUE_BOOL:
      *result = value.int_value != 0;
      return true;
    case FX_TORCH_VALUE_STRING:
      if (type != nullptr && type->kind() == c10::TypeKind::DeviceObjType) {
        const std::string text(value.string_value.data, value.string_value.size);
        *result = c10::Device(text);
      } else {
        *result = value.string_value.data == nullptr
                      ? std::string()
                      : std::string(value.string_value.data, value.string_value.size);
      }
      return true;
    case FX_TORCH_VALUE_LIST: {
      uint32_t element_kind = FX_TORCH_VALUE_INT;
      if (expected_type != nullptr && expected_type->kind() == c10::TypeKind::ListType) {
        const auto elem_type = expected_type->cast<c10::ListType>()->getElementType();
        switch (elem_type->kind()) {
          case c10::TypeKind::TensorType: element_kind = FX_TORCH_VALUE_TENSOR; break;
          case c10::TypeKind::FloatType: element_kind = FX_TORCH_VALUE_DOUBLE; break;
          case c10::TypeKind::BoolType: element_kind = FX_TORCH_VALUE_BOOL; break;
          case c10::TypeKind::StringType: element_kind = FX_TORCH_VALUE_STRING; break;
          default: element_kind = FX_TORCH_VALUE_INT; break;
        }
      }
      if (value.element_count != 0U) element_kind = value.elements[0].kind;
      if (value.element_count == 0U) {
        if (element_kind == FX_TORCH_VALUE_DOUBLE) *result = std::vector<double>();
        else if (element_kind == FX_TORCH_VALUE_BOOL) *result = std::vector<bool>();
        else if (element_kind == FX_TORCH_VALUE_STRING) *result = std::vector<std::string>();
        else if (element_kind == FX_TORCH_VALUE_TENSOR) *result = std::vector<at::Tensor>();
        else *result = std::vector<int64_t>();
        return true;
      }
      if (element_kind == FX_TORCH_VALUE_INT) {
        const bool want_symint =
            type != nullptr && type->kind() == c10::TypeKind::ListType &&
            type->cast<c10::ListType>()->getElementType()->kind() == c10::TypeKind::SymIntType;
        if (want_symint) {
          std::vector<c10::SymInt> list;
          list.reserve(value.element_count);
          for (size_t i = 0; i < value.element_count; ++i) {
            list.emplace_back(value.elements[i].int_value);
          }
          *result = std::move(list);
          return true;
        }
        std::vector<int64_t> list;
        list.reserve(value.element_count);
        for (size_t i = 0; i < value.element_count; ++i) list.push_back(value.elements[i].int_value);
        *result = std::move(list);
        return true;
      }
      if (element_kind == FX_TORCH_VALUE_DOUBLE) {
        std::vector<double> list;
        list.reserve(value.element_count);
        for (size_t i = 0; i < value.element_count; ++i) list.push_back(value.elements[i].double_value);
        *result = std::move(list);
        return true;
      }
      if (element_kind == FX_TORCH_VALUE_BOOL) {
        std::vector<bool> list;
        list.reserve(value.element_count);
        for (size_t i = 0; i < value.element_count; ++i) list.push_back(value.elements[i].int_value != 0);
        *result = std::move(list);
        return true;
      }
      if (element_kind == FX_TORCH_VALUE_STRING) {
        std::vector<std::string> list;
        list.reserve(value.element_count);
        for (size_t i = 0; i < value.element_count; ++i) {
          list.emplace_back(value.elements[i].string_value.data, value.elements[i].string_value.size);
        }
        *result = std::move(list);
        return true;
      }
      if (element_kind == FX_TORCH_VALUE_TENSOR) {
        std::vector<at::Tensor> list;
        list.reserve(value.element_count);
        for (size_t i = 0; i < value.element_count; ++i) {
          if (value.elements[i].kind != FX_TORCH_VALUE_TENSOR) return false;
          list.push_back(MakeTensor(value.elements[i].tensor, device_id));
        }
        *result = std::move(list);
        return true;
      }
      return false;
    }
    default:
      return false;
  }
}

void *CreatePlan(const FxTorchOpSpec *spec, FxTorchError *error) {
  if (spec == nullptr) {
    SetError(error, "null Torch op spec");
    return nullptr;
  }
  try {
    const auto plan = SelectPlan(*spec, error);
    return plan == nullptr ? nullptr : new TorchFallbackPlan(*plan);
  } catch (const std::exception &e) {
    SetError(error, e.what());
    return nullptr;
  }
}

bool ExtractResultTensors(const TorchFallbackPlan &plan, const std::vector<c10::IValue> &stack,
                          std::vector<at::Tensor> *results) {
  results->clear();
  results->reserve(plan.output_count);
  if (plan.return_kind == TorchFallbackPlan::ReturnKind::kTensorList) {
    if (stack.size() != 1U || !stack[0].isTensorList()) {
      return false;
    }
    const auto tensors = stack[0].toTensorVector();
    if (tensors.size() != plan.output_count) {
      return false;
    }
    results->assign(tensors.begin(), tensors.end());
    return true;
  }
  if (stack.size() != plan.output_count) {
    return false;
  }
  for (const auto &value : stack) {
    if (!value.isTensor()) {
      return false;
    }
    results->push_back(value.toTensor());
  }
  return true;
}

struct FallbackStreamBinding {
  c10_npu::NPUStream npu_stream;
  aclrtStream ge_stream = nullptr;
  bool need_join = false;
};

// Newer torch_npu rejects external (GE-wrapped) streams in several APIs
// ("External NPU stream is not supported" in ProcessGroupHCCL::collective,
// NPUEvent::record, ...), and the list grows per release. Fallbacks therefore
// DEFAULT to a torch_npu-managed stream. The caller must order device work
// across the GE and Torch streams. Only the
// proven-hot C++ aclnn wrappers (_C_ascend.*: npu_hc_pre_v2 / npu_hc_post and
// friends - pure kernel launches that never touch stream-checking APIs) keep
// the zero-overhead external wrap.
bool AllowExternalStreamWrap(const std::string &name) {
  return name.rfind("_C_ascend.", 0) == 0;
}

FallbackStreamBinding BindFallbackStream(uintptr_t stream, int32_t device_id,
                                         bool allow_external_wrap = false) {
  const auto index = static_cast<c10::DeviceIndex>(device_id);
#ifdef FXRT_HAS_C10_NPU_GET_STREAM_FROM_EXTERNAL
  if (allow_external_wrap) {
    if (stream != 0U) {
      return {c10_npu::getStreamFromExternal(reinterpret_cast<aclrtStream>(stream), index), nullptr, false};
    }
    return {c10_npu::getCurrentNPUStream(index), nullptr, false};
  }
#else
  (void)allow_external_wrap;
#endif
  FallbackStreamBinding binding{c10_npu::getCurrentNPUStream(index), nullptr, false};
  if (stream != 0U) {
    binding.ge_stream = reinterpret_cast<aclrtStream>(stream);
    if (binding.ge_stream != static_cast<aclrtStream>(binding.npu_stream)) {
      binding.need_join = true;
    }
  }
  return binding;
}

void JoinFallbackStream(const FallbackStreamBinding &binding) {
  if (binding.need_join) {
    // Drain Torch's host launch queue without waiting for device completion.
    (void)binding.npu_stream.stream(true);
  }
}

struct FallbackStreamJoiner {
  FallbackStreamBinding binding;
  ~FallbackStreamJoiner() { JoinFallbackStream(binding); }
};

bool PushFallbackArguments(const TorchFallbackPlan &plan, const c10::FunctionSchema &schema,
                           const FxTorchValue *arguments, size_t argument_count,
                           const std::vector<at::Tensor> &destinations, int32_t device_id,
                           std::vector<c10::IValue> *stack, FxTorchError *error) {
  stack->reserve(schema.arguments().size());
  size_t input_index = 0;
  for (size_t schema_index = 0; schema_index < schema.arguments().size(); ++schema_index) {
    const auto out_it =
        std::find(plan.out_arg_indices.begin(), plan.out_arg_indices.end(), schema_index);
    if (out_it != plan.out_arg_indices.end()) {
      const size_t output_index =
          static_cast<size_t>(std::distance(plan.out_arg_indices.begin(), out_it));
      stack->emplace_back(destinations[output_index]);
      continue;
    }
    if (input_index < argument_count) {
      c10::IValue value;
      if (!BuildIValue(arguments[input_index], plan.argument_types[input_index], device_id, &value)) {
        SetError(error, "unsupported Torch fallback argument value");
        return false;
      }
      stack->push_back(std::move(value));
      ++input_index;
      continue;
    }
    const auto &schema_arg = schema.arguments()[schema_index];
    if (!schema_arg.default_value().has_value()) {
      SetError(error, "Torch fallback argument count does not match selected schema");
      return false;
    }
    stack->push_back(*schema_arg.default_value());
  }
  return true;
}

// Dense D2D writeback matches AclnnInplaceCopy / AclnnContiguous: aclrtMemcpyAsync on the
// shared GE stream. Converter fallback cannot adopt ATen storage like OpTorchCall
// (SetDataPtrFromAten) because GE already malloc'd the destination. Caller must densify
// and drain the TorchNPU host queue before this memcpy.
bool CopyContiguousDeviceToGe(const at::Tensor &src, at::Tensor *dst, size_t dst_nbytes, aclrtStream stream,
                              FxTorchError *error) {
  if (dst == nullptr) {
    SetError(error, "invalid Torch fallback writeback arguments");
    return false;
  }
  if (!src.is_contiguous() || !dst->is_contiguous()) {
    SetError(error, "Torch fallback writeback requires contiguous src and GE destination");
    return false;
  }
  const size_t bytes = src.numel() * src.element_size();
  if (bytes > dst_nbytes) {
    SetError(error, "Torch fallback result exceeds GE output allocation");
    return false;
  }
  if (bytes == 0U || src.data_ptr() == dst->data_ptr()) {
    return true;
  }
  const aclError ret = aclrtMemcpyAsync(dst->data_ptr(), dst_nbytes, src.data_ptr(), bytes,
                                        ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
  if (ret != ACL_SUCCESS) {
    std::ostringstream detail;
    detail << "aclrtMemcpyAsync D2D failed for Torch fallback writeback, ret=" << static_cast<int>(ret);
    SetError(error, detail.str());
    return false;
  }
  return true;
}

at::Tensor ContiguousFallbackSource(const at::Tensor &src) {
  return src.is_contiguous() ? src : src.contiguous();
}

// c10d functional collectives return before HCCL finishes; wait_tensor holds the
// Work handle. Converter memcpy into a GE buffer drops that handle, so wait here
// before writeback. Plain compute tensors are a no-op.
bool WaitAsyncCollectiveTensors(std::vector<at::Tensor> *results, FxTorchError *error) {
  if (results == nullptr) {
    SetError(error, "invalid Torch fallback wait arguments");
    return false;
  }
  const auto found =
      c10::Dispatcher::singleton().findSchema({"_c10d_functional::wait_tensor", ""});
  if (!found.has_value()) {
    return true;
  }
  for (auto &tensor : *results) {
    c10::Stack stack;
    stack.emplace_back(tensor);
    found->callBoxed(&stack);
    if (stack.size() != 1U || !stack[0].isTensor()) {
      SetError(error, "c10d wait_tensor returned a non-tensor result");
      return false;
    }
    tensor = std::move(stack[0]).toTensor();
  }
  return true;
}

bool FallbackOutputMatchesGe(const at::Tensor &result, const FxTorchTensorView &output,
                             const std::vector<int64_t> &want_sizes, size_t index, FxTorchError *error) {
  if (result.scalar_type() == MapDtype(output.dtype) && result.sizes().vec() == want_sizes) {
    return true;
  }
  std::ostringstream got_shape;
  for (const auto d : result.sizes()) {
    got_shape << d << ",";
  }
  std::ostringstream want_shape;
  for (const auto d : want_sizes) {
    want_shape << d << ",";
  }
  std::ostringstream detail;
  detail << "Torch fallback output dtype or shape does not match GE allocation: out " << index
         << " got dtype=" << static_cast<int>(result.scalar_type())
         << " want_dtype=" << static_cast<int>(MapDtype(output.dtype)) << " got_sizes=[" << got_shape.str()
         << "] want_sizes=[" << want_shape.str() << "]";
  SetError(error, detail.str());
  return false;
}

// DSA CP compressor_metadata sizes rope for num_actual_tokens (e.g. 3 → dim0=1
// with cmp_ratio=4). getitem_slice IR may still carry the padded gather window
// (4), so aclnnCompressor tiling wants min(T, T/ratio+batch)=2 and dies.
// Narrow hidden to the largest T that matches the rope table.
void AlignCompressorHiddenToRope(const TorchFallbackPlan &plan, std::vector<c10::IValue> *stack) {
  if (stack == nullptr || stack->empty() || !(*stack)[0].isTensor()) {
    return;
  }
  const std::string &name = plan.qualified_name;
  if (name.find("compressor") == std::string::npos || name.find("metadata") != std::string::npos) {
    return;
  }
  at::Tensor hidden = (*stack)[0].toTensor();
  if (hidden.dim() < 1) {
    return;
  }
  const auto &schema_args = plan.handle.schema().arguments();
  int64_t ratio = 4;
  int64_t batch = 1;
  at::Tensor rope;
  bool have_rope = false;
  for (size_t i = 0; i < schema_args.size() && i < stack->size(); ++i) {
    const std::string arg_name = schema_args[i].name();
    if ((arg_name == "cmp_ratio" || arg_name == "compress_ratio") && (*stack)[i].isInt()) {
      ratio = (*stack)[i].toInt();
    }
    if (arg_name == "batch_size" && (*stack)[i].isInt()) {
      batch = (*stack)[i].toInt();
    }
    if ((arg_name.find("sin") != std::string::npos) && (*stack)[i].isTensor()) {
      rope = (*stack)[i].toTensor();
      have_rope = true;
    }
  }
  if (!have_rope) {
    for (size_t i = stack->size(); i-- > 1U;) {
      if ((*stack)[i].isTensor() && (*stack)[i].toTensor().scalar_type() == at::kFloat) {
        rope = (*stack)[i].toTensor();
        have_rope = true;
        break;
      }
    }
  }
  if (!have_rope || rope.dim() < 1 || ratio <= 0 || batch <= 0) {
    return;
  }
  const int64_t tokens = hidden.size(0);
  const int64_t rope0 = rope.size(0);
  if (tokens <= 0 || rope0 <= 0) {
    return;
  }
  if (std::min(tokens, tokens / ratio + batch) == rope0) {
    return;
  }
  int64_t aligned = -1;
  for (int64_t t = tokens; t >= 1; --t) {
    if (std::min(t, t / ratio + batch) == rope0) {
      aligned = t;
      break;
    }
  }
  if (aligned > 0 && aligned < tokens) {
    (*stack)[0] = hidden.narrow(/*dim=*/0, /*start=*/0, aligned).contiguous();
  }
}

int32_t RunIntoPlan(void *opaque_plan, const FxTorchValue *arguments, size_t argument_count,
                const FxTorchTensorView *outputs, size_t output_count, uintptr_t stream,
                int32_t device_id, FxTorchError *error) {
  if (opaque_plan == nullptr || (arguments == nullptr && argument_count != 0U) || outputs == nullptr ||
      output_count == 0U) {
    SetError(error, "invalid Torch fallback run arguments");
    return -1;
  }
  try {
    auto *plan = static_cast<TorchFallbackPlan *>(opaque_plan);
    const auto &schema = plan->handle.schema();
    if (argument_count > plan->argument_types.size() || output_count != plan->output_count) {
      SetError(error, "Torch fallback argument count does not match selected schema");
      return -1;
    }
    pybind11::gil_scoped_acquire gil;
    std::vector<at::Tensor> destinations;
    destinations.reserve(output_count);
    std::vector<std::vector<int64_t>> output_sizes(output_count);
    for (size_t i = 0; i < output_count; ++i) {
      if (outputs[i].rank != 0U) {
        output_sizes[i].assign(outputs[i].sizes, outputs[i].sizes + outputs[i].rank);
      }
      destinations.push_back(MakeTensor(outputs[i], device_id));
    }

    FallbackStreamJoiner stream_binding{
      BindFallbackStream(stream, device_id, AllowExternalStreamWrap(plan->qualified_name))};
    c10_npu::NPUStreamGuard stream_guard(stream_binding.binding.npu_stream);
    std::vector<c10::IValue> stack;
    if (!PushFallbackArguments(*plan, schema, arguments, argument_count, destinations, device_id, &stack,
                               error)) {
      return -1;
    }
    AlignCompressorHiddenToRope(*plan, &stack);
    plan->handle.callBoxed(&stack);
    if (plan->synthetic_outputs) {
      // Mutation-only plan: the outputs alias the write-alias inputs (ref
      // pairs); the op already wrote the shared buffers through its input
      // arguments. Nothing to extract or write back.
      (void)stream_binding.binding.npu_stream.stream(true);
      return 0;
    }
    std::vector<at::Tensor> results;
    if (!ExtractResultTensors(*plan, stack, &results)) {
      SetError(error, "Torch fallback returned an unexpected Tensor output structure or count");
      return -1;
    }
    for (size_t i = 0; i < output_count; ++i) {
      if (!FallbackOutputMatchesGe(results[i], outputs[i], output_sizes[i], i, error)) {
        return -1;
      }
    }
    if (plan->out_arg_indices.empty() && !WaitAsyncCollectiveTensors(&results, error)) {
      return -1;
    }
    std::vector<at::Tensor> writeback_sources;
    if (plan->out_arg_indices.empty()) {
      writeback_sources.reserve(output_count);
      for (size_t i = 0; i < output_count; ++i) {
        writeback_sources.push_back(ContiguousFallbackSource(results[i]));
      }
    }
    // Drain so Dispatcher kernels and any densify copies are on the ACL stream
    // before aclrtMemcpyAsync. Out overloads already wrote GE memory; skip copy.
    const aclrtStream acl_stream = stream_binding.binding.npu_stream.stream(true);
    for (size_t i = 0; i < writeback_sources.size(); ++i) {
      if (!CopyContiguousDeviceToGe(writeback_sources[i], &destinations[i], outputs[i].nbytes, acl_stream, error)) {
        return -1;
      }
    }
    return 0;
  } catch (const std::exception &e) {
    SetError(error, e.what());
    return -1;
  }
}


struct DeferredResult {
  std::vector<at::Tensor> tensors;
  // Mutation-only plan: outputs alias the write-alias inputs; materialize
  // must not copy anything (the buffers were written through the inputs).
  bool synthetic = false;
};

int32_t RunDeferredPlan(void *opaque_plan, const FxTorchValue *arguments, size_t argument_count,
                        uintptr_t stream, int32_t device_id, uintptr_t *result_handle,
                        FxTorchFallbackResultMetadata *metadata, size_t metadata_count,
                        FxTorchError *error) {
  if (opaque_plan == nullptr || result_handle == nullptr || metadata == nullptr) {
    SetError(error, "invalid deferred Torch fallback arguments");
    return -1;
  }
  *result_handle = 0;
  try {
    auto *plan = static_cast<TorchFallbackPlan *>(opaque_plan);
    if (argument_count != plan->argument_types.size() || metadata_count != plan->output_count) {
      SetError(error, "deferred Torch fallback requires a functional schema and matching counts");
      return -1;
    }
    // Out plans write a preallocated ND shell. A private-format input must
    // use the functional schema instead, so the kernel allocates the result
    // and keeps that format. The caller adopts the returned storage.
    c10::optional<c10::OperatorHandle> functional_handle;
    const c10::OperatorHandle *active = &plan->handle;
    if (!plan->out_arg_indices.empty()) {
      functional_handle = c10::Dispatcher::singleton().findSchema({plan->qualified_name, ""});
      if (!functional_handle.has_value()) {
        SetError(error, "torch fallback: private-format input has no functional schema");
        return -1;
      }
      active = &(*functional_handle);
    }
    pybind11::gil_scoped_acquire gil;
    std::vector<c10::IValue> stack;
    stack.reserve(argument_count);
    for (size_t i = 0; i < argument_count; ++i) {
      c10::IValue value;
      if (!BuildIValue(arguments[i], plan->argument_types[i], device_id, &value)) {
        SetError(error, "unsupported Torch fallback argument value");
        return -1;
      }
      stack.push_back(std::move(value));
    }
    // Mutation-only plans derive their synthetic outputs' metadata from the
    // write-alias INPUT tensors. callBoxed consumes the argument stack (the
    // post-call stack holds returns, and mutation-only schemas return
    // nothing), so the referenced inputs must be captured BEFORE the call.
    std::vector<at::Tensor> synthetic_inputs(metadata_count);
    if (plan->synthetic_outputs) {
      for (const auto &pair : plan->ref_pairs) {
        if (pair.first >= stack.size() || !stack[pair.first].isTensor() ||
            pair.second >= metadata_count) {
          SetError(error, "mutation-only Torch fallback has no tensor for a ref output");
          return -1;
        }
        synthetic_inputs[pair.second] = stack[pair.first].toTensor();
      }
    }
    AlignCompressorHiddenToRope(*plan, &stack);
    FallbackStreamJoiner stream_binding{
      BindFallbackStream(stream, device_id, AllowExternalStreamWrap(plan->qualified_name))};
    c10_npu::NPUStreamGuard stream_guard(stream_binding.binding.npu_stream);
    active->callBoxed(&stack);
    auto holder = std::make_unique<DeferredResult>();
    if (plan->synthetic_outputs) {
      // Mutation-only plan: the op already wrote the write-alias input
      // tensors in place; their pre-call shapes/dtypes describe the outputs.
      holder->synthetic = true;
      for (const auto &pair : plan->ref_pairs) {
        const auto &tensor = synthetic_inputs[pair.second];
        const auto rank = tensor.dim();
        if (rank > static_cast<int64_t>(FX_TORCH_FALLBACK_MAX_RANK)) {
          SetError(error, "Torch fallback result rank exceeds ABI limit");
          return -1;
        }
        metadata[pair.second].rank = static_cast<uint32_t>(rank);
        metadata[pair.second].dtype = ToGeDtype(tensor.scalar_type());
        for (size_t j = 0; j < rank; ++j) {
          metadata[pair.second].sizes[j] = tensor.size(static_cast<int64_t>(j));
        }
      }
      (void)stream_binding.binding.npu_stream.stream(true);
      *result_handle = reinterpret_cast<uintptr_t>(holder.release());
      return 0;
    }
    if (!ExtractResultTensors(*plan, stack, &holder->tensors)) {
      SetError(error, "Torch fallback returned an unexpected Tensor output structure or count");
      return -1;
    }
    if (!WaitAsyncCollectiveTensors(&holder->tensors, error)) {
      return -1;
    }
    for (size_t i = 0; i < holder->tensors.size(); ++i) {
      const auto &tensor = holder->tensors[i];
      const auto rank = tensor.dim();
      if (rank > static_cast<int64_t>(FX_TORCH_FALLBACK_MAX_RANK)) {
        SetError(error, "Torch fallback result rank exceeds ABI limit");
        return -1;
      }
      metadata[i].rank = static_cast<uint32_t>(rank);
      metadata[i].dtype = ToGeDtype(tensor.scalar_type());
      for (size_t j = 0; j < rank; ++j) {
        metadata[i].sizes[j] = tensor.size(static_cast<int64_t>(j));
      }
    }
    (void)stream_binding.binding.npu_stream.stream(true);
    *result_handle = reinterpret_cast<uintptr_t>(holder.release());
    return 0;
  } catch (const std::exception &e) {
    SetError(error, e.what());
    return -1;
  }
}

int32_t MaterializeResult(uintptr_t result_handle, const FxTorchTensorView *outputs, size_t output_count,
                          uintptr_t stream, int32_t device_id, FxTorchError *error) {
  if (result_handle == 0 || outputs == nullptr) {
    SetError(error, "invalid deferred Torch fallback result");
    return -1;
  }
  try {
    auto *holder = reinterpret_cast<DeferredResult *>(result_handle);
    if (holder->synthetic) {
      // Mutation-only plan: the outputs share the write-alias input buffers
      // (MakeOutputRefInput); the op already wrote them. Nothing to copy.
      return 0;
    }
    if (output_count != holder->tensors.size()) {
      SetError(error, "deferred Torch fallback output count mismatch");
      return -1;
    }
    pybind11::gil_scoped_acquire gil;
    FallbackStreamJoiner stream_binding{BindFallbackStream(stream, device_id)};
    c10_npu::NPUStreamGuard stream_guard(stream_binding.binding.npu_stream);
    std::vector<at::Tensor> destinations;
    destinations.reserve(output_count);
    std::vector<at::Tensor> writeback_sources;
    writeback_sources.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i) {
      const auto &source = holder->tensors[i];
      if (source.scalar_type() != MapDtype(outputs[i].dtype) ||
          source.dim() != static_cast<int64_t>(outputs[i].rank)) {
        SetError(error, "deferred Torch fallback output dtype or rank mismatch");
        return -1;
      }
      for (size_t j = 0; j < outputs[i].rank; ++j) {
        if (source.size(static_cast<int64_t>(j)) != outputs[i].sizes[j]) {
          SetError(error, "deferred Torch fallback output shape mismatch");
          return -1;
        }
      }
      destinations.push_back(MakeTensor(outputs[i], device_id));
      writeback_sources.push_back(ContiguousFallbackSource(source));
    }
    const aclrtStream acl_stream = stream_binding.binding.npu_stream.stream(true);
    for (size_t i = 0; i < output_count; ++i) {
      if (!CopyContiguousDeviceToGe(writeback_sources[i], &destinations[i], outputs[i].nbytes, acl_stream, error)) {
        return -1;
      }
    }
    return 0;
  } catch (const std::exception &e) {
    SetError(error, e.what());
    return -1;
  }
}

void DestroyResult(uintptr_t result_handle) {
  delete reinterpret_cast<DeferredResult *>(result_handle);
}

int32_t ExportDeferredStorage(uintptr_t result_handle, void **addrs, size_t *nbytes, size_t count,
                              FxTorchError *error) {
  if (result_handle == 0 || addrs == nullptr || nbytes == nullptr) {
    SetError(error, "invalid deferred Torch fallback storage export");
    return -1;
  }
  try {
    auto *holder = reinterpret_cast<DeferredResult *>(result_handle);
    if (holder->synthetic) {
      // Mutation-only plan: outputs alias the write-alias input buffers
      // (bound on the runtime side via MakeOutputRefInput); the result
      // carries no torch storages to export.
      for (size_t i = 0; i < count; ++i) {
        addrs[i] = nullptr;
        nbytes[i] = 0U;
      }
      return 0;
    }
    if (count != holder->tensors.size()) {
      SetError(error, "deferred Torch fallback storage count mismatch");
      return -1;
    }
    pybind11::gil_scoped_acquire gil;
    for (size_t i = 0; i < count; ++i) {
      const auto &tensor = holder->tensors[i];
      const bool has_storage = tensor.defined() && tensor.numel() > 0;
      addrs[i] = has_storage ? tensor.data_ptr() : nullptr;
      nbytes[i] = tensor.defined() ? static_cast<size_t>(tensor.nbytes()) : 0U;
    }
    return 0;
  } catch (const std::exception &e) {
    SetError(error, e.what());
    return -1;
  }
}

int32_t ExportDeferredLayout(uintptr_t result_handle, int32_t *formats, int64_t *storage_ranks,
                             int64_t *storage_sizes, size_t storage_stride, size_t count,
                             FxTorchError *error) {
  if (result_handle == 0 || formats == nullptr || storage_ranks == nullptr || storage_sizes == nullptr ||
      storage_stride == 0U) {
    SetError(error, "invalid deferred Torch fallback layout export");
    return -1;
  }
  try {
    auto *holder = reinterpret_cast<DeferredResult *>(result_handle);
    if (holder->synthetic) {
      for (size_t i = 0; i < count; ++i) {
        formats[i] = static_cast<int32_t>(ACL_FORMAT_ND);
        storage_ranks[i] = 0;
      }
      return 0;
    }
    if (count != holder->tensors.size()) {
      SetError(error, "deferred Torch fallback layout count mismatch");
      return -1;
    }
    pybind11::gil_scoped_acquire gil;
    for (size_t i = 0; i < count; ++i) {
      const auto &tensor = holder->tensors[i];
      if (!tensor.defined()) {
        formats[i] = static_cast<int32_t>(ACL_FORMAT_ND);
        storage_ranks[i] = 0;
        continue;
      }
      formats[i] = static_cast<int32_t>(at_npu::native::get_npu_format(tensor));
      const auto sizes = at_npu::native::get_npu_storage_sizes(tensor);
      if (sizes.size() > storage_stride) {
        SetError(error, "deferred Torch fallback storage rank exceeds export stride");
        return -1;
      }
      storage_ranks[i] = static_cast<int64_t>(sizes.size());
      for (size_t j = 0; j < sizes.size(); ++j) {
        storage_sizes[i * storage_stride + j] = sizes[j];
      }
    }
    return 0;
  } catch (const std::exception &e) {
    SetError(error, e.what());
    return -1;
  }
}

int32_t ExportDeferredStrides(uintptr_t result_handle, int64_t *ranks, int64_t *flat_strides, int64_t *offsets,
                              size_t stride_capacity, size_t count, FxTorchError *error) {
  if (result_handle == 0 || ranks == nullptr || flat_strides == nullptr || offsets == nullptr ||
      stride_capacity == 0U) {
    SetError(error, "invalid deferred Torch fallback stride export");
    return -1;
  }
  try {
    auto *holder = reinterpret_cast<DeferredResult *>(result_handle);
    if (holder->synthetic) {
      for (size_t i = 0; i < count; ++i) {
        ranks[i] = 0;
        offsets[i] = 0;
      }
      return 0;
    }
    if (count != holder->tensors.size()) {
      SetError(error, "deferred Torch fallback stride count mismatch");
      return -1;
    }
    pybind11::gil_scoped_acquire gil;
    for (size_t i = 0; i < count; ++i) {
      const auto &tensor = holder->tensors[i];
      if (!tensor.defined()) {
        ranks[i] = 0;
        offsets[i] = 0;
        continue;
      }
      if (static_cast<size_t>(tensor.dim()) > stride_capacity) {
        SetError(error, "deferred Torch fallback stride rank exceeds export capacity");
        return -1;
      }
      ranks[i] = tensor.dim();
      offsets[i] = tensor.storage_offset();
      for (int64_t j = 0; j < tensor.dim(); ++j) {
        flat_strides[i * stride_capacity + static_cast<size_t>(j)] = tensor.stride(j);
      }
    }
    return 0;
  } catch (const std::exception &e) {
    SetError(error, e.what());
    return -1;
  }
}

void DestroyPlan(void *opaque_plan) {
  delete static_cast<TorchFallbackPlan *>(opaque_plan);
}

int32_t PlanUsesOutSchema(void *opaque_plan) {
  const auto *plan = static_cast<const TorchFallbackPlan *>(opaque_plan);
  return (plan != nullptr && !plan->out_arg_indices.empty()) ? 1 : 0;
}

size_t PlanRefPairs(void *opaque_plan, size_t *schema_arg_indices, size_t *output_indices,
                    size_t capacity) {
  const auto *plan = static_cast<const TorchFallbackPlan *>(opaque_plan);
  if (plan == nullptr) {
    return 0U;
  }
  const size_t count = plan->ref_pairs.size();
  if (schema_arg_indices != nullptr && output_indices != nullptr) {
    const size_t fill = std::min(count, capacity);
    for (size_t i = 0; i < fill; ++i) {
      schema_arg_indices[i] = plan->ref_pairs[i].first;
      output_indices[i] = plan->ref_pairs[i].second;
    }
  }
  return count;
}

struct PythonCallPlan {
  pybind11::function func;
  std::string module_name;
  std::string func_name;
  size_t output_count = 0U;
};

pybind11::function ResolvePythonCallable(const std::string &module_name, const std::string &func_name,
                                         FxTorchError *error) {
  constexpr const char *kTorchOps = "torch.ops";
  constexpr const char *kTorchOpsInner = "torch._ops";
  try {
    if (module_name.rfind(kTorchOpsInner, 0) != 0) {
      return pybind11::module_::import(module_name.c_str()).attr(func_name.c_str()).cast<pybind11::function>();
    }
    std::string sub = module_name.substr(std::strlen(kTorchOpsInner));
    if (!sub.empty() && sub.front() == '.') {
      sub.erase(0, 1);
    }
    if (sub.empty()) {
      SetError(error, "invalid torch._ops module name");
      return {};
    }
    return pybind11::module_::import(kTorchOps)
        .attr(sub.c_str())
        .attr(func_name.c_str())
        .cast<pybind11::function>();
  } catch (const std::exception &e) {
    SetError(error, std::string("failed to import ") + module_name + "." + func_name + ": " + e.what());
    return {};
  }
}

bool BuildPyObject(const FxTorchValue &value, int32_t device_id, pybind11::object *out) {
  switch (value.kind) {
    case FX_TORCH_VALUE_NONE:
      *out = pybind11::none();
      return true;
    case FX_TORCH_VALUE_TENSOR:
      *out = pybind11::cast(MakeTensor(value.tensor, device_id));
      return true;
    case FX_TORCH_VALUE_DOUBLE:
      *out = pybind11::cast(value.double_value);
      return true;
    case FX_TORCH_VALUE_INT:
      *out = pybind11::cast(value.int_value);
      return true;
    case FX_TORCH_VALUE_BOOL:
      *out = pybind11::cast(value.int_value != 0);
      return true;
    case FX_TORCH_VALUE_STRING:
      *out = pybind11::cast(value.string_value.data == nullptr
                                ? std::string()
                                : std::string(value.string_value.data, value.string_value.size));
      return true;
    case FX_TORCH_VALUE_LIST: {
      pybind11::list items;
      for (size_t i = 0; i < value.element_count; ++i) {
        pybind11::object elem;
        if (!BuildPyObject(value.elements[i], device_id, &elem)) {
          return false;
        }
        items.append(std::move(elem));
      }
      *out = std::move(items);
      return true;
    }
    default:
      return false;
  }
}

bool CollectPythonArgs(const FxTorchValue *arguments, size_t argument_count, int32_t device_id,
                       pybind11::tuple *out, FxTorchError *error) {
  pybind11::tuple args(argument_count);
  for (size_t i = 0; i < argument_count; ++i) {
    pybind11::object obj;
    if (!BuildPyObject(arguments[i], device_id, &obj)) {
      SetError(error, "unsupported python call argument value");
      return false;
    }
    args[i] = std::move(obj);
  }
  *out = std::move(args);
  return true;
}

bool ExtractPythonTensors(const pybind11::object &result, size_t expect, std::vector<at::Tensor> *tensors,
                          FxTorchError *error) {
  try {
    if (pybind11::isinstance<pybind11::tuple>(result)) {
      const auto tup = result.cast<pybind11::tuple>();
      if (tup.size() != expect) {
        SetError(error, "python call tuple size does not match output count");
        return false;
      }
      tensors->reserve(expect);
      for (size_t i = 0; i < expect; ++i) {
        tensors->push_back(tup[i].cast<at::Tensor>());
      }
      return true;
    }
    if (expect != 1U) {
      SetError(error, "python call returned a single value but multiple outputs were expected");
      return false;
    }
    tensors->push_back(result.cast<at::Tensor>());
    return true;
  } catch (const std::exception &e) {
    SetError(error, std::string("python call result is not a Tensor / Tensor tuple: ") + e.what());
    return false;
  }
}

int32_t FillResultMetadata(const std::vector<at::Tensor> &tensors, FxTorchFallbackResultMetadata *metadata,
                           FxTorchError *error) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto &tensor = tensors[i];
    const auto rank = tensor.dim();
    if (rank > static_cast<int64_t>(FX_TORCH_FALLBACK_MAX_RANK)) {
      SetError(error, "python call result rank exceeds ABI limit");
      return -1;
    }
    metadata[i].rank = static_cast<uint32_t>(rank);
    metadata[i].dtype = ToGeDtype(tensor.scalar_type());
    for (int64_t j = 0; j < rank; ++j) {
      metadata[i].sizes[static_cast<size_t>(j)] = tensor.size(j);
    }
  }
  return 0;
}

// python_call reuses FxTorchOpSpec: the module rides in qualified_name, the
// function in overload_name.
void *CreatePythonCallPlan(const FxTorchOpSpec *spec, FxTorchError *error) {
  if (spec == nullptr) {
    SetError(error, "invalid python call spec");
    return nullptr;
  }
  const std::string module_name = QualifiedName(spec->qualified_name);
  const std::string func_name = QualifiedName(spec->overload_name);
  if (module_name.empty() || func_name.empty() || spec->output_count == 0U) {
    SetError(error, "python call requires module_name, func_name and at least one output");
    return nullptr;
  }
  pybind11::gil_scoped_acquire gil;
  auto func = ResolvePythonCallable(module_name, func_name, error);
  if (func.is_none() || !func) {
    return nullptr;
  }
  auto *plan = new PythonCallPlan();
  plan->func = std::move(func);
  plan->module_name = module_name;
  plan->func_name = func_name;
  plan->output_count = spec->output_count;
  return plan;
}

pybind11::object InvokePythonPlan(PythonCallPlan *plan, const FxTorchValue *arguments, size_t argument_count,
                                  int32_t device_id, uintptr_t stream, FxTorchError *error) {
  pybind11::tuple args;
  if (!CollectPythonArgs(arguments, argument_count, device_id, &args, error)) {
    return {};
  }
  FallbackStreamJoiner stream_binding{BindFallbackStream(stream, device_id, true)};
  c10_npu::NPUStreamGuard stream_guard(stream_binding.binding.npu_stream);
  try {
    pybind11::object result = argument_count == 0U ? plan->func() : plan->func(*args);
    (void)stream_binding.binding.npu_stream.stream(true);
    return result;
  } catch (const std::exception &e) {
    SetError(error, std::string("python call ") + plan->module_name + "." + plan->func_name +
                        " failed: " + e.what());
    return {};
  }
}

int32_t RunPythonCallInto(void *opaque_plan, const FxTorchValue *arguments, size_t argument_count,
                          const FxTorchTensorView *outputs, size_t output_count, uintptr_t stream,
                          int32_t device_id, FxTorchError *error) {
  if (opaque_plan == nullptr || outputs == nullptr || output_count == 0U) {
    SetError(error, "invalid python call run arguments");
    return -1;
  }
  try {
    auto *plan = static_cast<PythonCallPlan *>(opaque_plan);
    if (output_count != plan->output_count) {
      SetError(error, "python call output count mismatch");
      return -1;
    }
    pybind11::gil_scoped_acquire gil;
    const auto result = InvokePythonPlan(plan, arguments, argument_count, device_id, stream, error);
    if (!result) {
      return -1;
    }
    std::vector<at::Tensor> tensors;
    if (!ExtractPythonTensors(result, output_count, &tensors, error)) {
      return -1;
    }
    FallbackStreamJoiner stream_binding{BindFallbackStream(stream, device_id, true)};
    c10_npu::NPUStreamGuard stream_guard(stream_binding.binding.npu_stream);
    std::vector<at::Tensor> destinations;
    destinations.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i) {
      destinations.push_back(MakeTensor(outputs[i], device_id));
    }
    const aclrtStream acl_stream = stream_binding.binding.npu_stream.stream(true);
    for (size_t i = 0; i < output_count; ++i) {
      auto src = ContiguousFallbackSource(tensors[i]);
      if (!CopyContiguousDeviceToGe(src, &destinations[i], outputs[i].nbytes, acl_stream, error)) {
        return -1;
      }
    }
    return 0;
  } catch (const std::exception &e) {
    SetError(error, e.what());
    return -1;
  }
}

int32_t RunPythonCallDeferred(void *opaque_plan, const FxTorchValue *arguments, size_t argument_count,
                              uintptr_t stream, int32_t device_id, uintptr_t *result_handle,
                              FxTorchFallbackResultMetadata *metadata, size_t metadata_count,
                              FxTorchError *error) {
  if (opaque_plan == nullptr || result_handle == nullptr || metadata == nullptr) {
    SetError(error, "invalid deferred python call arguments");
    return -1;
  }
  *result_handle = 0;
  try {
    auto *plan = static_cast<PythonCallPlan *>(opaque_plan);
    if (metadata_count != plan->output_count) {
      SetError(error, "deferred python call metadata count mismatch");
      return -1;
    }
    pybind11::gil_scoped_acquire gil;
    const auto result = InvokePythonPlan(plan, arguments, argument_count, device_id, stream, error);
    if (!result) {
      return -1;
    }
    auto holder = std::make_unique<DeferredResult>();
    if (!ExtractPythonTensors(result, plan->output_count, &holder->tensors, error)) {
      return -1;
    }
    if (FillResultMetadata(holder->tensors, metadata, error) != 0) {
      return -1;
    }
    *result_handle = reinterpret_cast<uintptr_t>(holder.release());
    return 0;
  } catch (const std::exception &e) {
    SetError(error, e.what());
    return -1;
  }
}

void DestroyPythonCallPlan(void *opaque_plan) {
  if (opaque_plan == nullptr) {
    return;
  }
  pybind11::gil_scoped_acquire gil;
  delete static_cast<PythonCallPlan *>(opaque_plan);
}

int32_t PythonCallUsesOutSchema(void *) { return 0; }

size_t PythonCallRefPairs(void *, size_t *, size_t *, size_t) { return 0U; }

// python_call speaks the plain function table: no out= schema and no ref pairs.
const FxTorchFallbackApi kPythonCallApi = {
    FX_TORCH_FALLBACK_ABI_VERSION,
    sizeof(FxTorchFallbackApi),
    &CreatePythonCallPlan,
    &RunPythonCallInto,
    &RunPythonCallDeferred,
    &MaterializeResult,
    &DestroyResult,
    &DestroyPythonCallPlan,
    &PythonCallUsesOutSchema,
    &PythonCallRefPairs,
    &ExportDeferredStorage,
    &ExportDeferredLayout,
    &ExportDeferredStrides,
};

const FxTorchFallbackApi kFallbackApi = {
    FX_TORCH_FALLBACK_ABI_VERSION,
    sizeof(FxTorchFallbackApi),
    &CreatePlan,
    &RunIntoPlan,
    &RunDeferredPlan,
    &MaterializeResult,
    &DestroyResult,
    &DestroyPlan,
    &PlanUsesOutSchema,
    &PlanRefPairs,
    &ExportDeferredStorage,
    &ExportDeferredLayout,
    &ExportDeferredStrides,
};

void ReleaseRange(const std::vector<uintptr_t> &owned_ctxs, const std::vector<uintptr_t> &free_fns, size_t begin) {
  for (size_t i = begin; i < owned_ctxs.size(); ++i) {
    if (owned_ctxs[i] != 0 && free_fns[i] != 0) {
      reinterpret_cast<FxFreeOwnedFn>(free_fns[i])(reinterpret_cast<void *>(owned_ctxs[i]));
    }
  }
}

/**
 * Take ownership of GE model outputs and return them as torch NPU tensors.
 *
 * All-or-nothing: a buffer is either owned by a returned tensor or freed here. A record
 * with owned_ctx == 0 is wrapped non-owning (GE did not own that memory either).
 */
std::vector<at::Tensor> WrapGeOutputs(const std::vector<uintptr_t> &data_ptrs,
                                      const std::vector<std::vector<int64_t>> &shapes,
                                      const std::vector<at::ScalarType> &dtypes,
                                      const std::vector<uintptr_t> &owned_ctxs,
                                      const std::vector<uintptr_t> &free_fns, int64_t device_index) {
  const size_t n = data_ptrs.size();
  if (shapes.size() != n || dtypes.size() != n || owned_ctxs.size() != n || free_fns.size() != n) {
    ReleaseRange(owned_ctxs, free_fns, 0);
    TORCH_CHECK(false, "wrap_ge_outputs: mismatched record lengths");
  }

  std::vector<at::Tensor> tensors;
  tensors.reserve(n);
  size_t i = 0;
  try {
    for (; i < n; ++i) {
      TORCH_CHECK(data_ptrs[i] != 0, "wrap_ge_outputs: null device address for output ", i);
      const auto device = c10::Device(c10::DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(device_index));
      const auto options = at::TensorOptions().dtype(dtypes[i]).device(device);
      void *data = reinterpret_cast<void *>(data_ptrs[i]);

      at::Tensor t;
      if (owned_ctxs[i] != 0 && free_fns[i] != 0) {
        void *ctx = reinterpret_cast<void *>(owned_ctxs[i]);
        auto fn = reinterpret_cast<FxFreeOwnedFn>(free_fns[i]);
        t = at_npu::native::from_blob(data, shapes[i], [ctx, fn](void *) { fn(ctx); }, options, device);
      } else {
        t = at_npu::native::from_blob(data, shapes[i], device, options);
      }
      // from_blob succeeded: ownership of record i now lives in the tensor.
      tensors.push_back(std::move(t));
    }
  } catch (...) {
    ReleaseRange(owned_ctxs, free_fns, i);
    throw;
  }
  return tensors;
}

}  // namespace

PYBIND11_MODULE(_fxrt_torch_bridge, m) {
  m.doc() = "FxConverter ABI=1 bridge: adopt GE-allocated device memory into torch NPU tensors";
  m.def("wrap_ge_outputs", &WrapGeOutputs, py::arg("data_ptrs"), py::arg("shapes"), py::arg("dtypes"),
        py::arg("owned_ctxs"), py::arg("free_fns"), py::arg("device_index"));
  m.def("fallback_api_address", []() { return reinterpret_cast<uintptr_t>(&kFallbackApi); });
  m.def("python_call_api_address", []() { return reinterpret_cast<uintptr_t>(&kPythonCallApi); });
}
