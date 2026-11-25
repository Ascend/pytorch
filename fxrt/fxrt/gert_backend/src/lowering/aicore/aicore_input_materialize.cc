// AICore parameters stay on host for tiling and CopyFlowLaunch. The launch wiring
// owns device transport; per-Execute expression refresh only updates host values.

#include "lowering/aicore/aicore_input_materialize.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "exe_graph/execute_graph_builder.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "lowering/aicore/aicore_op_work.h"
#include "lowering/common/view_utils.h"
#include "lowering/common/wiring.h"
#include "lowering/op_lowering.h"
#include "shape/shape_list_utils.h"

namespace fx_converter {
namespace {

// Materialize an IR scalar (Double/Int/Bool) into float / fp16 / bf16 bytes.
bool WriteFloatScalarBytes(const fxrt::ir::Value &val, ge::DataType dtype, void *dst, size_t nbytes,
                           std::string *err) {
  float f = 0.f;
  if (val.IsDouble()) {
    f = static_cast<float>(val.ToDouble());
  } else if (val.IsInt()) {
    f = static_cast<float>(val.ToInt());
  } else if (val.IsBool()) {
    f = val.ToBool() ? 1.f : 0.f;
  } else {
    if (err != nullptr) {
      *err = "aicore-binary lowering: unsupported scalar tag for float dtype";
    }
    return false;
  }
  if (nbytes < sizeof(float)) {
    // Scalar is materialized as a 16-bit device value; produce the correct bit pattern.
    uint16_t bits16 = 0;
    if (dtype == ge::DT_FLOAT16) {
      bits16 = Float32ToFp16Bits(f);
    } else if (dtype == ge::DT_BF16) {
      bits16 = Float32ToBf16Bits(f);
    }
    std::memcpy(dst, &bits16, sizeof(bits16));
  } else {
    std::memcpy(dst, &f, sizeof(float));
  }
  return true;
}

// Materialize an IR scalar (Bool/Int/Double) into a single bool byte.
bool WriteBoolScalarBytes(const fxrt::ir::Value &val, void *dst, std::string *err) {
  uint8_t b = 0;
  if (val.IsBool()) {
    b = val.ToBool() ? 1 : 0;
  } else if (val.IsInt()) {
    b = val.ToInt() != 0 ? 1 : 0;
  } else if (val.IsDouble()) {
    b = val.ToDouble() != 0.0 ? 1 : 0;
  } else {
    if (err != nullptr) {
      *err = "aicore-binary lowering: unsupported scalar tag for bool dtype";
    }
    return false;
  }
  std::memcpy(dst, &b, 1);
  return true;
}

// Convert IR scalar Value to int64 for integer dtype materialization.
bool IrScalarToInt64(const fxrt::ir::Value &val, int64_t *out, std::string *err) {
  if (out == nullptr) {
    return false;
  }
  if (val.IsInt()) {
    *out = val.ToInt();
    return true;
  }
  if (val.IsDouble()) {
    *out = static_cast<int64_t>(val.ToDouble());
    return true;
  }
  if (val.IsBool()) {
    *out = val.ToBool() ? 1 : 0;
    return true;
  }
  if (err != nullptr) {
    *err = "aicore-binary lowering: unsupported scalar tag for integer dtype";
  }
  return false;
}

// Materialize an IR scalar into INT8/UINT8/INT32/INT64 device bytes (eq/ge/ne int tensors).
bool WriteIntScalarBytes(const fxrt::ir::Value &val, ge::DataType dtype, void *dst, size_t nbytes,
                         std::string *err) {
  int64_t i64 = 0;
  if (!IrScalarToInt64(val, &i64, err)) {
    return false;
  }
  if (dtype == ge::DT_INT64) {
    if (nbytes < sizeof(int64_t)) {
      if (err != nullptr) {
        *err = "aicore-binary lowering: INT64 scalar buffer too small";
      }
      return false;
    }
    std::memcpy(dst, &i64, sizeof(int64_t));
    return true;
  }
  if (dtype == ge::DT_INT32) {
    if (nbytes < sizeof(int32_t)) {
      if (err != nullptr) {
        *err = "aicore-binary lowering: INT32 scalar buffer too small";
      }
      return false;
    }
    const int32_t i32 = static_cast<int32_t>(i64);
    std::memcpy(dst, &i32, sizeof(int32_t));
    return true;
  }
  if (dtype == ge::DT_INT8) {
    if (nbytes < 1U) {
      if (err != nullptr) {
        *err = "aicore-binary lowering: INT8 scalar buffer too small";
      }
      return false;
    }
    const int8_t i8 = static_cast<int8_t>(i64);
    std::memcpy(dst, &i8, 1);
    return true;
  }
  if (dtype == ge::DT_UINT8) {
    if (nbytes < 1U) {
      if (err != nullptr) {
        *err = "aicore-binary lowering: UINT8 scalar buffer too small";
      }
      return false;
    }
    const uint8_t u8 = static_cast<uint8_t>(i64);
    std::memcpy(dst, &u8, 1);
    return true;
  }
  if (err != nullptr) {
    *err = "aicore-binary lowering: integer scalar dtype not supported for materialization";
  }
  return false;
}

bool WriteScalarValueBytes(const fxrt::ir::Value &val, ge::DataType dtype, void *dst, size_t nbytes,
                           std::string *err) {
  if (dst == nullptr) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: invalid scalar destination";
    }
    return false;
  }
  if (dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
    return WriteFloatScalarBytes(val, dtype, dst, nbytes, err);
  }
  if (dtype == ge::DT_BOOL) {
    return WriteBoolScalarBytes(val, dst, err);
  }
  if (dtype == ge::DT_INT64 || dtype == ge::DT_INT32 || dtype == ge::DT_INT8 || dtype == ge::DT_UINT8) {
    return WriteIntScalarBytes(val, dtype, dst, nbytes, err);
  }
  if (err != nullptr) {
    *err = "aicore-binary lowering: scalar dtype not supported for materialization dtype=" +
           std::to_string(static_cast<int>(dtype));
  }
  return false;
}

// A dim the graph left symbolic must arrive with its expression. Without one there is nothing to
// evaluate at Execute, and the old fallback -- inferring a batch from the feed shapes -- is how
// ETA ended up sizing 128-row kernels to 64.
bool ValidateSymbolicAxes(const std::vector<int64_t> &dims, const std::vector<fxrt::ir::SymbolicExprPtr> *axis_exprs,
                          std::string *err) {
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] != kSymbolicDim) {
      continue;
    }
    if (axis_exprs == nullptr || axis_exprs->size() != dims.size() || (*axis_exprs)[i] == nullptr) {
      if (err != nullptr) {
        *err = "aicore-binary lowering: dynamic dim " + std::to_string(i) + " has no symbol to evaluate";
      }
      return false;
    }
  }
  return true;
}

bool ValidateShapeTensorArgs(const std::vector<int64_t> &dims, ge::DataType payload_dtype, std::string *err) {
  if (payload_dtype != ge::DT_INT64 && payload_dtype != ge::DT_INT32) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: shape-list payload must be INT32/INT64";
    }
    return false;
  }
  if (payload_dtype == ge::DT_INT32 && std::any_of(dims.begin(), dims.end(), [](int64_t d) {
        return d > std::numeric_limits<int32_t>::max() || d < std::numeric_limits<int32_t>::min();
      })) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: int_list value does not fit the kernel-required INT32 payload";
    }
    return false;
  }
  return true;
}

void *AllocateShapeHostPayload(BuiltExecuteGraph *built, const std::vector<int64_t> &dims,
                               ge::DataType payload_dtype) {
  auto sanitize = [](int64_t d) { return (d < 0) ? 1 : d; };
  if (payload_dtype == ge::DT_INT32) {
    auto host32 = std::make_shared<std::vector<int32_t>>();
    host32->reserve(dims.size());
    std::transform(dims.begin(), dims.end(), std::back_inserter(*host32),
                   [&](int64_t d) { return static_cast<int32_t>(sanitize(d)); });
    // An empty dimensions vector describes a scalar Fill output. Own a valid
    // address while retaining the logical [0] shape and zero payload byte count.
    if (host32->empty()) host32->push_back(0);
    void *host_ptr = host32->data();
    built->aicore_device_bufs.emplace_back(host_ptr, [host32](void *) mutable { host32.reset(); });
    return host_ptr;
  }
  auto host = std::make_shared<std::vector<int64_t>>();
  host->reserve(dims.size());
  std::transform(dims.begin(), dims.end(), std::back_inserter(*host), sanitize);
  if (host->empty()) host->push_back(0);
  void *host_ptr = host->data();
  built->aicore_device_bufs.emplace_back(host_ptr, [host](void *) mutable { host.reset(); });
  return host_ptr;
}

// Tiling StorageShape dims of a non-Broadcast shape list: the INT64 vector meta shape [rank], or
// a rank-0 scalar for TopKV2 k (GE const scalar, not a length-1 vector). Both are fixed at Convert
// time — rank never changes at runtime, and the list VALUES live in the payload blob, not here —
// so this goes into a plain Const; no refreshable-pointer indirection is needed.
// Broadcast targets (Expand) do not come here: they reuse the live InferShape output shape.
std::vector<int64_t> ShapeTilingMetaDims(const std::vector<int64_t> &dims, ge::DataType payload_dtype,
                                         bool force_vector) {
  if (payload_dtype == ge::DT_INT32 && dims.size() == 1U && !force_vector) {
    return {};
  }
  return {static_cast<int64_t>(dims.size())};
}

// Owns a copy of `value` on `built` and wires it as a Const node.
template <typename T>
fx_converter::Node *AddPodConst(ge::Graph *main, BuiltExecuteGraph *built, const std::string &name, const T &value,
                          uint64_t &sym_gen, std::vector<fx_converter::Node *> *kernel_nodes,
                          FxrtConstType const_type = FxrtConstType::kBytePtr) {
  auto blob = std::make_shared<std::vector<uint8_t>>(sizeof(T));
  std::memcpy(blob->data(), &value, sizeof(T));
  built->binary_bindata_blobs.push_back(blob);
  fx_converter::Node *node = AddConstNode(main, name, blob->data(), blob->size(), false, sym_gen, const_type);
  kernel_nodes->push_back(node);
  return node;
}

// Host parameter pair: same types as SplitDataTensor out0/out1. CopyFlowLaunch
// converts the address at the launch seam; no caller may pass it directly to device.
// InferShape still uses the packed Tensor* producer; tiling may use tiling_ep.
bool BindConstLaunchPair(ge::Graph *main, BuiltExecuteGraph *built, const std::string &tag, void *host,
                         size_t nbytes, const std::vector<int64_t> &meta_shape, uint64_t &sym_gen,
                         std::vector<fx_converter::Node *> *kernel_nodes, TensorRef &out, std::string *err) {
  const bool empty_tensor = std::find(meta_shape.begin(), meta_shape.end(), 0) != meta_shape.end();
  if (host == nullptr || (nbytes == 0 && !empty_tensor)) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: invalid Const Launch pair";
    }
    return false;
  }
  fx_converter::Node *shape_node = nullptr;
  uint64_t s_shape = 0;
  if (!AddConcreteStorageShapeConst(main, built, tag + "_meta", meta_shape, sym_gen, kernel_nodes, &shape_node,
                                    &s_shape, err)) {
    return false;
  }
  gert::GertTensorData gtd;
  (void)gtd.MutableTensorData().SetAddr(host, nullptr);
  gtd.SetSize(nbytes);
  gtd.SetPlacement(gert::kOnHost);
  fx_converter::Node *addr_node = AddPodConst(main, built, "const_gtd_host_" + tag, gtd, sym_gen, kernel_nodes,
                                               FxrtConstType::kGertTensorDataPtr);
  out.host_parameter = true;
  out.unpacked.shape.node = shape_node;
  out.unpacked.shape.out = 0;
  out.unpacked.shape.sym = s_shape;
  out.unpacked.addr.node = addr_node;
  out.unpacked.addr.out = 0;
  out.unpacked.addr.sym = addr_node->GetExtendInfo()->GetOutputSymbol(0);
  return true;
}

// A symbolic axis needs a per-Execute refresh of the host payload; a list of plain
// numbers never changes and needs none. The trigger is the presence of an expression, not the
// value -1: PyTorch's own -1 ("infer this dim", reshape(x, [-1])) is a number the kernel must
// receive verbatim.
// payload_dtype must be carried over: the refresh rewrites these very buffers, and their element
// width is whatever the call site chose (kGraphIntListDtype, or a kernel-forced INT32) -- assuming
// one here is what made a symbolic AsStrided size read back as [B,0,128].
void RegisterShapeRefreshIfSymbolic(BuiltExecuteGraph *built, const std::vector<int64_t> &dims,
                                    const std::vector<fxrt::ir::SymbolicExprPtr> *axis_exprs, void *host_ptr,
                                    void *dev, ge::DataType payload_dtype, bool broadcast_target) {
  if (axis_exprs == nullptr || axis_exprs->size() != dims.size() ||
      std::none_of(axis_exprs->begin(), axis_exprs->end(),
                   [](const fxrt::ir::SymbolicExprPtr &e) { return e != nullptr; })) {
    return;
  }
  BuiltExecuteGraph::AiCoreShapeTensorRefresh refresh;
  refresh.host = host_ptr;
  refresh.dev = dev;
  refresh.size_template = dims;
  refresh.payload_type = payload_dtype;
  refresh.axis_exprs = *axis_exprs;
  refresh.allow_nonpositive = !broadcast_target;
  built->aicore_shape_refreshes.push_back(std::move(refresh));
}

// Host BuildRefTensor over an existing StorageShape producer — the tiling view of a shape list.
// The StorageShape source is either a static Const (vector meta shape) or, for a Broadcast target,
// the live InferShape output.
fx_converter::Node *AddShapeHostRefTensor(ge::Graph *main, BuiltExecuteGraph *built, const std::string &suffix,
                                    uint64_t &sym_gen, std::vector<fx_converter::Node *> *kernel_nodes,
                                    fx_converter::Node *ss_node, int32_t ss_out_index, uint64_t s_ss, void *host_ptr,
                                    size_t nbytes, ge::DataType payload_dtype, uint64_t *s_host_tensor_out) {
  const ge::TensorDesc empty_td;
  gert::GertTensorData host_gtd;
  (void)host_gtd.MutableTensorData().SetAddr(host_ptr, nullptr);
  host_gtd.SetSize(nbytes);
  host_gtd.SetPlacement(gert::kOnHost);
  fx_converter::Node *c_host_gtd = AddPodConst(main, built, "const_shape_gtd_host_" + suffix, host_gtd, sym_gen,
                                         kernel_nodes, FxrtConstType::kGertTensorDataPtr);

  const TensorFieldEndpoints fields =
      AddTensorFieldConsts(main, suffix, payload_dtype, gert::kOnHost, ge::FORMAT_ND, sym_gen, kernel_nodes);
  if (!fields.Valid() || c_host_gtd == nullptr || c_host_gtd->GetExtendInfo() == nullptr) {
    return nullptr;
  }
  auto build_desc = std::make_shared<FxNodeSpec>("build_shape_host_" + suffix, "FxBuildRefTensor");
  for (int i = 0; i < 6; ++i) {
    build_desc->AddInputDesc(empty_td);
  }
  build_desc->AddOutputDesc(empty_td);
  fx_converter::Node *build_host = AddNode(main, build_desc);
  if (build_host == nullptr) {
    return nullptr;
  }
  const uint64_t s_host_tensor = sym_gen++;
  const TensorEndpoint ports[] = {{ss_node, ss_out_index, s_ss},
                                  {c_host_gtd, 0, c_host_gtd->GetExtendInfo()->GetOutputSymbol(0)},
                                  fields.dtype,
                                  fields.placement,
                                  fields.storage_format,
                                  fields.origin_format};
  std::vector<uint64_t> input_symbols;
  for (const TensorEndpoint &port : ports) {
    input_symbols.push_back(port.sym);
    AddDataEdge(main, port.node, port.out, build_host, static_cast<int32_t>(input_symbols.size() - 1U));
  }
  WireSymbols(build_host, input_symbols, {s_host_tensor});
  kernel_nodes->push_back(build_host);
  *s_host_tensor_out = s_host_tensor;
  return build_host;
}

// Tiling view of a non-Broadcast shape list (Tile multiples / Slice offsets+size / TopKV2 k /
// Transpose perm): a plain Const StorageShape feeding BuildRefTensor.
//
// The tiling side only ever reads the vector META shape ([rank] / rank-0), which is fixed at
// Convert time; the list VALUES reach tiling through the host payload the Tensor points at
// (OpInferDepends → GetInputConstData), and RefreshAiCoreShapeTensors rewrites that payload in
// place, at a fixed address. Nothing here changes between Executes, so the Const's Load-time heap
// copy is harmless and the refreshable StorageShape* + StorageShapeFromPtr detour is unnecessary.
// Being all-Const also lets CEM hoist the whole chain to Init — built once, not once per Execute.
fx_converter::Node *WireShapeHostBuildTensor(ge::Graph *main, BuiltExecuteGraph *built, const std::string &suffix,
                                       uint64_t &sym_gen, std::vector<fx_converter::Node *> *kernel_nodes,
                                       const std::vector<int64_t> &tiling_meta_dims, void *host_ptr, size_t nbytes,
                                       ge::DataType payload_dtype, uint64_t *s_host_tensor_out, std::string *err) {
  fx_converter::Node *c_ss = nullptr;
  uint64_t s_ss = 0;
  if (!AddConcreteStorageShapeConst(main, built, "shape_tiling_" + suffix, tiling_meta_dims, sym_gen, kernel_nodes,
                                    &c_ss, &s_ss, err)) {
    return nullptr;
  }
  return AddShapeHostRefTensor(main, built, suffix, sym_gen, kernel_nodes, c_ss, 0, s_ss, host_ptr, nbytes,
                               payload_dtype, s_host_tensor_out);
}

// Host Tensor Const plus the launch parameter pair; fill TensorRef outs.
bool FinalizeShapeTensorConst(ge::Graph *main, BuiltExecuteGraph *built, const std::string &suffix,
                              uint64_t &sym_gen, std::vector<fx_converter::Node *> *kernel_nodes,
                              const std::vector<int64_t> &dims, bool broadcast_target, ge::DataType payload_dtype,
                              bool force_vector, void *host_ptr, size_t nbytes, fx_converter::Node *build_host,
                              uint64_t s_host_tensor, TensorRef &out, bool defer_tiling, std::string *err) {
  const int64_t rank = static_cast<int64_t>(dims.size());
  const bool scalar_k = (!broadcast_target && payload_dtype == ge::DT_INT32 && dims.size() == 1U && !force_vector);
  gert::StorageShape shape = scalar_k ? gert::StorageShape({}, {}) : gert::StorageShape({rank}, {rank});
  gert::StorageFormat fmt(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType());
  gert::Tensor tensor(shape, fmt, gert::kOnHost, payload_dtype, host_ptr);
  fx_converter::Node *c_tensor = AddPodConst(main, built, "const_shape_host_input_" + suffix, tensor, sym_gen,
                                              kernel_nodes, FxrtConstType::kGertTensorPtr);

  out.producer = c_tensor;
  out.symbol = c_tensor->GetExtendInfo()->GetOutputSymbol(0);
  // Concrete shape lists are immutable across Execute calls. Use a host Tensor
  // Const for tiling and avoid StorageShapeFromPtr + BuildRefTensor. Dynamic
  // keep-dim lists retain the refreshed BuildRefTensor path.
  //
  // A deferred broadcast target also arrives with build_host == nullptr, but its tiling view is
  // the live InferShape output that BindBroadcastTilingViewsToInferShape wires later -- emitting
  // a host Const here would just be overwritten, so leave the slot empty.
  if (!defer_tiling && build_host == nullptr) {
    gert::StorageFormat host_fmt(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType());
    gert::Tensor host_tensor(shape, host_fmt, gert::kOnHost, payload_dtype, host_ptr);
    fx_converter::Node *c_host = AddPodConst(main, built, "const_shape_host_" + suffix, host_tensor, sym_gen,
                                              kernel_nodes, FxrtConstType::kGertTensorPtr);
    out.tiling_ep.node = c_host;
    out.tiling_ep.sym = c_host->GetExtendInfo()->GetOutputSymbol(0);
  } else {
    out.tiling_ep.node = build_host;
    out.tiling_ep.sym = s_host_tensor;
  }
  out.shape = scalar_k ? std::vector<int64_t>{} : std::vector<int64_t>{rank};
  out.dtype = payload_dtype;
  // Launch meta-shape of the payload tensor (rank-0 k, else [rank]) — not Expand's broadcast target.
  const std::vector<int64_t> meta = scalar_k ? std::vector<int64_t>{} : std::vector<int64_t>{rank};
  return BindConstLaunchPair(main, built, suffix, host_ptr, nbytes, meta, sym_gen, kernel_nodes, out, err);
}

}  // namespace

void BindBroadcastTilingViewsToInferShape(LoweringContext &ctx, AiCoreOpWork &work) {
  fx_converter::Node *infer = work.stage.InferNode();
  if (infer == nullptr) {
    return;
  }
  const uint64_t s_infer_shape = work.stage.OutShapeSymAt(0);
  for (size_t i = 0; i < work.ins.size(); ++i) {
    TensorRef &ref = work.ins[i];
    if (ref.pending_tiling_host_ptr == nullptr) {
      continue;
    }
    uint64_t s_host_tensor = 0;
    fx_converter::Node *build_host = AddShapeHostRefTensor(
        ctx.main, ctx.built, work.tag + "_bcast_" + std::to_string(i), *ctx.sym_gen, ctx.kernel_nodes, infer, 0,
        s_infer_shape, ref.pending_tiling_host_ptr, ref.pending_tiling_nbytes, ref.dtype, &s_host_tensor);
    ref.tiling_ep = TensorEndpoint{build_host, 0, s_host_tensor};
    ref.pending_tiling_host_ptr = nullptr;
    ref.pending_tiling_nbytes = 0;
  }
}

bool MaterializeShapeTensor(ge::Graph *main, BuiltExecuteGraph *built, const std::vector<int64_t> &dims,
                            const std::vector<fxrt::ir::SymbolicExprPtr> *axis_exprs, const std::string &tag,
                            size_t idx, uint64_t &sym_gen, std::vector<fx_converter::Node *> *kernel_nodes,
                            bool broadcast_target, ge::DataType payload_dtype, TensorRef &out, std::string *err,
                            bool force_vector, bool defer_tiling_view) {
  if (!ValidateShapeTensorArgs(dims, payload_dtype, err) || !ValidateSymbolicAxes(dims, axis_exprs, err)) {
    return false;
  }
  const size_t elem_size = (payload_dtype == ge::DT_INT32) ? sizeof(int32_t) : sizeof(int64_t);
  const size_t nbytes = dims.size() * elem_size;
  const std::string suffix = tag + "_" + std::to_string(idx);
  void *host_ptr = AllocateShapeHostPayload(built, dims, payload_dtype);
  RegisterShapeRefreshIfSymbolic(built, dims, axis_exprs, host_ptr, nullptr, payload_dtype, broadcast_target);
  // Plain numerical lists need no refreshed BuildRefTensor. FinalizeShapeTensorConst
  // wires a host Tensor Const for them, which CEM can hoist to Init.
  const bool dynamic_payload =
      axis_exprs != nullptr && std::any_of(axis_exprs->begin(), axis_exprs->end(),
                                         [](const fxrt::ir::SymbolicExprPtr &e) { return e != nullptr; });
  uint64_t s_host_tensor = 0;
  fx_converter::Node *build_host = nullptr;
  // Three tiling sources, mutually exclusive:
  //   defer_tiling_view  -> the live InferShape output, bound by BindBroadcastTilingViewsToInferShape
  //   dynamic payload    -> host BuildRefTensor over the fixed [rank] meta (values ride the payload)
  //   static payload     -> plain host Tensor Const, wired inside FinalizeShapeTensorConst
  if (!defer_tiling_view && dynamic_payload) {
    build_host = WireShapeHostBuildTensor(main, built, suffix, sym_gen, kernel_nodes,
                                          ShapeTilingMetaDims(dims, payload_dtype, force_vector), host_ptr, nbytes,
                                          payload_dtype, &s_host_tensor, err);
    if (build_host == nullptr) {
      return false;
    }
  }
  if (!FinalizeShapeTensorConst(main, built, suffix, sym_gen, kernel_nodes, dims, broadcast_target, payload_dtype,
                                force_vector, host_ptr, nbytes, build_host, s_host_tensor, out, defer_tiling_view,
                                err)) {
    return false;
  }
  if (defer_tiling_view) {
    out.pending_tiling_host_ptr = host_ptr;
    out.pending_tiling_nbytes = nbytes;
  }
  return true;
}

// Materialize IR Double/Int/Bool as a host parameter (Broadcast-friendly [1]).
// host_payload_out / dev_payload_out return the payload addresses when the caller registers
// a per-Execute refresh for a symbolic scalar (a runtime-resolved SymInt).
bool MaterializeScalarTensor(ge::Graph *main, BuiltExecuteGraph *built, const fxrt::ir::Value &scalar_value,
                             ge::DataType dtype, const std::string &tag, size_t idx, uint64_t &sym_gen,
                             std::vector<fx_converter::Node *> *kernel_nodes, TensorRef &out, std::string *err,
                             void **host_payload_out, void **dev_payload_out) {
  const size_t nbytes = static_cast<size_t>(ge::GetSizeByDataType(dtype));
  // Kept alive on `built`: the tiling view below reads this host payload.
  auto host = std::make_shared<std::vector<uint8_t>>(nbytes, 0);
  if (!WriteScalarValueBytes(scalar_value, dtype, host->data(), nbytes, err)) {
    return false;
  }
  void *host_ptr = host->data();
  built->aicore_device_bufs.emplace_back(host_ptr, [host](void *) mutable { host.reset(); });
  if (host_payload_out != nullptr) {
    *host_payload_out = host_ptr;
  }
  if (dev_payload_out != nullptr) {
    *dev_payload_out = nullptr;
  }

  gert::StorageShape shape({1}, {1});
  gert::StorageFormat fmt(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType());
  gert::Tensor tensor(shape, fmt, gert::kOnHost, dtype, host_ptr);
  fx_converter::Node *c_tensor =
      AddPodConst(main, built, "const_scalar_tensor_" + tag + "_" + std::to_string(idx), tensor, sym_gen,
                  kernel_nodes, FxrtConstType::kGertTensorPtr);
  out.producer = c_tensor;
  out.symbol = c_tensor->GetExtendInfo()->GetOutputSymbol(0);
  out.shape = {1};
  out.dtype = dtype;
  // Tiling view over the host payload: value-dependent tilings (e.g. Cumsum reads the
  // axis at tiling time) must not dereference the device producer address.
  gert::Tensor host_tensor(shape, fmt, gert::kOnHost, dtype, host_ptr);
  fx_converter::Node *c_host = AddPodConst(main, built, "const_scalar_tensor_host_" + tag + "_" + std::to_string(idx),
                                     host_tensor, sym_gen, kernel_nodes, FxrtConstType::kGertTensorPtr);
  out.tiling_ep = TensorEndpoint{c_host, 0, c_host->GetExtendInfo()->GetOutputSymbol(0)};
  return BindConstLaunchPair(main, built, "scalar_" + tag + "_" + std::to_string(idx), host_ptr, nbytes, {1}, sym_gen,
                             kernel_nodes, out, err);
}

}  // namespace fx_converter
