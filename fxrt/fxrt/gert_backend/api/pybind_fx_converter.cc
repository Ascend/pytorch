/**
 * Nanobind bindings for FxConverter.
 *
 * Used by the fx_converter Python package:
 *   convert(exported_dict) -> ConverterSession   (Convert + Init once)
 *   ConverterSession.execute_alloc(...)          (GE-allocated Execute, many times)
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "api/fx_allocator_bridge.h"
#include "api/fx_converter_api.h"
#include "api/torch_input_bridge.h"
#include "exe_graph/shared_const_store.h"
#include "ir_import/ir_graph_spec.h"
#include "ir_import/mrt_ir_graph_builder.h"
#include "lowering/torch_fallback/torch_fallback_runtime.h"
#include "register/op_catalog.h"
#include "register/op_catalog_register.h"

namespace nb = nanobind;

// Defined in the ABI=1 TU (torch_pool_allocator.cc). POD arguments only: this TU is
// ABI=0 and must not call at_npu itself. *rank < 0 means the storage shape was not written.
extern "C" int32_t FxcQueryFeedLayout(PyObject *obj, int64_t *storage_dims, int32_t cap, int32_t *rank);

namespace {

std::vector<void *> ToPtrs(const std::vector<uintptr_t> &addrs) {
  std::vector<void *> out;
  out.reserve(addrs.size());
  std::transform(addrs.begin(), addrs.end(), std::back_inserter(out),
                 [](uintptr_t a) { return reinterpret_cast<void *>(a); });
  return out;
}

std::vector<fx_converter::InputTensorLayout> ToLayouts(const std::vector<std::vector<int64_t>> &shapes,
                                                       const std::vector<std::vector<int64_t>> &strides,
                                                       const std::vector<std::vector<int64_t>> &storage_shapes,
                                                       const std::vector<int64_t> &storage_offsets,
                                                       const std::vector<int32_t> &formats) {
  if (shapes.size() != strides.size() || shapes.size() != storage_shapes.size() ||
      shapes.size() != storage_offsets.size() || (!formats.empty() && formats.size() != shapes.size())) {
    throw std::invalid_argument("input layout vectors must have equal length");
  }
  constexpr int32_t kFormatNd = 2;
  std::vector<fx_converter::InputTensorLayout> out(shapes.size());
  for (size_t i = 0; i < out.size(); ++i) {
    out[i].logical_shape = shapes[i];
    out[i].stride = strides[i];
    out[i].storage_shape = storage_shapes[i];
    out[i].storage_offset = storage_offsets[i];
    out[i].format = formats.empty() ? kFormatNd : formats[i];
  }
  return out;
}

IrNodeKind KindFromStr(const std::string &s) {
  if (s == "input") {
    return IrNodeKind::kInput;
  }
  if (s == "parameter") {
    return IrNodeKind::kParameter;
  }
  if (s == "return") {
    return IrNodeKind::kReturn;
  }
  return IrNodeKind::kOp;
}

IrValueSpec ValueSpecFromDict(const nb::dict &d) {
  IrValueSpec value;
  if (!d.contains("kind")) {
    return value;
  }
  const auto kind = nb::cast<std::string>(d["kind"]);
  if (kind == "ref") {
    value.kind = IrValueSpec::Kind::kRef;
    value.ref = nb::cast<std::string>(d["ref"]);
  } else if (kind == "int") {
    value.kind = IrValueSpec::Kind::kInt;
    value.int_value = nb::cast<int64_t>(d["value"]);
  } else if (kind == "double") {
    value.kind = IrValueSpec::Kind::kDouble;
    value.double_value = nb::cast<double>(d["value"]);
  } else if (kind == "bool") {
    value.kind = IrValueSpec::Kind::kBool;
    value.bool_value = nb::cast<bool>(d["value"]);
  } else if (kind == "string") {
    value.kind = IrValueSpec::Kind::kString;
    value.string_value = nb::cast<std::string>(d["value"]);
  } else if (kind == "symbol") {
    value.kind = IrValueSpec::Kind::kSymbol;
    value.string_value = nb::cast<std::string>(d["value"]);
  } else if (kind == "tuple") {
    value.kind = IrValueSpec::Kind::kTuple;
    if (d.contains("elements")) {
      const auto elements = nb::cast<nb::list>(d["elements"]);
      value.elements.reserve(elements.size());
      for (const auto &item : elements) {
        value.elements.push_back(ValueSpecFromDict(nb::cast<nb::dict>(item)));
      }
    }
  }
  return value;
}

template <typename T>
void AssignIfPresent(const nb::dict &node_dict, const char *key, T *field) {
  if (node_dict.contains(key)) {
    *field = nb::cast<T>(node_dict[key]);
  }
}

void SetOptionalScalar(const nb::dict &node_dict, IrNodeSpec *node) {
  if (!node_dict.contains("scalar")) {
    return;
  }
  node->has_scalar = true;
  node->scalar = nb::cast<double>(node_dict["scalar"]);
}

void SetOptionalNodeFields(const nb::dict &node_dict, IrNodeSpec *node) {
  AssignIfPresent(node_dict, "inputs", &node->inputs);
  AssignIfPresent(node_dict, "shape", &node->shape);
  AssignIfPresent(node_dict, "stride", &node->stride);
  AssignIfPresent(node_dict, "storage_shape", &node->storage_shape);
  AssignIfPresent(node_dict, "storage_offset", &node->storage_offset);
  AssignIfPresent(node_dict, "format", &node->format);
  AssignIfPresent(node_dict, "output_formats", &node->output_formats);
  AssignIfPresent(node_dict, "sym_shape", &node->sym_shape);
  AssignIfPresent(node_dict, "dtype", &node->dtype);
  // axis defaults to INT64_MIN (unset) on IrNodeSpec.
  AssignIfPresent(node_dict, "axis", &node->axis);
  AssignIfPresent(node_dict, "sym_axis", &node->sym_axis);
  SetOptionalScalar(node_dict, node);
  AssignIfPresent(node_dict, "extra_ints", &node->extra_ints);
  AssignIfPresent(node_dict, "extra_syms", &node->extra_syms);
  AssignIfPresent(node_dict, "int_list", &node->int_list);
  AssignIfPresent(node_dict, "sym_list", &node->sym_list);
  AssignIfPresent(node_dict, "bool_list", &node->bool_list);
  AssignIfPresent(node_dict, "output_shapes", &node->output_shapes);
  AssignIfPresent(node_dict, "output_dtypes", &node->output_dtypes);
  AssignIfPresent(node_dict, "output_sym_shapes", &node->output_sym_shapes);
  AssignIfPresent(node_dict, "overload", &node->overload);
}

void SetNodeArguments(const nb::dict &node_dict, IrNodeSpec *node) {
  if (!node_dict.contains("arguments")) {
    return;
  }
  const auto arguments = nb::cast<nb::list>(node_dict["arguments"]);
  node->arguments.reserve(arguments.size());
  for (const auto &argument_item : arguments) {
    node->arguments.push_back(ValueSpecFromDict(nb::cast<nb::dict>(argument_item)));
  }
}

void SetNodeAttrs(const nb::dict &node_dict, IrNodeSpec *node) {
  if (!node_dict.contains("attrs")) {
    return;
  }
  const auto attrs = nb::cast<nb::dict>(node_dict["attrs"]);
  for (const auto &attr_item : attrs) {
    const auto key = nb::cast<std::string>(attr_item.first);
    node->attrs.emplace(key, ValueSpecFromDict(nb::cast<nb::dict>(attr_item.second)));
  }
}

IrNodeSpec NodeSpecFromDict(const nb::dict &node_dict) {
  if (!node_dict.contains("name") || !node_dict.contains("kind")) {
    throw std::invalid_argument("IR node dict requires keys 'name' and 'kind'");
  }
  IrNodeSpec node;
  node.name = nb::cast<std::string>(node_dict["name"]);
  node.kind = KindFromStr(nb::cast<std::string>(node_dict["kind"]));
  node.op = node_dict.contains("op") ? nb::cast<std::string>(node_dict["op"]) : "";
  SetOptionalNodeFields(node_dict, &node);
  SetNodeArguments(node_dict, &node);
  SetNodeAttrs(node_dict, &node);
  return node;
}

// The IR export dict is a cross-repository contract, and its failure mode is silent: the
// reader below matches keys by name, so a field torch_npu renamed or stopped emitting
// reads back as that field's default rather than raising. Refuse an unrecognised version
// outright instead of mis-parsing one.
//
// FXC_EXPECTED_IR_SCHEMA_VERSION comes from cmake/torch_npu_fxrt_core.cmake, which also
// checks it against the torch_npu tree at configure time. This runtime check catches the
// other case: a converter built against one torch_npu and then run against another.
#ifndef FXC_EXPECTED_IR_SCHEMA_VERSION
#error "FXC_EXPECTED_IR_SCHEMA_VERSION must be defined by the build"
#endif

void CheckExportSchemaVersion(const nb::dict &exported) {
  if (!exported.contains("schema_version")) {
    throw std::invalid_argument(
        "export_ir_graph dict carries no 'schema_version'. That torch_npu predates the "
        "versioned IR export contract and is known to omit the tensor layout and symbol "
        "channel keys this converter requires; upgrade torch_npu.");
  }
  const auto version = nb::cast<int>(exported["schema_version"]);
  if (version != FXC_EXPECTED_IR_SCHEMA_VERSION) {
    throw std::invalid_argument(
        "IR export schema mismatch: this fx_converter reads version " +
        std::to_string(FXC_EXPECTED_IR_SCHEMA_VERSION) + " but torch_npu produced version " +
        std::to_string(version) +
        ". Rebuild fx_converter against the matching torch_npu source tree.");
  }
}

IrGraphSpec SpecFromExportedDict(const nb::dict &exported) {
  IrGraphSpec spec;
  CheckExportSchemaVersion(exported);
  if (!exported.contains("name") || !exported.contains("nodes")) {
    throw std::invalid_argument("export_ir_graph dict requires keys 'name' and 'nodes'");
  }
  spec.name = nb::cast<std::string>(exported["name"]);
  nb::list nodes = nb::cast<nb::list>(exported["nodes"]);
  spec.nodes.reserve(nodes.size());
  for (const auto &node_item : nodes) {
    spec.nodes.push_back(NodeSpecFromDict(nb::cast<nb::dict>(node_item)));
  }
  return spec;
}

std::vector<fx_converter::ExecutedOutput> PyConverterSessionExecuteAlloc(
    fx_converter::ConverterSession &session, const std::vector<uintptr_t> &input_addrs,
    const std::vector<std::vector<int64_t>> &input_shapes,
    const std::vector<std::vector<int64_t>> &input_strides,
    const std::vector<std::vector<int64_t>> &input_storage_shapes,
    const std::vector<int64_t> &input_storage_offsets, uintptr_t stream,
    const std::vector<std::pair<std::string, int64_t>> &scalar_syms,
    const std::vector<int32_t> &input_formats) {
  std::string err;
  std::vector<fx_converter::ExecutedOutput> outs;
  // GE and fallback kernels may enter Python on their own worker threads.
  nb::gil_scoped_release release;
  if (!session.ExecuteAlloc(
          ToPtrs(input_addrs),
          ToLayouts(input_shapes, input_strides, input_storage_shapes, input_storage_offsets, input_formats),
          reinterpret_cast<void *>(stream), &outs, &err, scalar_syms)) {
    throw std::runtime_error(err.empty() ? "ConverterSession.execute_alloc failed" : err);
  }
  return outs;
}

// Each CompiledModel owns one buffer. Capacity is reused, but every update reads
// all live addresses and metadata; no input identity or shape is assumed stable.
class TensorInputBuffer {
 public:
  void Update(nb::sequence tensors) {
    auto lock = Lock();
    UpdateLocked(tensors);
  }

  nb::dict Describe() {
    auto lock = Lock();
    if (!valid_) {
      throw std::runtime_error("tensor input buffer has no valid descriptors");
    }
    nb::list shapes, strides, storage_shapes, offsets, formats, addresses;
    for (size_t i = 0; i < layouts_.size(); ++i) {
      shapes.append(nb::cast(layouts_[i].logical_shape));
      strides.append(nb::cast(layouts_[i].stride));
      storage_shapes.append(nb::cast(layouts_[i].storage_shape));
      offsets.append(layouts_[i].storage_offset);
      formats.append(layouts_[i].format);
      addresses.append(reinterpret_cast<uintptr_t>(addresses_[i]));
    }
    nb::dict result;
    result["shapes"] = shapes;
    result["strides"] = strides;
    result["storage_shapes"] = storage_shapes;
    result["offsets"] = offsets;
    result["formats"] = formats;
    result["addresses"] = addresses;
    return result;
  }

  std::vector<fx_converter::ExecutedOutput> Execute(fx_converter::ConverterSession *session, nb::sequence tensors,
                                                    uintptr_t stream,
                                                    const std::vector<std::pair<std::string, int64_t>> &scalar_syms) {
    if (session == nullptr) {
      throw std::invalid_argument("a converter session is required");
    }
    auto lock = Lock();
    UpdateLocked(tensors);
    std::string error;
    std::vector<fx_converter::ExecutedOutput> outputs;
    nb::gil_scoped_release release;
    if (!session->ExecuteAlloc(addresses_, layouts_, reinterpret_cast<void *>(stream), &outputs, &error, scalar_syms)) {
      throw std::runtime_error(error.empty() ? "ConverterSession.execute failed" : error);
    }
    return outputs;
  }

 private:
  std::unique_lock<std::mutex> Lock() {
    // An executing thread reacquires the GIL before releasing this mutex.
    // Waiters must not hold the GIL while waiting for the same buffer.
    nb::gil_scoped_release release;
    return std::unique_lock<std::mutex>(mutex_);
  }

  void UpdateLocked(nb::sequence tensors) {
    valid_ = false;
    const size_t count = nb::len(tensors);
    layouts_.resize(count);
    descriptors_.resize(count);
    addresses_.resize(count);
    for (size_t i = 0; i < count; ++i) {
      auto &layout = layouts_[i];
      // Use the runtime's rank capacity rather than a model-specific limit.
      layout.logical_shape.resize(gert::Stride::kMaxDimNum);
      layout.stride.resize(gert::Stride::kMaxDimNum);
      layout.storage_shape.resize(gert::Stride::kMaxDimNum);
      auto &desc = descriptors_[i];
      desc.shape = layout.logical_shape.data();
      desc.stride = layout.stride.data();
      desc.storage_shape = layout.storage_shape.data();
      desc.capacity = gert::Stride::kMaxDimNum;
    }
    char error[1024] = {};
    if (!fx_converter::FxcCollectTensorInputs(tensors.ptr(), descriptors_.data(), count, error, sizeof(error))) {
      throw std::invalid_argument(error[0] == '\0' ? "failed to collect tensor inputs" : error);
    }
    for (size_t i = 0; i < count; ++i) {
      auto &layout = layouts_[i];
      const auto &desc = descriptors_[i];
      layout.logical_shape.resize(desc.rank);
      layout.stride.resize(desc.rank);
      layout.storage_shape.resize(desc.storage_rank);
      layout.storage_offset = desc.storage_offset;
      layout.format = desc.format;
      addresses_[i] = desc.address;
    }
    valid_ = true;
  }

  std::mutex mutex_;
  std::vector<fx_converter::InputTensorLayout> layouts_;
  std::vector<fx_converter::TorchInputDescriptor> descriptors_;
  std::vector<void *> addresses_;
  bool valid_ = false;
};

bool PyFreeOwned(uintptr_t owned_ctx, uintptr_t free_fn) {
  if (owned_ctx == 0 || free_fn == 0) {
    return false;
  }
  reinterpret_cast<fx_converter::FxFreeOwnedFn>(free_fn)(reinterpret_cast<void *>(owned_ctx));
  return true;
}

bool PyMemcpyD2D(uintptr_t dst, uintptr_t src, size_t nbytes) {
  std::string err;
  if (!fx_converter::ConverterSession::MemcpyDeviceToDevice(reinterpret_cast<void *>(dst),
                                                            reinterpret_cast<const void *>(src), nbytes, &err)) {
    throw std::runtime_error(err.empty() ? "memcpy_d2d failed" : err);
  }
  return true;
}

std::unique_ptr<fx_converter::ConverterSession> PyConvert(const nb::dict &exported, int32_t device_id,
                                                          uintptr_t stream) {
  const IrGraphSpec spec = SpecFromExportedDict(exported);
  auto graph = fx_converter::BuildMrtIrGraphFromSpec(spec);
  auto session = std::make_unique<fx_converter::ConverterSession>();
  std::string err;
  // GE initializes the TE Python runtime and manages the GIL internally.
  // Parse the Python dict above, then release the GIL for compile/load.
  nb::gil_scoped_release release;
  if (!session->Convert(graph, spec.name, device_id, reinterpret_cast<void *>(stream), &err)) {
    throw std::runtime_error(err.empty() ? "convert failed" : err);
  }
  return session;
}

bool HasKey(const nb::dict &spec, const char *key) { return spec.contains(key) && !spec[key].is_none(); }

std::string DictString(const nb::dict &spec, const char *key, const std::string &fallback) {
  return HasKey(spec, key) ? nb::cast<std::string>(spec[key]) : fallback;
}

bool DictBool(const nb::dict &spec, const char *key, bool fallback) {
  return HasKey(spec, key) ? nb::cast<bool>(spec[key]) : fallback;
}

int64_t DictInt(const nb::dict &spec, const char *key, int64_t fallback) {
  return HasKey(spec, key) ? nb::cast<int64_t>(spec[key]) : fallback;
}

void ReadCatalogAttrs(const nb::dict &spec, fx_converter::CatalogOpSpec *row) {
  if (!HasKey(spec, "attrs")) {
    return;
  }
  for (nb::handle item : nb::cast<nb::list>(spec["attrs"])) {
    const nb::dict attr = nb::cast<nb::dict>(item);
    fx_converter::CatalogAttrSpec one;
    one.name = DictString(attr, "name", "");
    one.src = DictString(attr, "src", "bool");
    one.literal = DictInt(attr, "literal", 0);
    one.text = DictString(attr, "text", "");
    one.required = DictBool(attr, "required", false);
    one.derived = DictString(attr, "derived", "");
    row->attrs.push_back(std::move(one));
  }
}

fx_converter::CatalogOpSpec CatalogSpecFromDict(const nb::dict &spec) {
  fx_converter::CatalogOpSpec row;
  row.logical_name = DictString(spec, "logical_name", "");
  row.opp_type = DictString(spec, "opp_type", "");
  row.arity = static_cast<size_t>(DictInt(spec, "arity", 2));
  row.lowering = DictString(spec, "lowering", "aicore");
  row.cni_io = DictString(spec, "cni_io", "indexed0");
  row.recipe = DictString(spec, "recipe", "tensor_io");
  row.shape_input_index = static_cast<int32_t>(DictInt(spec, "shape_input_index", -1));
  row.shape_list_broadcast_target = DictBool(spec, "shape_list_broadcast_target", true);
  row.opp_input_names = DictString(spec, "opp_input_names", "");
  row.opp_output_names = DictString(spec, "opp_output_names", "");
  row.opp_output_dtypes = DictString(spec, "opp_output_dtypes", "");
  row.shape_list_as_int32 = DictBool(spec, "shape_list_as_int32", false);
  row.capture_axis = DictBool(spec, "capture_axis", false);
  row.axis_attr = DictString(spec, "axis_attr", "");
  row.capture_topk_bools = DictBool(spec, "capture_topk_bools", false);
  row.capture_fill_value = DictBool(spec, "capture_fill_value", false);
  row.capture_skip_int = DictBool(spec, "capture_skip_int", false);
  row.capture_skip_bool = DictBool(spec, "capture_skip_bool", false);
  row.infer_transpose_from_layout = DictBool(spec, "infer_transpose_from_layout", false);
  row.prefer_high_performance = DictBool(spec, "prefer_high_performance", true);
  row.bool_as_int8 = DictBool(spec, "bool_as_int8", false);
  row.strided_inputs = DictBool(spec, "strided_inputs", false);
  row.select_suffix = DictString(spec, "select_suffix", "");
  row.infer_depends = DictString(spec, "infer_depends", "");
  row.optional_inputs = DictString(spec, "optional_inputs", "");
  ReadCatalogAttrs(spec, &row);
  return row;
}

std::pair<std::vector<std::optional<std::vector<int64_t>>>, std::vector<int32_t>> PyFeedLayouts(
    nb::sequence tensors) {
  const size_t count = nb::len(tensors);
  std::vector<std::optional<std::vector<int64_t>>> storage(count);
  std::vector<int32_t> formats(count, 2);
  constexpr int32_t kCap = 16;
  int64_t dims[kCap];
  for (size_t i = 0; i < count; ++i) {
    int32_t rank = 0;
    const nb::object item = tensors[i];
    formats[i] = FxcQueryFeedLayout(item.ptr(), dims, kCap, &rank);
    if (rank >= 0) {
      storage[i] = std::vector<int64_t>(dims, dims + rank);
    }
  }
  return {std::move(storage), std::move(formats)};
}

void PyRegisterCatalogOp(const nb::dict &spec) {
  const fx_converter::CatalogOpSpec row = CatalogSpecFromDict(spec);
  std::string err;
  if (!fx_converter::CatalogRegistry::Instance().Register(row, &err)) {
    throw std::runtime_error(err.empty() ? "register_catalog_op failed" : err);
  }
}

}  // namespace

NB_MODULE(_fxrt_fx_converter, m) {
  // The schema version this module reads, so Python can compare it against
  // fxrt._fxrt_ir.IR_EXPORT_SCHEMA_VERSION before building a graph.
  m.attr("IR_EXPORT_SCHEMA_VERSION") = FXC_EXPECTED_IR_SCHEMA_VERSION;
  m.doc() = "FxConverter: ABI=0 IR rebuild + ES graph compile and execute";
  m.def("register_torch_fallback_api", [](uintptr_t address) {
    std::string error;
    if (!fx_converter::RegisterTorchFallbackApi(address, &error)) {
      throw std::runtime_error(error.empty() ? "failed to register Torch fallback API" : error);
    }
  });
  m.def("register_python_call_api", [](uintptr_t address) {
    std::string error;
    if (!fx_converter::RegisterPythonCallApi(address, &error)) {
      throw std::runtime_error(error.empty() ? "failed to register python call API" : error);
    }
  });
  m.def("supported_logical_ops", []() { return fx_converter::ListConverterSupportedLogicalOps(); });

  nb::class_<fx_converter::OutputViewSpec>(m, "OutputViewSpec")
      .def_ro("output_index", &fx_converter::OutputViewSpec::output_index)
      .def_ro("preserve_layout", &fx_converter::OutputViewSpec::preserve_layout)
      .def_ro("shape", &fx_converter::OutputViewSpec::shape)
      .def_ro("stride", &fx_converter::OutputViewSpec::stride)
      .def_ro("storage_offset", &fx_converter::OutputViewSpec::storage_offset);
  nb::class_<fx_converter::ExecutedOutput>(m, "ExecutedOutput")
      .def_ro("shape", &fx_converter::ExecutedOutput::shape)
      .def_ro("storage_shape", &fx_converter::ExecutedOutput::storage_shape)
      .def_ro("format", &fx_converter::ExecutedOutput::format)
      .def_ro("dtype", &fx_converter::ExecutedOutput::dtype)
      .def_ro("device_addr", &fx_converter::ExecutedOutput::device_addr)
      .def_ro("nbytes", &fx_converter::ExecutedOutput::nbytes)
      .def_ro("stride", &fx_converter::ExecutedOutput::stride)
      .def_ro("logical_views", &fx_converter::ExecutedOutput::logical_views)
      // Ownership handles, passed to the ABI=1 torch bridge as plain integers. The
      // caller must invoke free_fn(owned_ctx) exactly once (normally as a torch deleter).
      .def_prop_ro("owned_ctx",
                   [](const fx_converter::ExecutedOutput &o) { return reinterpret_cast<uintptr_t>(o.owned_ctx); })
      .def_prop_ro("free_fn",
                   [](const fx_converter::ExecutedOutput &o) { return reinterpret_cast<uintptr_t>(o.free_fn); });

  nb::class_<fx_converter::ConverterSession>(m, "ConverterSession")
    .def("ready", &fx_converter::ConverterSession::ready)
    .def_prop_ro("es_runner_", &fx_converter::ConverterSession::has_es_runner)
    .def("has_fallback_calls", &fx_converter::ConverterSession::has_fallback_calls)
    .def("output_dtypes", &fx_converter::ConverterSession::output_dtypes)
    .def("release", &fx_converter::ConverterSession::release)
    .def("execute_alloc", &PyConverterSessionExecuteAlloc, nb::arg("input_addrs"), nb::arg("input_shapes"),
         nb::arg("input_strides"), nb::arg("input_storage_shapes"), nb::arg("input_storage_offsets"),
         nb::arg("stream") = 0, nb::arg("scalar_syms") = std::vector<std::pair<std::string, int64_t>>{},
         nb::arg("input_formats") = std::vector<int32_t>{});

  nb::class_<TensorInputBuffer>(m, "TensorInputBuffer")
    .def(nb::init<>())
    .def("update", &TensorInputBuffer::Update, nb::arg("tensors"),
         "Refresh live input descriptors while reusing their storage; retains no tensors.")
    .def("describe", &TensorInputBuffer::Describe, "Copy the last successful descriptors for inspection.")
    .def("execute", &TensorInputBuffer::Execute, nb::arg("session"), nb::arg("tensors"), nb::arg("stream") = 0,
         nb::arg("scalar_syms") = std::vector<std::pair<std::string, int64_t>>{},
         "Atomically refresh descriptors and execute the session with these inputs.");

  // Escape hatch: release an ExecutedOutput the caller decided not to wrap.
  m.def("free_owned", &PyFreeOwned, nb::arg("owned_ctx"), nb::arg("free_fn"));

  m.def("memcpy_d2d", &PyMemcpyD2D, nb::arg("dst"), nb::arg("src"), nb::arg("nbytes"));

  // One get_npu_format per feed, together with the storage shape that format implies.
  // A None storage shape means the query failed; the caller falls back to the logical shape.
  m.def("feed_layouts", &PyFeedLayouts, nb::arg("tensors"));

  // Convert + Init once; returns a session for repeated execute_alloc (torch.compile path).
  // Device memory is wired to torch_npu's caching allocator inside Convert (see
  // FxRegisterTorchPoolAllocator), so no Python-side argument is needed.
  m.def("convert", &PyConvert, nb::arg("exported"), nb::arg("device_id") = 0, nb::arg("stream") = 0);

  // Cube precision switches, mirrored from torch.npu.{conv,matmul}.allow_hf32.
  m.def("set_allow_hf32", &fx_converter::SetAllowHf32, nb::arg("conv"), nb::arg("matmul"));

  // Overlay a catalog row. Takes effect on the next convert; no rebuild.
  m.def("register_catalog_op", &PyRegisterCatalogOp, nb::arg("spec"));

  // Logical names Convert lowers. The op-support hook keeps these and demotes the rest.
  m.def("supported_logical_ops", []() { return fx_converter::ListConverterSupportedLogicalOps(); });
}
