#include "exe_graph/execute_graph_metadata.h"
#include "exe_graph/es_graph_builder.h"

#include <algorithm>
#include <string>
#include <vector>

#include "common/logger.h"
#include "exe_graph/graph_builder_utils.h"
#include "graph/ascend_string.h"
#include "graph/attr_value.h"
#include "graph/graph.h"
#include "graph/tensor.h"
#include "graph/types.h"
#include "lowering/catalog_attrs.h"
#include "nlohmann/json.hpp"

namespace {

constexpr const char *kSourceOp = "gert.source_op";
constexpr const char *kModelInputs = "gert.model.inputs";
constexpr const char *kModelOutputs = "gert.model.outputs";
constexpr const char *kModelStreamCount = "gert.model.stream_count";
constexpr const char *kModelEventCount = "gert.model.event_count";
constexpr const char *kModelNotifyCount = "gert.model.notify_count";
constexpr const char *kModelAttachedStreamCount = "gert.model.attached_stream_count";

using Json = nlohmann::json;

// Match ge::TypeUtils::DataTypeToSerialString names used by gert.source_op v1.
// Keep a local table so the converter does not import TypeUtils from CANN stubs.
const char *DataTypeSerialName(ge::DataType dtype) {
  switch (dtype) {
    case ge::DT_FLOAT:
      return "DT_FLOAT";
    case ge::DT_FLOAT16:
      return "DT_FLOAT16";
    case ge::DT_INT8:
      return "DT_INT8";
    case ge::DT_INT16:
      return "DT_INT16";
    case ge::DT_UINT16:
      return "DT_UINT16";
    case ge::DT_UINT8:
      return "DT_UINT8";
    case ge::DT_INT32:
      return "DT_INT32";
    case ge::DT_INT64:
      return "DT_INT64";
    case ge::DT_UINT32:
      return "DT_UINT32";
    case ge::DT_UINT64:
      return "DT_UINT64";
    case ge::DT_BOOL:
      return "DT_BOOL";
    case ge::DT_DOUBLE:
      return "DT_DOUBLE";
    case ge::DT_STRING:
      return "DT_STRING";
    case ge::DT_BF16:
      return "DT_BFLOAT16";
    default:
      return "DT_UNDEFINED";
  }
}

Json MakePort(ge::DataType dtype, ge::Format fmt, bool exists = true) {
  Json port = Json::object();
  port["dtype"] = DataTypeSerialName(dtype);
  port["origin_format"] = static_cast<uint32_t>(fx_converter::OriginFormatOf(fmt));
  port["storage_format"] = static_cast<uint32_t>(fmt);
  if (!exists) {
    port["exists"] = false;
  }
  return port;
}

Json MakeIrMapping(size_t start, size_t count) {
  return Json::object({{"start", start}, {"count", count}});
}

Json MakeAttrJson(const fx_converter::ResolvedCatalogAttr &attr) {
  Json item = Json::object();
  item["name"] = attr.name;
  switch (attr.type) {
    case fx_converter::GeAttrType::kBool:
      item["type"] = "bool";
      item["value"] = attr.b;
      break;
    case fx_converter::GeAttrType::kInt:
      item["type"] = "int64";
      item["value"] = attr.i;
      break;
    case fx_converter::GeAttrType::kFloat:
      item["type"] = "float32";
      item["value"] = attr.f;
      break;
    case fx_converter::GeAttrType::kStr:
      item["type"] = "string";
      item["value"] = attr.s;
      break;
    case fx_converter::GeAttrType::kListInt:
      item["type"] = "list<int64>";
      item["value"] = Json::array({attr.i});
      break;
  }
  return item;
}

bool BuildSourceOpJson(const OpCniSpec &spec, size_t index, std::string *text) {
  if (text == nullptr) {
    return false;
  }
  Json root = Json::object();
  root["version"] = 1;
  if (!spec.compute_nodes.empty() && spec.compute_nodes.front() != nullptr) {
    root["name"] = spec.compute_nodes.front()->GetName();
  } else {
    root["name"] = "op_" + std::to_string(index);
  }
  root["type"] = spec.op_type;

  Json inputs = Json::array();
  for (size_t i = 0; i < spec.input_shapes.size(); ++i) {
    const ge::DataType dt =
        (i < spec.input_dtypes.size()) ? static_cast<ge::DataType>(spec.input_dtypes[i]) : ge::DT_FLOAT;
    const ge::Format fmt =
        (i < spec.input_formats.size()) ? static_cast<ge::Format>(spec.input_formats[i]) : ge::FORMAT_ND;
    inputs.push_back(MakePort(dt, fmt));
  }
  root["inputs"] = std::move(inputs);

  Json outputs = Json::array();
  Json ir_outputs = Json::array();
  if (!spec.output_shapes.empty()) {
    for (size_t i = 0; i < spec.output_shapes.size(); ++i) {
      const ge::DataType dt =
          (i < spec.output_dtypes.size()) ? static_cast<ge::DataType>(spec.output_dtypes[i]) : ge::DT_FLOAT;
      const ge::Format fmt =
          (i < spec.output_formats.size()) ? static_cast<ge::Format>(spec.output_formats[i]) : ge::FORMAT_ND;
      outputs.push_back(MakePort(dt, fmt));
      ir_outputs.push_back(MakeIrMapping(i, 1));
    }
  } else {
    const ge::Format out_fmt =
        spec.output_formats.empty() ? ge::FORMAT_ND : static_cast<ge::Format>(spec.output_formats.front());
    outputs.push_back(MakePort(static_cast<ge::DataType>(spec.output_dtype), out_fmt));
    ir_outputs.push_back(MakeIrMapping(0, 1));
  }
  root["outputs"] = std::move(outputs);
  root["ir_outputs"] = std::move(ir_outputs);

  Json ir_inputs = Json::array();
  if (spec.cni_io == fx_converter::CniIoStyle::kDynamicPackedX) {
    if (!spec.input_shapes.empty()) {
      ir_inputs.push_back(MakeIrMapping(0, spec.input_shapes.size()));
    }
  } else {
    for (size_t i = 0; i < spec.input_shapes.size(); ++i) {
      ir_inputs.push_back(MakeIrMapping(i, 1));
    }
  }
  if (spec.catalog != nullptr && spec.catalog->optional_inputs != nullptr &&
      spec.catalog->optional_inputs[0] != '\0') {
    const std::string csv(spec.catalog->optional_inputs);
    size_t begin = 0;
    while (begin < csv.size()) {
      const size_t comma = csv.find(',', begin);
      const size_t end = (comma == std::string::npos) ? csv.size() : comma;
      if (end > begin) {
        ir_inputs.push_back(MakeIrMapping(spec.input_shapes.size(), 0));
      }
      if (comma == std::string::npos) {
        break;
      }
      begin = comma + 1;
    }
  }
  root["ir_inputs"] = std::move(ir_inputs);

  std::vector<fx_converter::ResolvedCatalogAttr> attrs;
  if (!fx_converter::ResolveCatalogAttrs(spec.catalog, spec.axis, &spec.attrs, spec.input_shapes.size(), &attrs,
                                         spec.op_type.c_str())) {
    return false;
  }
  Json attrs_json = Json::array();
  for (const auto &attr : attrs) {
    attrs_json.push_back(MakeAttrJson(attr));
  }
  root["attrs"] = std::move(attrs_json);
  *text = root.dump();
  return true;
}

bool AttachSourceOpAttrs(fx_converter::FxGraphBuilder *builder, const std::vector<OpCniSpec> &op_specs) {
  if (builder == nullptr) {
    return false;
  }
  for (size_t i = 0; i < op_specs.size(); ++i) {
    const auto &spec = op_specs[i];
    std::string text;
    if (!BuildSourceOpJson(spec, i, &text)) {
      RT_GLOG(ERROR) << "Failed to build gert.source_op JSON for index " << i << " type=" << spec.op_type;
      return false;
    }
    for (fx_converter::Node *node : spec.compute_nodes) {
      if (node == nullptr) {
        continue;
      }
      if (builder->SetNodeStringAttr(node, kSourceOp, text.c_str()) != ge::GRAPH_SUCCESS) {
        RT_GLOG(ERROR) << "Failed to set gert.source_op for " << node->GetName();
        return false;
      }
    }
  }
  return true;
}

ge::Tensor MakeModelIoTensor(const ModelIoSpec &spec) {
  const std::vector<int64_t> &storage_dims = spec.storage_shape.empty() ? spec.shape : spec.storage_shape;
  ge::TensorDesc desc(ge::Shape(storage_dims), spec.format, static_cast<ge::DataType>(spec.data_type));
  desc.SetOriginFormat(fx_converter::OriginFormatOf(spec.format));
  desc.SetOriginShape(ge::Shape(spec.shape));
  // Name lives on the TensorDesc. Use the char* overload; SetName(std::string) is deprecated.
  desc.SetName(spec.name.c_str());
  return ge::Tensor(desc);
}

bool AttachModelDescAttrs(ge::Graph *graph, const std::vector<ModelIoSpec> &inputs,
                          const std::vector<ModelIoSpec> &outputs) {
  if (graph == nullptr) {
    return false;
  }
  std::vector<ge::Tensor> model_inputs;
  std::vector<ge::Tensor> model_outputs;
  model_inputs.reserve(inputs.size());
  model_outputs.reserve(outputs.size());
  for (const auto &spec : inputs) {
    model_inputs.push_back(MakeModelIoTensor(spec));
  }
  for (const auto &spec : outputs) {
    model_outputs.push_back(MakeModelIoTensor(spec));
  }

  ge::AttrValue inputs_attr;
  ge::AttrValue outputs_attr;
  ge::AttrValue stream_count;
  ge::AttrValue event_count;
  ge::AttrValue notify_count;
  ge::AttrValue attached_stream_count;
  if (inputs_attr.SetAttrValue(model_inputs) != ge::GRAPH_SUCCESS ||
      outputs_attr.SetAttrValue(model_outputs) != ge::GRAPH_SUCCESS ||
      stream_count.SetAttrValue(static_cast<int64_t>(kReusableStreamNum)) != ge::GRAPH_SUCCESS ||
      event_count.SetAttrValue(static_cast<int64_t>(0)) != ge::GRAPH_SUCCESS ||
      notify_count.SetAttrValue(static_cast<int64_t>(kReusableNotifyNum)) != ge::GRAPH_SUCCESS ||
      attached_stream_count.SetAttrValue(static_cast<int64_t>(0)) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "Failed to encode gert.model.* AttrValues";
    return false;
  }
  if (graph->SetAttr(ge::AscendString(kModelInputs), inputs_attr) != ge::GRAPH_SUCCESS ||
      graph->SetAttr(ge::AscendString(kModelOutputs), outputs_attr) != ge::GRAPH_SUCCESS ||
      graph->SetAttr(ge::AscendString(kModelStreamCount), stream_count) != ge::GRAPH_SUCCESS ||
      graph->SetAttr(ge::AscendString(kModelEventCount), event_count) != ge::GRAPH_SUCCESS ||
      graph->SetAttr(ge::AscendString(kModelNotifyCount), notify_count) != ge::GRAPH_SUCCESS ||
      graph->SetAttr(ge::AscendString(kModelAttachedStreamCount), attached_stream_count) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "Failed to set gert.model.* graph attributes";
    return false;
  }
  return true;
}

}  // namespace

bool AttachModelV2Metadata(ge::Graph *root, const std::vector<fx_converter::Node *> &kernel_nodes,
                           const std::vector<OpCniSpec> &op_specs, const std::vector<ModelIoSpec> &inputs,
                           const std::vector<ModelIoSpec> &outputs) {
  auto *builder = fx_converter::FxGraphBuilder::Lookup(root);
  if (root == nullptr || builder == nullptr || outputs.empty()) {
    return false;
  }

  // Helpers created after logical OPP CNI assembly still need a source_op. Fold
  // unassigned executable nodes into the first logical spec before emitting JSON.
  std::vector<OpCniSpec> specs = op_specs;
  if (specs.empty()) {
    OpCniSpec host;
    host.op_type = "Identity";
    host.output_shape = outputs.front().shape;
    host.output_dtype = outputs.front().data_type;
    specs.push_back(std::move(host));
  }
  if (!specs.empty()) {
    auto &fallback = specs.front();
    for (auto *const node : kernel_nodes) {
      if (node == nullptr) {
        continue;
      }
      const bool assigned = std::any_of(specs.cbegin(), specs.cend(), [node](const OpCniSpec &spec) {
        return std::find(spec.compute_nodes.cbegin(), spec.compute_nodes.cend(), node) != spec.compute_nodes.cend();
      });
      if (!assigned) {
        fallback.compute_nodes.push_back(node);
      }
    }
  }

  // KernelExtendInfo is generated by GE from node name/type; FXRT must not build
  // the pool or write KernelExtendIndex. CNI / ModelDesc are likewise prepared by
  // GE from gert.source_op and gert.model.* (see om2 3.2.3).
  if (!AttachSourceOpAttrs(builder, specs)) {
    return false;
  }
  return AttachModelDescAttrs(root, inputs, outputs);
}
