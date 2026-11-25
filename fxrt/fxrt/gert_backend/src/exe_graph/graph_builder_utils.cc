#include "exe_graph/graph_builder_utils.h"

#include <memory>
#include <string>
#include <utility>

#include "common/logger.h"
#include "exe_graph/es_graph_builder.h"

namespace fx_converter {
bool IsEsGraphRunnerEnabled() { return true; }

ge::Format OriginFormatOf(ge::Format storage_format) {
  switch (storage_format) {
    case ge::FORMAT_NC1HWC0:
    case ge::FORMAT_FRACTAL_Z:
    case ge::FORMAT_NC1HWC0_C04:
      return ge::FORMAT_NCHW;
    case ge::FORMAT_FRACTAL_NZ:
      return ge::FORMAT_ND;
    case ge::FORMAT_NDC1HWC0:
    case ge::FORMAT_FRACTAL_Z_3D:
    case ge::FORMAT_NDHWC:
      return ge::FORMAT_NCDHW;
    default:
      return storage_format;
  }
}

ge::TensorDesc MakeTensorDesc(const std::vector<int64_t> &shape, ge::DataType dtype, ge::Format storage_format) {
  ge::TensorDesc td(ge::Shape(shape), storage_format, dtype);
  td.SetOriginFormat(OriginFormatOf(storage_format));
  td.SetOriginShape(ge::Shape(shape));
  return td;
}

ge::TensorDesc MakeNdTensorDesc(const std::vector<int64_t> &shape, ge::DataType dtype) {
  return MakeTensorDesc(shape, dtype, ge::FORMAT_ND);
}

fx_converter::Node *AddFeedNode(ge::Graph *graph, const std::string &name, const char *node_type, int32_t index,
                          const ge::TensorDesc &td, uint64_t &sym_gen) {
  auto *builder = FxGraphBuilder::Lookup(graph);
  if (builder == nullptr) {
    return nullptr;
  }
  auto *node = builder->AddFeed(name, node_type, index, td);
  if (node != nullptr) {
    WireSymbols(node, {}, {sym_gen++});
  }
  return node;
}

fx_converter::Node *AddConstNode(ge::Graph *graph, const std::string &name, const void *data, size_t size,
                           bool is_string, uint64_t &sym_gen, FxrtConstType const_type, const char *stage) {
  auto *builder = FxGraphBuilder::Lookup(graph);
  if (builder == nullptr) {
    return nullptr;
  }
  auto *node = builder->AddConst(name, data, size, is_string, const_type, stage);
  if (node != nullptr) {
    WireSymbols(node, {}, {sym_gen++});
  }
  return node;
}

RtGraphBuilder::Value RtGraphBuilder::Const(const std::string &name, const void *data, size_t size, bool is_string,
                                            FxrtConstType const_type, const char *stage) {
  fx_converter::Node *node = AddConstNode(main_, name, data, size, is_string, sym_gen_, const_type, stage);
  kernel_nodes_->push_back(node);
  return Value{node, 0, node->GetExtendInfo()->GetOutputSymbol(0)};
}

RtGraphBuilder::NodeResult RtGraphBuilder::Node(const char *type, const std::string &name,
                                                const std::vector<Value> &ins, size_t out_count) {
  return TypedNode(type, name, ins, std::vector<ge::TensorDesc>(ins.size(), empty_td_),
                   std::vector<ge::TensorDesc>(out_count));
}

RtGraphBuilder::NodeResult RtGraphBuilder::TypedNode(const char *type, const std::string &name,
                                                     const std::vector<Value> &ins,
                                                     const std::vector<ge::TensorDesc> &td_ins,
                                                     const std::vector<ge::TensorDesc> &td_outs) {
  if (td_ins.size() != ins.size()) {
    RT_GLOG(ERROR) << "RtGraphBuilder::TypedNode " << name << " (" << type << "): " << ins.size()
                   << " inputs but " << td_ins.size() << " input descriptors";
    return NodeResult{};
  }
  auto desc = std::make_shared<FxNodeSpec>(name, type);
  for (size_t i = 0; i < ins.size(); ++i) {
    desc->AddInputDesc(td_ins[i]);
  }
  for (size_t o = 0; o < td_outs.size(); ++o) {
    desc->AddOutputDesc(td_outs[o]);
  }
  NodeResult result;
  result.node = AddNode(main_, desc);
  std::vector<uint64_t> in_syms(ins.size());
  std::transform(ins.begin(), ins.end(), in_syms.begin(), [](const Value &in) { return in.sym; });
  std::vector<uint64_t> out_syms(td_outs.size());
  result.outs.reserve(td_outs.size());
  for (size_t o = 0; o < td_outs.size(); ++o) {
    out_syms[o] = sym_gen_++;
    result.outs.push_back(Value{result.node, static_cast<int32_t>(o), out_syms[o]});
  }
  WireSymbols(result.node, in_syms, out_syms);
  for (size_t i = 0; i < ins.size(); ++i) {
    if (ins[i].node == nullptr) {
      RT_GLOG(ERROR) << "RtGraphBuilder::TypedNode " << name << " (" << type << "): null producer at input " << i;
      continue;
    }
    AddDataEdge(main_, ins[i].node, ins[i].out, result.node, static_cast<int32_t>(i));
  }
  kernel_nodes_->push_back(result.node);
  return result;
}

RtGraphBuilder::NodeResult RtGraphBuilder::SourcedNode(const char *type, const std::string &name,
                                                       const std::vector<Value> &ins, size_t out_count) {
  auto desc = std::make_shared<FxNodeSpec>(name, type);
  for (size_t i = 0; i < ins.size(); ++i) {
    desc->AddInputDesc(empty_td_);
  }
  for (size_t o = 0; o < out_count; ++o) {
    desc->AddOutputDesc(empty_td_);
  }
  NodeResult result;
  result.node = AddNode(main_, desc);
  std::vector<uint64_t> in_syms(ins.size());
  std::transform(ins.begin(), ins.end(), in_syms.begin(), [](const Value &in) { return in.sym; });
  std::vector<uint64_t> out_syms(out_count);
  result.outs.reserve(out_count);
  for (size_t o = 0; o < out_count; ++o) {
    out_syms[o] = sym_gen_++;
    result.outs.push_back(Value{result.node, static_cast<int32_t>(o), out_syms[o]});
  }
  WireSymbols(result.node, in_syms, out_syms);
  for (size_t i = 0; i < ins.size(); ++i) {
    if (ins[i].node == nullptr) {
      RT_GLOG(ERROR) << "RtGraphBuilder::SourcedNode " << name << " (" << type << "): null producer at input " << i;
      continue;
    }
    AddDataEdge(main_, ins[i].node, ins[i].out, result.node, static_cast<int32_t>(i));
  }
  kernel_nodes_->push_back(result.node);
  return result;
}

std::vector<int64_t> ShapeDims(const gert::Tensor *tensor) {
  if (tensor == nullptr) { return {}; }
  const auto &shape = tensor->GetStorageShape();
  std::vector<int64_t> dims(shape.GetDimNum());
  for (size_t i = 0; i < dims.size(); ++i) {
    dims[i] = shape.GetDim(static_cast<int64_t>(i));
  }
  return dims;
}
std::vector<int64_t> ShapeDims(const gert::Tensor &tensor) {
  return ShapeDims(&tensor);
}

}  // namespace fx_converter
