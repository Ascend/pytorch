#include "exe_graph/const_value_pass.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "common/logger.h"
#include "exe_graph/const_value_serialization.h"
#include "exe_graph/es_graph_builder.h"
#include "exe_graph/fx_node_spec.h"
#include "exe_graph/fxrt_kernel_registration.h"
#include "exe_graph/graph_builder_utils.h"
#include "ge/ge_ir_build.h"

namespace fx_converter {
namespace {

constexpr const char *kConstValueAttr = "value";
constexpr const char *kIsStringAttr = "is_string";

bool IsConst(const Node *node) {
  return node != nullptr && node->GetType() == "Const";
}

bool IsFxrtCustomKernel(const Node *node) {
  if (node == nullptr) {
    return false;
  }
  ge::AscendString attr_name(kCustomKernelNameAttr);
  ge::AscendString attr_value;
  return node->gnode().GetAttr(attr_name, attr_value) == ge::GRAPH_SUCCESS && attr_value.GetString() != nullptr &&
         attr_value.GetString()[0] != '\0';
}

void AppendKernelNode(std::vector<Node *> *kernel_nodes, Node *node) {
  if (kernel_nodes != nullptr && node != nullptr &&
      std::find(kernel_nodes->begin(), kernel_nodes->end(), node) == kernel_nodes->end()) {
    kernel_nodes->push_back(node);
  }
}

ge::graphStatus SetInputSymbol(Node *node, const int32_t input_index, const uint64_t symbol) {
  if (node == nullptr || input_index < 0 || static_cast<size_t>(input_index) >= node->InputCount()) {
    return ge::GRAPH_PARAM_INVALID;
  }
  std::vector<uint64_t> input_symbols;
  std::vector<uint64_t> output_symbols;
  input_symbols.reserve(node->InputCount());
  output_symbols.reserve(node->OutputCount());
  for (size_t i = 0U; i < node->InputCount(); ++i) {
    input_symbols.push_back(GetInputSymbol(node, i));
  }
  for (size_t i = 0U; i < node->OutputCount(); ++i) {
    output_symbols.push_back(GetOutputSymbol(node, i));
  }
  input_symbols[static_cast<size_t>(input_index)] = symbol;
  WireSymbols(node, input_symbols, output_symbols);
  return ge::GRAPH_SUCCESS;
}

std::shared_ptr<ge::ModelBufferData> MakeModelBufferData(const std::vector<uint8_t> &bytes) {
  try {
    auto model_buffer = std::make_shared<ge::ModelBufferData>();
    if (!bytes.empty()) {
      auto *const data = new uint8_t[bytes.size()];
      std::memcpy(data, bytes.data(), bytes.size());
      model_buffer->data = std::shared_ptr<uint8_t>(data, std::default_delete<uint8_t[]>());
    }
    model_buffer->length = static_cast<uint64_t>(bytes.size());
    return model_buffer;
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}

}  // namespace

ge::graphStatus RewriteTypedConstNodes(ge::Graph *graph, FxGraphBuilder *builder,
                                       std::vector<Node *> *kernel_nodes, uint64_t &sym_gen) {
  if (graph == nullptr || builder == nullptr || builder->GetGraph() == nullptr) {
    return ge::GRAPH_PARAM_INVALID;
  }
  static std::atomic<uint64_t> decoder_index{0U};
  const auto nodes = builder->GetAllNodes();
  for (auto *const constant : nodes) {
    if (!IsConst(constant)) {
      continue;
    }
    const auto all_edges = builder->GetOutDataEdges(constant);
    std::vector<FxGraphBuilder::DataEdgeInfo> custom_edges;
    custom_edges.reserve(all_edges.size());
    for (const auto &edge : all_edges) {
      if (IsFxrtCustomKernel(edge.dst)) {
        custom_edges.push_back(edge);
      }
    }
    if (custom_edges.empty()) {
      continue;
    }

    int64_t type_value = static_cast<int64_t>(FxrtConstType::kUnknown);
    if (!constant->IntAttr(kCustomConstTypeAttr, &type_value)) {
      RT_GLOG(ERROR) << "FXRT Const " << constant->GetName() << " has no construction-time type";
      return ge::GRAPH_PARAM_INVALID;
    }
    const auto type = static_cast<FxrtConstType>(type_value);
    bool is_string = false;
    (void)constant->BoolAttr(kIsStringAttr, &is_string);

    std::vector<uint8_t> source_bytes;
    if (!constant->BytesAttr(kConstValueAttr, &source_bytes)) {
      RT_GLOG(ERROR) << "FXRT Const " << constant->GetName() << " has no value bytes";
      return ge::GRAPH_PARAM_INVALID;
    }
    const bool has_non_custom_consumers = custom_edges.size() != all_edges.size();
    const uint64_t suffix = decoder_index.fetch_add(1U, std::memory_order_relaxed);
    const std::string tag = constant->GetName() + "_" + std::to_string(suffix);

    if (IsFxrtConstPassedThrough(type, is_string)) {
      if (!has_non_custom_consumers) {
        continue;
      }
      Node *const custom_copy = AddConstNode(graph, "fxrt_custom_string_" + tag, source_bytes.data(),
                                              source_bytes.size(), true, sym_gen, type, kStageInit);
      if (custom_copy == nullptr) {
        return ge::GRAPH_FAILED;
      }
      for (const auto &edge : custom_edges) {
        if (builder->RemoveDataEdge(edge.src, edge.src_port, edge.dst, edge.dst_port) != ge::GRAPH_SUCCESS ||
            builder->AddDataEdge(custom_copy, 0, edge.dst, edge.dst_port) != ge::GRAPH_SUCCESS ||
            SetInputSymbol(edge.dst, edge.dst_port, GetOutputSymbol(custom_copy, 0U)) != ge::GRAPH_SUCCESS) {
          return ge::GRAPH_FAILED;
        }
      }
      AppendKernelNode(kernel_nodes, custom_copy);
      continue;
    }

    std::vector<uint8_t> serialized;
    if (!SerializeFxrtConst(type, source_bytes.data(), source_bytes.size(), &serialized)) {
      RT_GLOG(ERROR) << "FXRT Const " << constant->GetName() << " cannot be serialized as FXRT type " << type_value;
      return ge::GRAPH_PARAM_INVALID;
    }
    const auto *const decoder_type = FxrtConstDecoderType(type);
    if (decoder_type == nullptr) {
      RT_GLOG(ERROR) << "FXRT Const " << constant->GetName() << " has no decoder for type " << type_value;
      return ge::GRAPH_PARAM_INVALID;
    }

    auto model_buffer = MakeModelBufferData(serialized);
    if (model_buffer == nullptr) {
      RT_GLOG(ERROR) << "Failed to allocate ModelBufferData for FXRT Const " << constant->GetName();
      return ge::GRAPH_FAILED;
    }
    ge::ModelBufferData *model_buffer_ptr = model_buffer.get();
    uint8_t model_buffer_ptr_bytes[sizeof(model_buffer_ptr)]{};
    std::memcpy(model_buffer_ptr_bytes, &model_buffer_ptr, sizeof(model_buffer_ptr));

    Node *serialized_constant = constant;
    if (has_non_custom_consumers) {
      serialized_constant = AddConstNode(graph, "fxrt_const_model_buffer_" + tag, model_buffer_ptr_bytes,
                                         sizeof(model_buffer_ptr), false, sym_gen, type, kStageInit);
      if (serialized_constant == nullptr) {
        return ge::GRAPH_FAILED;
      }
    }
    if (builder->RetainModelBufferData(model_buffer) != ge::GRAPH_SUCCESS ||
        (has_non_custom_consumers == false &&
         builder->SetNodeBytesAttr(serialized_constant, kConstValueAttr, model_buffer_ptr_bytes,
                                   sizeof(model_buffer_ptr)) != ge::GRAPH_SUCCESS) ||
        builder->SetNodeStringAttr(serialized_constant, kStageAttr, kStageInit) != ge::GRAPH_SUCCESS) {
      RT_GLOG(ERROR) << "Failed to attach ModelBufferData to FXRT Const " << constant->GetName();
      return ge::GRAPH_FAILED;
    }

    const ge::TensorDesc empty_desc;
    auto decoder_spec = std::make_shared<FxNodeSpec>("fxrt_deserialize_const_" + tag, decoder_type);
    decoder_spec->AddInputDesc(empty_desc);
    decoder_spec->AddOutputDesc(empty_desc);
    if (!FxAttrUtils::SetStr(decoder_spec, kStageAttr, kStageInit)) {
      return ge::GRAPH_FAILED;
    }
    auto *const decoder = builder->AddNode(decoder_spec);
    if (decoder == nullptr) {
      return ge::GRAPH_FAILED;
    }

    const uint64_t decoder_symbol = sym_gen++;
    WireSymbols(decoder, {GetOutputSymbol(serialized_constant, 0U)}, {decoder_symbol});
    AppendKernelNode(kernel_nodes, decoder);
    for (const auto &edge : custom_edges) {
      if (builder->RemoveDataEdge(edge.src, edge.src_port, edge.dst, edge.dst_port) != ge::GRAPH_SUCCESS ||
          builder->AddDataEdge(decoder, 0, edge.dst, edge.dst_port) != ge::GRAPH_SUCCESS ||
          SetInputSymbol(edge.dst, edge.dst_port, decoder_symbol) != ge::GRAPH_SUCCESS) {
        RT_GLOG(ERROR) << "Failed to redirect FXRT Const " << constant->GetName() << " to "
                       << edge.dst->GetName() << "[" << edge.dst_port << "]";
        return ge::GRAPH_FAILED;
      }
    }
    if (builder->AddDataEdge(serialized_constant, 0, decoder, 0) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    AppendKernelNode(kernel_nodes, serialized_constant);
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace fx_converter
