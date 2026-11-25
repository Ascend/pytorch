#include "exe_graph/infer_shape_fusion.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "common/logger.h"
#include "exe_graph/graph_builder_utils.h"
#include "exe_graph/fx_node_spec.h"
#include "shape/fx_fused_infer_shape.h"
#include "shape/fx_infer_func.h"
#include "shape/fx_shape_rule.h"
#include "shape/fx_symbol_env.h"
#include "ir/symbolic/symbolic.h"

namespace fx_converter {
namespace {

constexpr const char *kInferShapeType = "InferShape";
constexpr const char *kConstType = "Const";

// One original InferShape node and one rule per output slot.
struct InferSite {
  fx_converter::Node *node = nullptr;
  std::vector<const FxShapeRule *> rules;
  size_t out_num = 0;
  size_t first_slot = 0;
};

// Read back the 8-byte FxShapeRule* an AddConstNode Const carries.
const FxShapeRule *ReadRulePtrFromConst(const fx_converter::Node *const_node) {
  if (const_node == nullptr || const_node->GetTypePtr() == nullptr ||
      std::strcmp(const_node->GetTypePtr(), kConstType) != 0) {
    return nullptr;
  }
  std::vector<uint8_t> bytes;
  if (!const_node->BytesAttr("value", &bytes) || bytes.size() != sizeof(const FxShapeRule *)) {
    return nullptr;
  }
  const FxShapeRule *rule = nullptr;
  std::memcpy(&rule, bytes.data(), sizeof(rule));
  return rule;
}

fx_converter::Node *PeerInputNode(const fx_converter::Node *node, int32_t index) {
  return node == nullptr ? nullptr : node->InputProducer(index);
}

bool IsGraphFeed(const fx_converter::Node *node, int64_t *feed_index) {
  if (node == nullptr || feed_index == nullptr || node->GetTypePtr() == nullptr ||
      std::strcmp(node->GetTypePtr(), "Data") != 0) {
    return false;
  }
  return node->IntAttr("index", feed_index) && *feed_index >= 0;
}

ge::TensorDesc OutputDescOf(const fx_converter::Node *node, uint32_t index) {
  ge::TensorDesc desc;
  if (node == nullptr || !node->CopyOutputDesc(index, &desc)) {
    return ge::TensorDesc();
  }
  return desc;
}

std::vector<fx_converter::Node *> CollectGraphFeeds(ge::Graph *main) {
  std::vector<std::pair<int64_t, fx_converter::Node *>> feeds;
  for (auto *node : GetAllNodes(main)) {
    int64_t feed_index = -1;
    if (IsGraphFeed(node, &feed_index)) {
      feeds.emplace_back(feed_index, node);
    }
  }
  std::sort(feeds.begin(), feeds.end(), [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
  std::vector<fx_converter::Node *> result;
  result.reserve(feeds.size());
  for (size_t i = 0U; i < feeds.size(); ++i) {
    if (feeds[i].first != static_cast<int64_t>(i) || feeds[i].second->GetExtendInfo() == nullptr) {
      RT_GLOG(ERROR) << "FuseInferShape: graph feed indices are not contiguous at " << i;
      return {};
    }
    result.push_back(feeds[i].second);
  }
  return result;
}

size_t GetFallbackShapeInputCount(const BuiltExecuteGraph &built) {
  size_t count = 0U;
  for (const auto &call : built.torch_fallback_calls) {
    if (call != nullptr) {
      count = std::max(count, call->shape_input_count);
    }
  }
  return count;
}

// Rules sit immediately before infer_func. AiCore wires one FxShapeRule* per output
// ([data..., rule0..ruleM-1, infer_func]). Older AddInferShapeNode wires a single shared
// rule; that one pointer is repeated across every output slot.
bool ReadInferShapeRules(const fx_converter::Node *node, const std::unordered_set<const FxShapeRule *> &known,
                         std::vector<const FxShapeRule *> *rules) {
  if (node == nullptr) {
    return false;
  }
  const size_t in_num = node->InputCount();
  const size_t out_num = node->OutputCount();
  if (in_num < 2U || out_num == 0U) {
    return false;
  }
  std::vector<const FxShapeRule *> found;
  for (size_t back = 2U; back <= in_num && found.size() < out_num; ++back) {
    const auto *rule = ReadRulePtrFromConst(PeerInputNode(node, static_cast<int32_t>(in_num - back)));
    if (rule == nullptr || known.find(rule) == known.end()) {
      break;
    }
    found.push_back(rule);
  }
  if (found.size() == out_num) {
    std::reverse(found.begin(), found.end());
    *rules = std::move(found);
    return true;
  }
  if (found.size() == 1U) {
    rules->assign(out_num, found.front());
    return true;
  }
  return false;
}

// Collect every InferShape node with a rule that is genuinely one of ours. Anything unrecognised
// aborts the whole pass rather than fusing a subset -- a half-fused graph is harder to reason
// about than an unfused one, and this only ever runs on graphs we built ourselves.
bool CollectInferSites(ge::Graph *main, const BuiltExecuteGraph &built, std::vector<InferSite> *sites) {
  std::unordered_set<const FxShapeRule *> known;
  known.reserve(built.shape_rules.size());
  for (const auto &r : built.shape_rules) {
    known.insert(r.get());
  }

  for (fx_converter::Node *node : GetAllNodes(main)) {
    if (node == nullptr || node->GetTypePtr() == nullptr) {
      continue;
    }
    if (std::strcmp(node->GetTypePtr(), kInferShapeType) != 0) {
      continue;
    }
    InferSite site;
    site.node = node;
    if (!ReadInferShapeRules(node, known, &site.rules)) {
      RT_GLOG(WARNING) << "FuseInferShape: cannot recover per-slot rules for " << node->GetName()
                       << ", skipping fusion";
      return false;
    }
    site.out_num = site.rules.size();
    sites->push_back(std::move(site));
  }
  return true;
}

// Build the fused node: one input (the rules Const) and one output desc per original output,
// carrying the original output symbols so no consumer's symbol wiring has to change.
fx_converter::Node *BuildDeserializeInferFuncNode(ge::Graph *main, fx_converter::Node *const serialized_func,
                                             uint64_t &sym_gen) {
  if (main == nullptr || serialized_func == nullptr) {
    return nullptr;
  }
  const ge::TensorDesc empty_td;
  auto desc = std::make_shared<FxNodeSpec>("deserialize_fused_infer_func", "FxDeserializeInferFunc");
  if (!FxAttrUtils::SetStr(desc, kStageAttr, kStageInit)) {
    RT_GLOG(ERROR) << "FuseInferShape: failed to mark FxDeserializeInferFunc as Init";
    return nullptr;
  }
  (void)desc->AddInputDesc(empty_td);
  (void)desc->AddOutputDesc(empty_td);
  fx_converter::Node *node = AddNode(main, desc);
  if (node == nullptr) {
    return nullptr;
  }
  const uint64_t output_symbol = sym_gen++;
  WireSymbols(node, {serialized_func->GetExtendInfo()->GetOutputSymbol(0U)}, {output_symbol});
  if (AddDataEdge(main, serialized_func, 0, node, 0) != ge::GRAPH_SUCCESS) {
    return nullptr;
  }
  return node;
}

fx_converter::Node *BuildFusedNode(ge::Graph *main, const std::vector<InferSite> &sites, size_t total_slots,
                             size_t feed_shape_count, fx_converter::Node *const infer_func, uint64_t &sym_gen) {
  if (main == nullptr || infer_func == nullptr) {
    return nullptr;
  }
  auto desc = std::make_shared<FxNodeSpec>("fused_infer_shape", kFxFusedInferShapeType);
  (void)desc->AddInputDesc(ge::TensorDesc());
  const auto feeds = CollectGraphFeeds(main);
  for (auto *feed : feeds) {
    (void)desc->AddInputDesc(OutputDescOf(feed, 0U));
  }
  std::vector<uint64_t> out_syms;
  out_syms.reserve(total_slots + feed_shape_count);
  for (const auto &site : sites) {
    for (size_t j = 0; j < site.out_num; ++j) {
      (void)desc->AddOutputDesc(OutputDescOf(site.node, static_cast<uint32_t>(j)));
      out_syms.push_back(site.node->GetExtendInfo()->GetOutputSymbol(j));
    }
  }
  if (feed_shape_count > feeds.size()) {
    return nullptr;
  }
  for (size_t i = 0U; i < feed_shape_count; ++i) {
    (void)desc->AddOutputDesc(OutputDescOf(feeds[i], 0U));
    out_syms.push_back(sym_gen++);
  }
  fx_converter::Node *fused = AddNode(main, desc);
  if (fused == nullptr) {
    return nullptr;
  }
  // The fused node reuses the original output symbols, so no new ones are minted here.
  std::vector<uint64_t> input_symbols{infer_func->GetExtendInfo()->GetOutputSymbol(0U)};
  std::transform(feeds.begin(), feeds.end(), std::back_inserter(input_symbols),
                 [](const fx_converter::Node *feed) { return feed->GetExtendInfo()->GetOutputSymbol(0U); });
  WireSymbols(fused, input_symbols, out_syms);
  if (AddDataEdge(main, infer_func, 0, fused, 0) != ge::GRAPH_SUCCESS) {
    return nullptr;
  }
  for (size_t i = 0U; i < feeds.size(); ++i) {
    if (AddDataEdge(main, feeds[i], 0, fused, static_cast<int32_t>(i + 1U)) != ge::GRAPH_SUCCESS) {
      return nullptr;
    }
  }
  return fused;
}

bool RewireFallbackShapeInputs(ge::Graph *main, fx_converter::Node *fused,
                               const size_t feed_shape_slot_begin) {
  for (auto *node : GetAllNodes(main)) {
    if (node == nullptr || node == fused) {
      continue;
    }
    int64_t input_begin = -1;
    int64_t input_count = 0;
    if (!node->IntAttr("fx_fallback_shape_input_begin", &input_begin) ||
        !node->IntAttr("fx_fallback_shape_input_count", &input_count) ||
        input_begin < 0 || input_count < 0) {
      continue;
    }
    // Each destination slot is rewired once, so the original edge snapshot
    // remains sufficient. ES peer lookup resolves every source node; doing it
    // again for each shape input makes large fallback graphs quadratic here.
    const auto edges = GetInDataEdges(main, node);
    for (int64_t i = 0; i < input_count; ++i) {
      const int32_t dst_input = static_cast<int32_t>(input_begin + i);
      const auto existing = std::find_if(edges.begin(), edges.end(), [dst_input](const auto &edge) {
        return edge.dst_port == dst_input;
      });
      if (existing == edges.end() || RemoveDataEdge(main, existing->src, existing->src_port, node, dst_input) !=
                                          ge::GRAPH_SUCCESS ||
          AddDataEdge(main, fused, static_cast<int32_t>(feed_shape_slot_begin + static_cast<size_t>(i)), node,
                      dst_input) != ge::GRAPH_SUCCESS) {
        RT_GLOG(ERROR) << "FuseInferShape: failed to connect fallback shape input " << i << " for "
                       << node->GetName();
        return false;
      }
    }
  }
  return true;
}

// Move every consumer of (site.node, j) to (fused, slot). Data and control edges both move;
// dst input indices are preserved, so consumer symbol wiring stays valid.
bool MoveOutEdges(ge::Graph *main, const InferSite &site, fx_converter::Node *fused) {
  for (size_t j = 0; j < site.out_num; ++j) {
    const auto slot = static_cast<int32_t>(site.first_slot + j);
    const auto edges = GetOutDataEdges(main, site.node);
    for (const auto &edge : edges) {
      if (edge.src_port != static_cast<int32_t>(j)) {
        continue;
      }
      if (RemoveDataEdge(main, edge.src, edge.src_port, edge.dst, edge.dst_port) != ge::GRAPH_SUCCESS) {
        RT_GLOG(ERROR) << "FuseInferShape: RemoveEdge failed on " << site.node->GetName();
        return false;
      }
      if (AddDataEdge(main, fused, slot, edge.dst, edge.dst_port) != ge::GRAPH_SUCCESS) {
        RT_GLOG(ERROR) << "FuseInferShape: AddEdge to fused slot " << slot << " failed";
        return false;
      }
    }
  }
  // Control out-edges: the fused node inherits them so downstream ordering holds.
  const auto ctrl_nodes = GetOutControlNodes(main, site.node);
  for (auto *dst : ctrl_nodes) {
    if (RemoveControlEdge(main, site.node, dst) != ge::GRAPH_SUCCESS) {
      return false;
    }
    if (dst != fused && AddControlEdge(main, fused, dst) != ge::GRAPH_SUCCESS) {
      return false;
    }
  }
  return true;
}

// Drop all remaining edges into the node, then the node itself. Its rule / infer_func Consts are
// left to GE's pruner: they may still be shared, and a dead Const costs nothing in Main.
bool DropNode(ge::Graph *main, fx_converter::Node *node) {
  for (const auto &edge : GetInDataEdges(main, node)) {
    if (RemoveDataEdge(main, edge.src, edge.src_port, edge.dst, edge.dst_port) != ge::GRAPH_SUCCESS) {
      return false;
    }
  }
  for (auto *src : GetInControlNodes(main, node)) {
    if (RemoveControlEdge(main, src, node) != ge::GRAPH_SUCCESS) {
      return false;
    }
  }
  return RemoveNode(main, node) == ge::GRAPH_SUCCESS;
}

void DropFromKernelNodes(std::vector<fx_converter::Node *> *kernel_nodes,
                         const std::unordered_set<const fx_converter::Node *> &removed) {
  if (kernel_nodes == nullptr) {
    return;
  }
  kernel_nodes->erase(std::remove_if(kernel_nodes->begin(), kernel_nodes->end(),
                                     [&removed](const fx_converter::Node *n) { return removed.count(n) != 0U; }),
                      kernel_nodes->end());
}

void DropFromCniSpecs(std::vector<OpCniSpec> *cni_specs, const std::unordered_set<const fx_converter::Node *> &removed) {
  if (cni_specs == nullptr) {
    return;
  }
  for (auto &spec : *cni_specs) {
    spec.compute_nodes.erase(std::remove_if(spec.compute_nodes.begin(), spec.compute_nodes.end(),
                                            [&removed](const fx_converter::Node *n) { return removed.count(n) != 0U; }),
                             spec.compute_nodes.end());
  }
}

bool RemoveBindSymbolEnv(ge::Graph *main, fx_converter::Node *bind_symbols,
                         std::vector<fx_converter::Node *> *kernel_nodes,
                         std::vector<OpCniSpec> *cni_specs) {
  if (main == nullptr || bind_symbols == nullptr) {
    return true;
  }
  fx_converter::Node *const env_const = PeerInputNode(bind_symbols, 0);
  for (const auto &edge : GetOutDataEdges(main, bind_symbols)) {
    if (RemoveDataEdge(main, edge.src, edge.src_port, edge.dst, edge.dst_port) != ge::GRAPH_SUCCESS) {
      return false;
    }
  }
  for (auto *dst : GetOutControlNodes(main, bind_symbols)) {
    if (RemoveControlEdge(main, bind_symbols, dst) != ge::GRAPH_SUCCESS) {
      return false;
    }
  }
  if (!DropNode(main, bind_symbols)) {
    return false;
  }
  std::unordered_set<const fx_converter::Node *> removed{bind_symbols};
  if (env_const != nullptr && GetOutDataEdges(main, env_const).empty() &&
      GetOutControlNodes(main, env_const).empty()) {
    if (!DropNode(main, env_const)) {
      return false;
    }
    removed.insert(env_const);
  }
  DropFromKernelNodes(kernel_nodes, removed);
  DropFromCniSpecs(cni_specs, removed);
  return true;
}

}  // namespace

size_t FuseInferShapeNodes(BuiltExecuteGraph *built, fx_converter::Node *bind_symbols,
                           std::vector<fx_converter::Node *> *kernel_nodes, std::vector<OpCniSpec> *cni_specs,
                           uint64_t &sym_gen) {
  if (built == nullptr || built->main == nullptr) {
    return 0;
  }
  ge::Graph *main = built->main.get();

  std::vector<InferSite> sites;
  if (!CollectInferSites(main, *built, &sites)) {
    return 0;
  }
  const size_t feed_shape_count = GetFallbackShapeInputCount(*built);
  if (sites.empty() && feed_shape_count == 0U) {
    // Static fallback graphs have no shape work, but the temporary binder must
    // still be removed before handing the execution graph to GE.
    if (!RemoveBindSymbolEnv(main, bind_symbols, kernel_nodes, cni_specs)) {
      RT_GLOG(ERROR) << "FuseInferShape: failed to remove unused BindSymbolEnv";
    }
    RT_GLOG(INFO) << "FuseInferShape: " << sites.size() << " InferShape node(s), nothing to fuse";
    return 0;
  }
  const auto feeds = CollectGraphFeeds(main);
  if (feed_shape_count > feeds.size()) {
    RT_GLOG(ERROR) << "FuseInferShape: fallback shape input count exceeds graph feeds";
    return 0;
  }

  size_t total_slots = 0;
  auto fused_rules = std::make_shared<FxFusedShapeRules>();
  for (auto &site : sites) {
    site.first_slot = total_slots;
    fused_rules->slot_rules.insert(fused_rules->slot_rules.end(), site.rules.begin(), site.rules.end());
    total_slots += site.out_num;
  }
  const size_t feed_shape_slot_begin = total_slots;
  for (size_t i = 0U; i < feed_shape_count; ++i) {
    fused_rules->feed_shape_indices.push_back(i);
    ++total_slots;
  }

  built->fused_shape_rules = std::move(fused_rules);
  FxInferFuncArtifact infer_func_artifact;
  if (!BuildInferShapeSharedObject(*built->fused_shape_rules, &infer_func_artifact)) {
    RT_GLOG(WARNING) << "FuseInferShape: failed to build fused InferShape shared object";
    return 0;
  }
  std::vector<uint8_t> infer_func_payload;
  if (!SerializeInferFuncPayload(infer_func_artifact, &infer_func_payload)) {
    RT_GLOG(ERROR) << "FuseInferShape: failed to serialize fused InferShape shared object";
    return 0;
  }
  fx_converter::Node *c_infer_func = AddConstNode(main, "const_fused_infer_func", infer_func_payload.data(),
                                             infer_func_payload.size(), false, sym_gen);
  fx_converter::Node *deserialize = BuildDeserializeInferFuncNode(main, c_infer_func, sym_gen);
  if (deserialize == nullptr) {
    RT_GLOG(ERROR) << "FuseInferShape: failed to create FxDeserializeInferFunc";
    return 0;
  }
  fx_converter::Node *fused = BuildFusedNode(main, sites, total_slots, feed_shape_count, deserialize, sym_gen);
  if (fused == nullptr) {
    RT_GLOG(ERROR) << "FuseInferShape: failed to build fused node";
    return 0;
  }
  if (!RewireFallbackShapeInputs(main, fused, feed_shape_slot_begin)) {
    return 0;
  }

  std::unordered_set<const fx_converter::Node *> removed;
  removed.reserve(sites.size());
  const bool rewire_failed = std::any_of(sites.begin(), sites.end(),
                                         [main, fused](const InferSite &s) { return !MoveOutEdges(main, s, fused); });
  if (rewire_failed) {
    return 0;
  }
  for (const auto &site : sites) {
    if (!DropNode(main, site.node)) {
      RT_GLOG(ERROR) << "FuseInferShape: failed to drop " << site.node->GetName();
      return 0;
    }
    removed.insert(site.node);
  }

  if (!RemoveBindSymbolEnv(main, bind_symbols, kernel_nodes, cni_specs)) {
    RT_GLOG(ERROR) << "FuseInferShape: failed to remove BindSymbolEnv";
    return 0;
  }

  DropFromKernelNodes(kernel_nodes, removed);
  DropFromCniSpecs(cni_specs, removed);
  if (kernel_nodes != nullptr) {
    kernel_nodes->push_back(c_infer_func);
    kernel_nodes->push_back(deserialize);
    kernel_nodes->push_back(fused);
  }
  // Dump the finalized Main graph after the old per-op InferShape nodes have been removed.
  // ge::DumpGraph is a no-op
  // unless DUMP_GE_GRAPH is enabled; format, level, and output path remain controlled by GE's
  // existing DUMP_GRAPH_FORMAT / DUMP_GRAPH_LEVEL / DUMP_GRAPH_PATH settings.
  if (IsEsGraphRunnerEnabled() && main != nullptr) {
    (void)main->DumpToFile(ge::Graph::DumpFormat::kReadable, ge::AscendString("FxConverterAfterInferShapeFusion"));
  }
  RT_GLOG(INFO) << "FuseInferShape: " << sites.size() << " InferShape nodes -> 1 FxFusedInferShape with "
                << total_slots << " output slots (including " << feed_shape_count << " fallback shape carriers)";
  return sites.size();
}

}  // namespace fx_converter
