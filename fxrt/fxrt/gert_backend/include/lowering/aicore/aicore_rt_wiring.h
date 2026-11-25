#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "exe_graph/graph_builder_utils.h"
#include "lowering/op_lowering.h"

namespace fx_converter {

struct AiCoreBinArtifacts;
struct AiCoreOpWork;

// GE-isomorphic RT wiring state for one Path C op: InferShape + IO splits feed the
// tiling chain and Launch. Per-input / per-output endpoints live in the io vectors
// below (single source of truth); InferShape output symbols stay in s_shapes.
struct AiCoreStage {
  // One GE AfterSplit-style endpoint. Slot == RtGraphBuilder::Value == TensorEndpoint,
  // so stage endpoints, builder values and TensorRef endpoints are interchangeable.
  using Slot = RtGraphBuilder::Value;
  // Per-input wiring: the full Tensor* for FxAiCoreTiling (OpImpl tiling
  // reads GetInputTensor, not a bare StorageShape*; gert::Tensor starts with
  // StorageShape, so UpdateIOShapeToOp still works), plus the Launch pair
  // (addr=GertTensorData*, shape=StorageShape*).
  struct InputIo {
    Slot tensor;
    Slot addr;
    Slot shape;
  };
  // Per-output wiring: a final output boxes an OutputData Tensor*; an intermediate
  // output stays an (shape, addr) pair and leaves tensor null. A ref output borrows
  // an input address and does not own the allocation.
  struct OutputIo {
    Slot tensor;
    Slot addr;
    bool owns_addr = false;
  };

  fx_converter::Node* infer = nullptr;
  fx_converter::Node* alloc_arg = nullptr;
  fx_converter::Node* tiling = nullptr; // FxAiCoreTiling
  fx_converter::Node* tiling_parse = nullptr; // OpImpl TilingParse (needs CNI for GetNodeName)
  fx_converter::Node* sink_bin = nullptr;
  fx_converter::Node* launch = nullptr;
  std::vector<InputIo> ins;
  std::vector<OutputIo> outs;
  // One StorageShape symbol per InferShape output.
  std::vector<uint64_t> s_shapes;
  uint64_t s_rt_arg = 0;

  // Chain endpoints exposed to the entry / materialize side; fields stay
  // wiring-internal so only this header names the stage layout.
  fx_converter::Node* InferNode() const {
    return infer;
  }
  fx_converter::Node* TilingNode() const {
    return tiling;
  }
  fx_converter::Node* LaunchNode() const {
    return launch;
  }

  // StorageShape symbol of output o; falls back to output 0 when the InferShape node
  // emitted fewer symbols than declared outputs (0 when there are none).
  uint64_t OutShapeSymAt(size_t o) const {
    if (s_shapes.empty()) {
      return 0;
    }
    return (o < s_shapes.size()) ? s_shapes[o] : s_shapes.front();
  }

  // Shape endpoint of output o on the InferShape node (anchor = output o).
  Slot OutShapeSlotAt(size_t o) const {
    return Slot{infer, static_cast<int32_t>(o), OutShapeSymAt(o)};
  }

  // LoweringResult-facing output ref: final outputs keep OutputData as producer;
  // shape/addr always point at the InferShape output and the alloc.
  TensorRef OutputRef(size_t o, const std::vector<int64_t>& shape, ge::DataType dtype) const;

  // Nodes LoweringResult.cni.compute_nodes should report, in chain order.
  std::vector<fx_converter::Node*> ComputeNodes() const;

  // Intermediate output allocs that must be freed after downstream launches
  // (GE Guarder / last-consumer semantics): outputs whose final_mask slot is false.
  std::vector<fx_converter::Node*> IntermediateAllocs(const std::vector<bool>& final_mask) const;
};

// FxExprInferShape + one shape-rule Const per output slot -> InferShape node;
// bind_symbols anchors CNI.
void WireInferShape(
    RtGraphBuilder& gb,
    AiCoreOpWork& work,
    const std::vector<RtGraphBuilder::Value>& c_rules,
    RtGraphBuilder::Value c_infer,
    fx_converter::Node* bind_symbols);

// Input Split pairs + output Allocs, per output slot of final_mask:
// final slot -> OutputData + AllocModelOutTensor; intermediate slot -> AllocMemory only.
void WireIoAddrs(RtGraphBuilder& gb, AiCoreOpWork& work, LoweringContext& ctx);

// GE TilingCacheUtils::IsOpSupportTilingCache: bit i set when input i is data/tiling-value
// dependent (shape-list / host Const tensor). Origin-shape-only inputs stay 0.
size_t BuildTilingDataDependencyMask(const std::vector<TensorRef>& ins, int32_t shape_input_index);

// Append placeholder io slots for trailing optional inputs that stay unwired (catalog
// OptionalInputs). The kernel ABI reserves one io slot per proto input, so an absent
// trailing optional must still occupy its slot -- otherwise the kernel reads past the
// io array and faults on device. Mirrors GE gen_placeholder: null-addr GertTensorData,
// an int32(1) shape stand-in, and a null Tensor* for tiling.
void PadAiCoreOptionalInputs(RtGraphBuilder& gb, AiCoreOpWork& work);

// Launch io grows by work.optional.pad (GE gen_placeholder); tiling keeps the connected
// arity so ComputeNodeInfo slots stay aligned.
bool WireAiCoreExecute(RtGraphBuilder& gb, AiCoreOpWork& work, const SharedRuntime& rt, std::string* err);

// Graph-return outputs need FxEnsureTensorAtOutMemory after Launch so ExecuteAlloc /
// from_blob see the InferShape'd shape on the OutputData Tensor. Only slots whose
// final_mask entry is set get an Ensure node.
void WireAiCoreFinalOutputs(RtGraphBuilder& gb, AiCoreOpWork& work, const SharedRuntime& rt);

// True only when the platform SoC declares SoCInfo/ffts_mode = "ffts-plus" (GE
// PlatformUtils::GetFftsMode == FFTS_MODE_FFTS_PLUS). Gates the FFTS sync-addr args
// slot exactly like GE tbe_task_builder_adapter::InitInput: SoCs such as
// Ascend950PR_9599 (no ffts_mode) must not prepend it -- launch would call
// aclrtGetHardwareSyncAddr and fail with EE1005.
bool QueryPlatformRunsFftsPlus();

} // namespace fx_converter
