// AICore RT wiring: InferShape -> input Split pairs / output Allocs ->
// FxAiCoreTiling -> FxAiCoreLaunch. Mix uses the same kernels.

#include "lowering/aicore/aicore_rt_wiring.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "acl/acl_rt.h"
#include "base/runtime/runtime_attrs_def.h"
#include "common/logger.h"
#include "exe_graph/graph_builder_utils.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "lowering/aicore/aicore_bin_artifacts.h"
#include "lowering/aicore/aicore_op_work.h"
#include "lowering/aicore/aicore_rt_layout.h"
#include "lowering/common/view_utils.h"
#include "lowering/common/wiring.h"
#include "opp/opp_registry.h"
#include "platform/platform_info.h"
#include "runtime/rt.h"

namespace fx_converter {
namespace {

namespace lyt = aicore_rt;

// Mirror the newer CANN definition. These bytes are serialized into GE's launch graph.
enum AclRtLaunchKernelAttrIdCompat : int32_t {
  kAclRtLaunchKernelAttrSchemMode = 1,
  kAclRtLaunchKernelAttrDynUbufSize = 2,
  kAclRtLaunchKernelAttrEngineType = 3,
  kAclRtLaunchKernelAttrBlockDimOffset = 4,
  kAclRtLaunchKernelAttrBlockTaskPrefetch = 5,
  kAclRtLaunchKernelAttrDataDump = 6,
  kAclRtLaunchKernelAttrTimeout = 7,
  kAclRtLaunchKernelAttrTimeoutUs = 8,
  kAclRtLaunchKernelAttrEnableProfiling = 9,
};

struct AclRtTimeoutUsCompat {
  uint32_t timeoutLow;
  uint32_t timeoutHigh;
};

union AclRtLaunchKernelAttrValueCompat {
  uint8_t schemMode;
  uint32_t localMemorySize;
  uint32_t dynUBufSize;
  aclrtEngineType engineType;
  uint32_t blockDimOffset;
  uint8_t isBlockTaskPrefetch;
  uint8_t isDataDump;
  uint16_t timeout;
  AclRtTimeoutUsCompat timeoutUs;
  uint8_t enableProfiling;
  uint32_t rsv[4];
};

struct AclRtLaunchKernelAttrCompat {
  AclRtLaunchKernelAttrIdCompat id;
  AclRtLaunchKernelAttrValueCompat value;
};

static_assert(sizeof(AclRtLaunchKernelAttrIdCompat) == sizeof(aclrtLaunchKernelAttrId));
static_assert(sizeof(AclRtLaunchKernelAttrValueCompat) == sizeof(aclrtLaunchKernelAttrValue));
static_assert(alignof(AclRtLaunchKernelAttrCompat) == alignof(aclrtLaunchKernelAttr));
static_assert(offsetof(AclRtLaunchKernelAttrCompat, value) == offsetof(aclrtLaunchKernelAttr, value));
static_assert(sizeof(AclRtLaunchKernelAttrCompat) == sizeof(aclrtLaunchKernelAttr));

// Enum-indexed input slot; keeps the GE input-order enum names at each assignment.
template <typename E>
size_t Idx(E e) {
  return static_cast<size_t>(e);
}

void WireAiCoreInputSplits(ge::Graph *main, const std::string &tag, const std::vector<TensorRef> &ins,
                           uint64_t &sym_gen, std::vector<fx_converter::Node *> *kernel_nodes, AiCoreStage &stage,
                           LoweringContext &ctx) {
  stage.ins.clear();
  stage.ins.reserve(ins.size());
  stage.outs.clear();

  for (size_t i = 0; i < ins.size(); ++i) {
    AiCoreStage::InputIo io;
    // FxAiCoreTiling reads GetInputTensor (gert::Tensor*), not a
    // bare StorageShape*. ViewAlias only forwards (shape, addr); box at this ABI
    // boundary so Expand broadcast tiling can see the view's storage shape
    // (ETA: unsqueeze(topk_idx).expand -> otherwise "add input shapes failed").
    if (ins[i].tiling_ep.Valid()) {
      io.tensor.node = ins[i].tiling_ep.node;
      io.tensor.sym = ins[i].tiling_ep.sym;
    } else {
      TensorRef boxed = ins[i];
      if (EnsureTensorRefBoxed(&boxed, main, tag + "_tiling_in" + std::to_string(i), sym_gen, kernel_nodes) &&
          boxed.producer != nullptr) {
        io.tensor.node = boxed.producer;
        io.tensor.sym = boxed.symbol;
      } else if (ins[i].HasUnpackedPair()) {
        io.tensor.node = ins[i].unpacked.shape.node;
        io.tensor.sym = ins[i].unpacked.shape.sym;
      } else {
        io.tensor.node = ins[i].producer;
        io.tensor.sym = ins[i].symbol;
      }
    }

    // GE AfterSplit: Launch/Mix both take (addr=GertTensorData*, shape=StorageShape*).
    // AllocMemory OutputsCreator = GertTensorData; InferShape OutputsCreator = Tensor
    // (starts as StorageShape). Same types as SplitDataTensor out1/out0. Graph
    // inputs already have a Split pair from UnpackGraphInputTensors.
    if (ins[i].HasUnpackedPair()) {
      io.addr = ins[i].unpacked.addr;
      io.shape = ins[i].unpacked.shape;
    } else {
      fx_converter::Node *split = GetOrCreateSplitDataTensor(ctx, ins[i], "in_" + tag, i, &io.shape.sym, &io.addr.sym);
      io.addr.node = split;
      io.addr.out = 1;
      io.shape.node = split;
      io.shape.out = 0;
    }
    stage.ins.push_back(io);
  }
}

// SharedRuntime endpoints as builder values (Init-once InnerData / feed nodes).
struct RtValues {
  RtGraphBuilder::Value stream;      // split_streams out0: logic stream 0 (main)
  RtGraphBuilder::Value sub_stream;  // split_streams out1: Mix VectorCore sub-task
  RtGraphBuilder::Value notify0;     // create_notifies out0
  RtGraphBuilder::Value notify1;     // create_notifies out1
  RtGraphBuilder::Value allocator;   // select_l2 out0
  RtGraphBuilder::Value platform;    // inner_platform out0 (Init-once GetPlatformInfo)
  RtGraphBuilder::Value core_num;    // inner_core_num out0
  bool has_mix_resources = false;
};

RtValues MakeRtValues(const SharedRuntime &rt) {
  return RtValues{RtGraphBuilder::From(rt.split_streams, 0, rt.s_stream),
                  RtGraphBuilder::From(rt.split_streams, 1, rt.s_sub_stream),
                  RtGraphBuilder::From(rt.create_notifies, 0, rt.s_notify0),
                  RtGraphBuilder::From(rt.create_notifies, 1, rt.s_notify1),
                  RtGraphBuilder::From(rt.select_l2, 0, rt.s_allocator),
                  RtGraphBuilder::From(rt.inner_platform, 0, rt.s_platform),
                  RtGraphBuilder::From(rt.inner_core_num, 0, rt.s_core_num),
                  rt.has_mix_resources};
}

// One output alloc: final -> OutputData + AllocModelOutTensor; intermediate -> AllocMemory.
// A ref output borrows the input address and writes that storage in place.
void WireAiCoreOneOutput(RtGraphBuilder &gb, const std::string &osuffix, bool is_final,
                         const ge::TensorDesc &td_out, RtGraphBuilder::Value allocator,
                         RtGraphBuilder::Value size, AiCoreStage &stage, size_t o) {
  if (is_final) {
    stage.outs[o].tensor = gb.TypedNode("OutputData", "user_output_" + osuffix, {}, {}, {td_out}).outs.front();
    stage.outs[o].addr =
        gb.Node("AllocModelOutTensor", "alloc_model_out_" + osuffix, {allocator, size, stage.outs[o].tensor}, 1)
            .outs.front();
  } else {
    stage.outs[o].addr = gb.Node("AllocMemory", "alloc_" + osuffix, {allocator, size}, 1).outs.front();
    // Intermediate AiCore outputs are already represented by (shape, address).
    // Keep the pair for downstream Launch/Tiling and avoid a boxed BuildTensor.
    stage.outs[o].tensor.node = nullptr;
    stage.outs[o].tensor.sym = 0;
  }
  stage.outs[o].owns_addr = true;
}

void WireAiCoreRefOutput(AiCoreStage &stage, size_t o, int32_t ref_input) {
  stage.outs[o].addr = stage.ins[static_cast<size_t>(ref_input)].addr;
  stage.outs[o].tensor.node = nullptr;
  stage.outs[o].tensor.sym = 0;
  stage.outs[o].owns_addr = false;
}

void WireAiCoreOutputs(RtGraphBuilder &gb, const std::string &tag, const std::vector<bool> &final_mask,
                       const RtValues &rtv, const std::vector<ge::TensorDesc> &td_outs,
                       const std::vector<ge::DataType> &out_dtypes, const OpCatalogEntry &entry,
                       AiCoreStage &stage) {
  const size_t nout = td_outs.size();
  stage.outs.resize(nout);
  if (stage.s_shapes.size() < nout) {
    stage.s_shapes.resize(nout, stage.OutShapeSymAt(0));
  }
  for (size_t o = 0; o < nout; ++o) {
    // Each catalog pair writes that output into the collected input. Later readers of
    // that storage see the write because compute nodes run in IR order.
    const int32_t ref_input = CatalogRefInput(entry, o);
    if (ref_input >= 0) {
      WireAiCoreRefOutput(stage, o, ref_input);
      continue;
    }
    const std::string osuffix = tag + ((nout > 1) ? ("_o" + std::to_string(o)) : "");
    const ge::DataType out_dtype = out_dtypes[o];
    const RtGraphBuilder::Value dtype =
        gb.Const("const_dtype_" + osuffix, &out_dtype, sizeof(out_dtype), false, FxrtConstType::kGeDataType);
    const RtGraphBuilder::Value shape = stage.OutShapeSlotAt(o);

    // Model outputs are allocated by GE from the InferShape'd size (ExecuteAlloc hands
    // AllocModelOutTensor a null address), so final and intermediate outputs size the
    // same way. dtype is this output's real dtype -- a wrong one under-allocates.
    const auto size = gb.SourcedNode("CalcTensorSizeFromStorage", "calc_size_" + osuffix, {dtype, shape}, 1);
    WireAiCoreOneOutput(gb, osuffix, o < final_mask.size() && final_mask[o], td_outs[o], rtv.allocator,
                        size.outs.front(), stage, o);
  }
}

// GE bg_kernel_launch MixType: MIX_VECTOR_CORE / MIX_AIV select vector mix.
// MIX enhance ("MIX" / "Mix") stays AICORE-style. taskRation is logged by the
// caller; fe GetCoreTypeAndRatio does not rewrite the enhance type from it.
aicore_rt::MixType MixTypeFromCoreIndex(aicore_rt::CoreTypeIndex idx) {
  if (idx == aicore_rt::CoreTypeIndex::kMixAiVector || idx == aicore_rt::CoreTypeIndex::kMixAiv) {
    return aicore_rt::MixType::kMixVectorCore;
  }
  return aicore_rt::MixType::kMixAiCore;
}

// FE fills GetCoreNum only after SetCoreNumByCoreType. GetCoreNumByType("MIX_VECTOR_CORE")
// returns the cube count on 910B, which is the wrong MixCoreArgs.all_core_num.
bool ReadMixCoreNums(fe::PlatFormInfos *infos, uint32_t *all_core, uint32_t *ai_core, uint32_t *vec_core) {
  infos->SetCoreNumByCoreType("MIX_VECTOR_CORE");
  *all_core = infos->GetCoreNum();
  infos->SetCoreNumByCoreType("AIC");
  *ai_core = infos->GetCoreNum();
  infos->SetCoreNumByCoreType("AIV");
  *vec_core = infos->GetCoreNum();
  if (*all_core > 0U && *ai_core > 0U && *vec_core > 0U) {
    return true;
  }
  infos->SetCoreNumByCoreType("Mix");
  *all_core = infos->GetCoreNum();
  infos->SetCoreNumByCoreType("AiCore");
  *ai_core = infos->GetCoreNum();
  infos->SetCoreNumByCoreType("VectorCore");
  *vec_core = infos->GetCoreNum();
  return *all_core > 0U && *ai_core > 0U && *vec_core > 0U;
}

// Live device first: a virtualized chip can have fewer cores than its ini.
bool LoadRuntimePlatformInfos(fe::PlatFormInfos *infos) {
  int32_t device_id = -1;
  if (aclrtGetDevice(&device_id) != ACL_SUCCESS || device_id < 0) {
    return false;
  }
  (void)fe::PlatformInfoManager::GeInstance().InitializePlatformInfo();
  return fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(
             static_cast<uint32_t>(device_id), *infos, true) == 0U;
}

// Full SoC name (Ascend910B3), not Short_SoC_version. B1-B4 share Ascend910B, and that
// ini has vector_core_cnt=0, so WithOutSocVersion cannot supply Mix core nums.
bool LoadNamedPlatformInfos(fe::PlatFormInfos *infos) {
  const char *soc_name = aclrtGetSocName();
  if (soc_name == nullptr || soc_name[0] == '\0') {
    return false;
  }
  fe::OptionalInfos opti;
  (void)fe::PlatformInfoManager::GeInstance().InitializePlatformInfo();
  if (fe::PlatformInfoManager::GeInstance().GetPlatformInfos(soc_name, *infos, opti) == 0U) {
    return true;
  }
  (void)fe::PlatformInfoManager::Instance().InitializePlatformInfo();
  return fe::PlatformInfoManager::Instance().GetPlatformInfos(soc_name, *infos, opti) == 0U;
}

bool LoadShortSocPlatformInfos(fe::PlatFormInfos *infos) {
  fe::OptionalInfos opti;
  (void)fe::PlatformInfoManager::GeInstance().InitializePlatformInfo();
  uint32_t ret = fe::PlatformInfoManager::GeInstance().GetPlatformInfoWithOutSocVersion(*infos, opti);
  if (ret != 0U) {
    (void)fe::PlatformInfoManager::Instance().InitializePlatformInfo();
    ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(*infos, opti);
  }
  return ret == 0U;
}

// GE task_builder GetPlatformCoreNum: [all, ai, vec] for MixCoreArgs.
bool QueryMixPlatformCoreNums(uint32_t *all_core, uint32_t *ai_core, uint32_t *vec_core) {
  if (all_core == nullptr || ai_core == nullptr || vec_core == nullptr) {
    return false;
  }
  fe::PlatFormInfos infos;
  if (LoadRuntimePlatformInfos(&infos) && ReadMixCoreNums(&infos, all_core, ai_core, vec_core)) {
    return true;
  }
  if (LoadNamedPlatformInfos(&infos) && ReadMixCoreNums(&infos, all_core, ai_core, vec_core)) {
    return true;
  }
  return LoadShortSocPlatformInfos(&infos) && ReadMixCoreNums(&infos, all_core, ai_core, vec_core);
}

// GE enables FFTS synchronization only when SoCInfo declares ffts-plus.
bool PlatformDeclaresFftsPlus(fe::PlatFormInfos &infos) {
  std::string ffts_mode;
  return infos.GetPlatformRes("SoCInfo", "ffts_mode", ffts_mode) && ffts_mode == "ffts-plus";
}

// Per-op const nodes shared by the whole RT chain (tiling + launch).
struct AiCoreConsts {
  RtGraphBuilder::Value node_desc;
  RtGraphBuilder::Value args_info;
  RtGraphBuilder::Value compile;
  RtGraphBuilder::Value compile_key;
  RtGraphBuilder::Value op_type;
  RtGraphBuilder::Value bindata;
  RtGraphBuilder::Value bin_key;
  RtGraphBuilder::Value magic;
  RtGraphBuilder::Value kernel_name;
  RtGraphBuilder::Value with_handle;
  RtGraphBuilder::Value cfg_attrs;
  RtGraphBuilder::Value node_info;
  RtGraphBuilder::Value cfg;
  RtGraphBuilder::Value dfx;
  RtGraphBuilder::Value null_sb;
  RtGraphBuilder::Value io_num;
  // Constant 0 for the launch function-lookup key; see the single-function note in
  // WireAiCoreLaunch.
  RtGraphBuilder::Value tiling_key_zero;
};

AiCoreConsts BuildAiCoreConsts(RtGraphBuilder &gb, const std::string &tag, const AiCoreBinArtifacts &art,
                               const char *opp_type, size_t io_num) {
  AiCoreConsts cs;
  cs.node_desc =
      gb.Const("const_node_desc_" + tag, art.compute_node_desc->data(), art.compute_node_desc->size(), false,
               FxrtConstType::kAiCoreComputeNodeDescPtr);
  cs.args_info = gb.Const("const_args_info_" + tag, art.args_info_desc->data(), art.args_info_desc->size(), false,
                          FxrtConstType::kAiCoreArgsInfosDescPtr);
  cs.compile = gb.Const("const_compile_info_" + tag, art.compile_info_cstr->data(), art.compile_info_cstr->size(),
                        true);
  cs.compile_key = gb.Const("const_compile_key_" + tag, art.compile_info_key_cstr->data(),
                            art.compile_info_key_cstr->size(), true);
  cs.op_type =
      gb.Const("const_op_type_" + tag, opp_type, std::char_traits<char>::length(opp_type) + 1, true);
  cs.bindata = gb.Const("const_bindata_" + tag, art.bindata_blob->data(), art.bindata_blob->size(), false,
                        FxrtConstType::kBytePtr);
  cs.bin_key = gb.Const("const_bin_key_" + tag, art.bin_key_cstr->data(), art.bin_key_cstr->size(), true);
  cs.magic = gb.Const("const_magic_" + tag, &art.binary_magic, sizeof(art.binary_magic), false,
                      FxrtConstType::kUInt32T);
  cs.kernel_name = gb.Const("const_kernel_name_" + tag, art.kernel_name.c_str(), art.kernel_name.size() + 1, true);
  const uint32_t with_handle = 1U;
  cs.with_handle = gb.Const("const_with_handle_" + tag, &with_handle, sizeof(with_handle), false,
                            FxrtConstType::kUInt32T);
  cs.node_info = gb.Const("const_node_info_" + tag, art.node_info_cstr->data(), art.node_info_cstr->size(), true);

  AclRtLaunchKernelAttrCompat attrs[5]{};
  attrs[0].id = kAclRtLaunchKernelAttrSchemMode;
  attrs[1].id = kAclRtLaunchKernelAttrDynUbufSize;
  attrs[2].id = kAclRtLaunchKernelAttrEngineType;
  attrs[2].value.engineType = art.core_type == "VectorCore" ? ACL_RT_ENGINE_TYPE_AIV : ACL_RT_ENGINE_TYPE_AIC;
  attrs[3].id = kAclRtLaunchKernelAttrBlockDimOffset;
  attrs[4].id = kAclRtLaunchKernelAttrEnableProfiling;
  attrs[4].value.enableProfiling = 1U;
  cs.cfg_attrs = gb.Const("const_cfg_attrs_" + tag, attrs, sizeof(attrs), false,
                          FxrtConstType::kAclrtLaunchKernelAttrPtr);
  aclrtLaunchKernelCfg cfg{};
  cfg.numAttrs = 5U;
  cs.cfg = gb.Const("const_cfg_" + tag, &cfg, sizeof(cfg), false, FxrtConstType::kAclrtLaunchKernelCfgPtr);
  aicore_rt::DfxExeArg dfx{};
  cs.dfx = gb.Const("const_dfx_" + tag, &dfx, sizeof(dfx), false, FxrtConstType::kAiCoreDfxExeArgPtr);
  void *null_shapebuf = nullptr;
  cs.null_sb = gb.Const("const_null_shapebuf_" + tag, &null_shapebuf, sizeof(null_shapebuf), false,
                        FxrtConstType::kVoidPtr);
  cs.io_num = gb.Const("const_io_num_" + tag, &io_num, sizeof(io_num), false, FxrtConstType::kSizeT);
  // A constant 0 for the launch function-lookup key, selected only when the chosen bin
  // ships a single function (function_count == 1). An OpImpl tiling may still emit a
  // nonzero tilingKey (Cumsum picks 204 once the axis value is host-readable); that value
  // is a tiling-data id, not a function variant, so a single-function bin must look the
  // function up with key 0.
  const uint64_t tiling_key_zero = 0;
  cs.tiling_key_zero = gb.Const("const_tiling_key_zero_" + tag, &tiling_key_zero, sizeof(tiling_key_zero), false,
                                FxrtConstType::kUInt64T);
  return cs;
}

std::vector<ge::TensorDesc> MakeOutDescs(const AiCoreOpWork &work) {
  std::vector<ge::TensorDesc> td_outs;
  td_outs.reserve(work.dtypes.size());
  for (size_t o = 0; o < work.dtypes.size(); ++o) {
    td_outs.push_back(MakeNdTensorDesc(work.out_shapes[o], work.dtypes[o]));
  }
  return td_outs;
}

}  // namespace

TensorRef AiCoreStage::OutputRef(size_t o, const std::vector<int64_t> &shape, ge::DataType dtype) const {
  TensorRef ref;
  if (o >= outs.size()) {
    return ref;
  }
  ref.producer = outs[o].tensor.node;
  ref.symbol = outs[o].tensor.sym;
  ref.shape = shape;
  ref.dtype = dtype;
  ref.unpacked.shape = TensorEndpoint{infer, static_cast<int32_t>(o), OutShapeSymAt(o)};
  ref.unpacked.addr = outs[o].addr;
  if (outs[o].owns_addr) ref.allocation = outs[o].addr;
  return ref;
}

std::vector<fx_converter::Node *> AiCoreStage::ComputeNodes() const {
  std::vector<fx_converter::Node *> nodes{infer, tiling, launch};
  for (const OutputIo &out_io : outs) {
    // A ref output borrows an input address that already belongs to another node.
    if (out_io.owns_addr && out_io.addr.node != nullptr) {
      nodes.push_back(out_io.addr.node);
    }
  }
  if (tiling_parse != nullptr) {
    nodes.push_back(tiling_parse);
  }
  return nodes;
}

std::vector<fx_converter::Node *> AiCoreStage::IntermediateAllocs(const std::vector<bool> &final_mask) const {
  std::vector<fx_converter::Node *> allocs;
  for (size_t o = 0; o < outs.size(); ++o) {
    if (o < final_mask.size() && final_mask[o]) {
      continue;  // Final outputs live in model-owned memory; nothing to free.
    }
    if (outs[o].owns_addr && outs[o].addr.node != nullptr) {
      allocs.push_back(outs[o].addr.node);
    }
  }
  return allocs;
}

void WireInferShape(RtGraphBuilder &gb, AiCoreOpWork &work, const std::vector<RtGraphBuilder::Value> &c_rules,
                    RtGraphBuilder::Value c_infer, fx_converter::Node *bind_symbols) {
  const std::string &tag = work.tag;
  const std::vector<TensorRef> &ins = work.ins;
  AiCoreStage &stage = work.stage;
  const std::vector<ge::TensorDesc> td_outs = MakeOutDescs(work);
  std::vector<RtGraphBuilder::Value> in_vals;
  std::vector<ge::TensorDesc> td_ins;
  in_vals.reserve(ins.size() + c_rules.size() + 1);
  td_ins.reserve(ins.size() + c_rules.size() + 1);
  for (const TensorRef &in : ins) {
    // Keep one input slot per logical tensor for GE scheduling, but feed the unpacked
    // StorageShape value when available instead of boxing a Tensor.
    td_ins.push_back(MakeNdTensorDesc(PhysicalShape(in), in.dtype));
    in_vals.push_back(in.HasUnpackedPair() ? RtGraphBuilder::From(in.unpacked.shape)
                                           : RtGraphBuilder::From(in.producer, 0, in.symbol));
  }
  // One shape-rule const per output slot (FxExprInferFromRule reads rule i for slot i).
  for (const RtGraphBuilder::Value &c_rule : c_rules) {
    td_ins.push_back(gb.empty_td());  // rule
    in_vals.push_back(c_rule);
  }
  td_ins.push_back(gb.empty_td());  // infer_func
  in_vals.push_back(c_infer);
  auto infer = gb.TypedNode("InferShape", "infer_" + tag, in_vals, td_ins, td_outs);
  stage.infer = infer.node;
  stage.s_shapes.clear();
  stage.s_shapes.reserve(infer.outs.size());
  for (const RtGraphBuilder::Value &out : infer.outs) {
    stage.s_shapes.push_back(out.sym);
  }
  if (bind_symbols != nullptr && stage.infer != nullptr) {
    (void)AddControlEdge(gb.main(), bind_symbols, stage.infer);
  }
}

void WireIoAddrs(RtGraphBuilder &gb, AiCoreOpWork &work, LoweringContext &ctx) {
  WireAiCoreInputSplits(gb.main(), work.tag, work.ins, gb.sym_gen(), gb.kernel_nodes(), work.stage, ctx);
  WireAiCoreOutputs(gb, work.tag, work.final_mask, MakeRtValues(*ctx.rt), MakeOutDescs(work), work.dtypes,
                    work.entry, work.stage);
}

size_t BuildTilingDataDependencyMask(const std::vector<TensorRef> &ins, int32_t shape_input_index) {
  constexpr size_t kByteBitCount = 8;
  if (ins.size() >= sizeof(size_t) * kByteBitCount) {
    return 0;
  }
  size_t mask = 0;
  for (size_t i = 0; i < ins.size(); ++i) {
    if (static_cast<int32_t>(i) == shape_input_index || ins[i].tiling_ep.Valid()) {
      mask |= (static_cast<size_t>(1) << i);
    }
  }
  return mask;
}

void PadAiCoreOptionalInputs(RtGraphBuilder &gb, AiCoreOpWork &work) {
  const std::string &tag = work.tag;
  AiCoreStage &stage = work.stage;
  for (size_t i = 0; i < work.optional.pad; ++i) {
    const std::string suffix = tag + "_opt" + std::to_string(i);
    AiCoreStage::InputIo io;
    // GE gen_placeholder (InitInputsForMixL2): GertTensorData with a null device addr.
    // The name must start with const_gtd_: OM2 ConstValueCollector restores an inline
    // GertTensorData only for that prefix. const_opt_gtd_ missed it and was rejected
    // as an 88-byte pointer.
    gert::GertTensorData empty_gtd;
    (void)empty_gtd.MutableTensorData().SetAddr(nullptr, nullptr);
    empty_gtd.SetPlacement(gert::kOnDeviceHbm);
    io.addr = gb.Const("const_gtd_opt_" + suffix, &empty_gtd, sizeof(empty_gtd), false,
                       FxrtConstType::kGertTensorDataPtr);
    // Launch reads this slot as StorageShape* when a merged-copy arg needs the shape.
    // An int32(1) is 4 bytes, which OM2 rejects on a kVoidPtr port. A rank-1 [1]
    // StorageShape under the const_storage_shape_ prefix is the object that collector
    // restores, and it stays a legal shape if the kernel does read it.
    gert::StorageShape placeholder_shape;
    placeholder_shape.MutableOriginShape().SetDimNum(1);
    placeholder_shape.MutableStorageShape().SetDimNum(1);
    placeholder_shape.MutableOriginShape().SetDim(0, 1);
    placeholder_shape.MutableStorageShape().SetDim(0, 1);
    io.shape = gb.Const("const_storage_shape_opt_" + suffix, &placeholder_shape, sizeof(placeholder_shape), false,
                        FxrtConstType::kGertStorageShapePtr);
    io.tensor = io.shape;
    stage.ins.push_back(io);
  }
}

// Own tiling + launch kernels. They call the CANN OpImpl tiling function and
// aclrtLaunchKernelWithHostArgs. Mix stays on the GE launch chain.
float CatalogDefaultFloat(const GeAttr &attr) {
  if (attr.text != nullptr && attr.text[0] != '\0') {
    return std::strtof(attr.text, nullptr);
  }
  return static_cast<float>(attr.literal);
}

// IR-order RuntimeAttrs. Offsets are from the start of the blob, matching RuntimeAttrs::GetAttrPointer.
std::vector<uint8_t> PackTilingAttrs(const AiCoreOpWork &work) {
  const GeAttr *attrs = work.entry.attrs;
  const size_t count = attrs == nullptr ? 0U : work.entry.attr_count;
  const size_t header = (sizeof(RuntimeAttrsDef) + count * sizeof(size_t) + 7U) & ~size_t{7};
  std::vector<uint8_t> values;
  std::vector<size_t> offsets(count, 0);
  auto align8 = [](std::vector<uint8_t> *buf) {
    while (buf->size() % 8U != 0U) {
      buf->push_back(0);
    }
  };
  for (size_t i = 0; i < count; ++i) {
    const GeAttr &attr = attrs[i];
    align8(&values);
    offsets[i] = header + values.size();
    const int64_t ivalue = work.extras.IntOr(attr.name, attr.literal);
    switch (attr.type) {
      case GeAttrType::kBool: {
        const uint8_t bit = work.extras.BoolOr(attr.name, attr.literal != 0) ? 1U : 0U;
        values.push_back(bit);
        break;
      }
      case GeAttrType::kInt: {
        const size_t at = values.size();
        values.resize(at + sizeof(int64_t));
        std::memcpy(values.data() + at, &ivalue, sizeof(ivalue));
        break;
      }
      case GeAttrType::kFloat: {
        const float fvalue = work.extras.FloatOr(attr.name, CatalogDefaultFloat(attr));
        const size_t at = values.size();
        values.resize(at + sizeof(float));
        std::memcpy(values.data() + at, &fvalue, sizeof(fvalue));
        break;
      }
      case GeAttrType::kStr: {
        const char *text = work.extras.StrOr(attr.name, attr.text);
        if (text == nullptr) {
          text = "";
        }
        values.insert(values.end(), text, text + std::char_traits<char>::length(text) + 1);
        break;
      }
      case GeAttrType::kListInt: {
        auto holder = gert::ContinuousVector::Create<int64_t>(1);
        auto *vec = reinterpret_cast<gert::TypedContinuousVector<int64_t> *>(holder.get());
        (void)vec->SetSize(1);
        vec->MutableData()[0] = ivalue;
        const size_t bytes = sizeof(gert::ContinuousVector);
        const size_t at = values.size();
        values.resize(at + bytes);
        std::memcpy(values.data() + at, holder.get(), bytes);
        break;
      }
    }
  }
  std::vector<uint8_t> blob(header + values.size(), 0);
  auto *def = reinterpret_cast<RuntimeAttrsDef *>(blob.data());
  def->attr_num = count;
  for (size_t i = 0; i < count; ++i) {
    def->offset[i] = offsets[i];
  }
  if (!values.empty()) {
    std::memcpy(blob.data() + header, values.data(), values.size());
  }
  return blob;
}

bool WireFxAiCore(RtGraphBuilder &gb, const std::string &tag, size_t nin, size_t nout, size_t launch_nin,
                  const AiCoreBinArtifacts &art, AiCoreStage &stage, const RtValues &rtv, const AiCoreConsts &cs,
                  aicore_rt::CoreTypeIndex core_type, const AiCoreOpWork &work,
                  const aicore_rt::MixCoreArgs *mix, uint32_t need_ffts) {
  const auto *node_desc = reinterpret_cast<const aicore_rt::ComputeNodeDesc *>(art.compute_node_desc->data());
  const uint32_t max_tiling = static_cast<uint32_t>(node_desc->max_tiling_data);
  const uint32_t workspace_cap = static_cast<uint32_t>(node_desc->workspace_cap);
  const uint32_t function_count = art.function_count;
  const uint32_t nin32 = static_cast<uint32_t>(nin);
  const uint32_t nout32 = static_cast<uint32_t>(nout);
  const uint32_t launch_nin32 = static_cast<uint32_t>(launch_nin);
  const auto c_max = gb.Const("const_fx_max_tiling_" + tag, &max_tiling, sizeof(max_tiling), false,
                              FxrtConstType::kUInt32T);
  const auto c_cap = gb.Const("const_fx_ws_cap_" + tag, &workspace_cap, sizeof(workspace_cap), false,
                              FxrtConstType::kUInt32T);
  const auto c_fn = gb.Const("const_fx_fn_count_" + tag, &function_count, sizeof(function_count), false,
                             FxrtConstType::kUInt32T);
  const auto c_nin = gb.Const("const_fx_nin_" + tag, &nin32, sizeof(nin32), false, FxrtConstType::kUInt32T);
  const auto c_nout = gb.Const("const_fx_nout_" + tag, &nout32, sizeof(nout32), false, FxrtConstType::kUInt32T);
  const auto c_launch_nin = gb.Const("const_fx_launch_nin_" + tag, &launch_nin32, sizeof(launch_nin32), false,
                                     FxrtConstType::kUInt32T);
  const std::vector<uint8_t> attr_blob = PackTilingAttrs(work);
  const uint64_t attr_bytes = attr_blob.size();
  const auto c_attrs = gb.Const("const_fx_attrs_" + tag, attr_blob.data(), attr_blob.size(), false,
                                FxrtConstType::kBytePtr);
  const auto c_attr_bytes = gb.Const("const_fx_attr_bytes_" + tag, &attr_bytes, sizeof(attr_bytes), false,
                                     FxrtConstType::kSizeT);
  ge::DataType dtype_pad = ge::DT_FLOAT;
  const ge::DataType *dtype_data = work.dtypes.empty() ? &dtype_pad : work.dtypes.data();
  const size_t dtype_bytes = (work.dtypes.empty() ? 1U : work.dtypes.size()) * sizeof(ge::DataType);
  const auto c_out_dtypes = gb.Const("const_fx_out_dtypes_" + tag, dtype_data, dtype_bytes, false);

  const int32_t op_ai_core_num = -1;
  const int32_t op_vec_core_num = -1;
  const RtGraphBuilder::Value c_core_type = gb.Const("const_core_type_" + tag, &core_type, sizeof(core_type), false,
                                                     FxrtConstType::kGertCoreTypeIndexPtr);
  const RtGraphBuilder::Value c_op_ai_num =
      gb.Const("const_op_aicore_num_" + tag, &op_ai_core_num, sizeof(op_ai_core_num), false);
  const RtGraphBuilder::Value c_op_vec_num =
      gb.Const("const_op_vec_core_num_" + tag, &op_vec_core_num, sizeof(op_vec_core_num), false);
  std::vector<RtGraphBuilder::Value> append_ins(Idx(lyt::AppendCoreTypeToPlatformInputIndex::kNum));
  append_ins[Idx(lyt::AppendCoreTypeToPlatformInputIndex::kPlatform)] = rtv.platform;
  append_ins[Idx(lyt::AppendCoreTypeToPlatformInputIndex::kCoreType)] = c_core_type;
  append_ins[Idx(lyt::AppendCoreTypeToPlatformInputIndex::kOpAiCoreNum)] = c_op_ai_num;
  append_ins[Idx(lyt::AppendCoreTypeToPlatformInputIndex::kOpVectorCoreNum)] = c_op_vec_num;
  append_ins[Idx(lyt::AppendCoreTypeToPlatformInputIndex::kCoreNumInfos)] = rtv.core_num;
  const auto append = gb.Node("AppendCoreTypeToPlatform", "append_core_type_" + tag, append_ins, 2);
  const RtGraphBuilder::Value platform = append.outs.front();

  std::vector<RtGraphBuilder::Value> tiling_ins;
  tiling_ins.reserve(9 + nin + nout);
  tiling_ins.push_back(cs.op_type);
  tiling_ins.push_back(cs.compile);
  tiling_ins.push_back(platform);
  tiling_ins.push_back(c_max);
  tiling_ins.push_back(c_nin);
  tiling_ins.push_back(c_nout);
  tiling_ins.push_back(c_attrs);
  tiling_ins.push_back(c_attr_bytes);
  tiling_ins.push_back(c_out_dtypes);
  for (size_t i = 0; i < nin; ++i) {
    tiling_ins.push_back(stage.ins[i].tensor);
  }
  for (size_t o = 0; o < nout; ++o) {
    tiling_ins.push_back(stage.OutShapeSlotAt(o));
  }
  const auto tiling = gb.Node("FxAiCoreTiling", "fx_tiling_" + tag, tiling_ins, 5);
  stage.tiling = tiling.node;

  const uint32_t mix_enable = mix == nullptr ? 0U : 1U;
  aicore_rt::MixCoreArgs mix_pad{};
  const aicore_rt::MixCoreArgs &mix_src = mix == nullptr ? mix_pad : *mix;
  const auto c_mix_enable = gb.Const("const_fx_mix_enable_" + tag, &mix_enable, sizeof(mix_enable), false,
                                     FxrtConstType::kUInt32T);
  const auto c_mix = gb.Const("const_fx_mix_args_" + tag, &mix_src, sizeof(mix_src), false,
                              FxrtConstType::kAiCoreMixCoreArgsPtr);
  const auto c_ffts = gb.Const("const_fx_need_ffts_" + tag, &need_ffts, sizeof(need_ffts), false,
                               FxrtConstType::kUInt32T);
  std::vector<RtGraphBuilder::Value> launch_ins;
  launch_ins.reserve(16 + launch_nin + nout);
  launch_ins.push_back(cs.bindata);
  launch_ins.push_back(cs.magic);
  launch_ins.push_back(cs.kernel_name);
  launch_ins.push_back(tiling.outs[1]);
  launch_ins.push_back(tiling.outs[2]);
  launch_ins.push_back(c_fn);
  launch_ins.push_back(rtv.stream);
  launch_ins.push_back(tiling.outs[0]);
  launch_ins.push_back(tiling.outs[3]);
  launch_ins.push_back(tiling.outs[4]);
  launch_ins.push_back(c_launch_nin);
  launch_ins.push_back(c_nout);
  launch_ins.push_back(c_cap);
  launch_ins.push_back(c_mix_enable);
  launch_ins.push_back(c_mix);
  launch_ins.push_back(c_ffts);
  for (size_t i = 0; i < launch_nin; ++i) {
    launch_ins.push_back(stage.ins[i].addr);
  }
  for (size_t o = 0; o < nout; ++o) {
    launch_ins.push_back(stage.outs[o].addr);
  }
  stage.launch = gb.Node("FxAiCoreLaunch", "fx_launch_" + tag, launch_ins, 0).node;
  for (size_t o = 0; o < nout; ++o) {
    (void)AddControlEdge(gb.main(), stage.outs[o].addr.node, stage.launch);
  }
  return true;
}

bool WireAiCoreExecute(RtGraphBuilder &gb, AiCoreOpWork &work, const SharedRuntime &rt, std::string *err) {
  const std::string &tag = work.tag;
  const char *opp_type = work.bind.opp_type;
  const size_t nin = work.ins.size();
  const size_t nout = work.dtypes.size();
  const AiCoreBinArtifacts &art = work.art;
  AiCoreStage &stage = work.stage;
  // nin/nout are the tiling op description blob io counts. The launch io array
  // alone grows by work.optional.pad (GE gen_placeholder semantics).
  const size_t launch_nin = nin + work.optional.pad;
  const RtValues rtv = MakeRtValues(rt);
  const aicore_rt::CoreTypeIndex core_type = aicore_rt::CoreTypeIndexFromJson(art.core_type);
  const AiCoreConsts cs = BuildAiCoreConsts(gb, tag, art, opp_type, launch_nin + nout);
  const bool ffts = QueryPlatformRunsFftsPlus();
  aicore_rt::MixCoreArgs mix_payload{};
  const aicore_rt::MixCoreArgs *mix = nullptr;
  uint32_t need_ffts = 0U;
  if (aicore_rt::IsMixLaunchCoreType(core_type) && ffts) {
    uint32_t all_core = 0;
    uint32_t ai_core = 0;
    uint32_t vec_core = 0;
    if (!QueryMixPlatformCoreNums(&all_core, &ai_core, &vec_core)) {
      if (err != nullptr) {
        *err = "aicore-binary lowering: platform Mix core nums unavailable for " + tag;
      }
      return false;
    }
    mix_payload.mix_type = MixTypeFromCoreIndex(core_type);
    mix_payload.all_core_num = all_core;
    mix_payload.ai_core_num = ai_core;
    mix_payload.vec_core_num = vec_core;
    mix = &mix_payload;
    need_ffts = art.intercore_sync != 0 ? 1U : 0U;
    RT_GLOG(INFO) << "WireAiCoreExecute: Mix launch tag=" << tag << " type="
                  << static_cast<int>(mix_payload.mix_type) << " cores(all/ai/vec)=" << mix_payload.all_core_num
                  << "/" << mix_payload.ai_core_num << "/" << mix_payload.vec_core_num
                  << " taskRation=" << art.task_ration << " intercoreSync=" << art.intercore_sync;
  }
  return WireFxAiCore(gb, tag, nin, nout, launch_nin, art, stage, rtv, cs, core_type, work, mix, need_ffts);
}

void WireAiCoreFinalOutputs(RtGraphBuilder &gb, AiCoreOpWork &work, const SharedRuntime &rt) {
  const std::string &tag = work.tag;
  const std::vector<ge::DataType> &out_dtypes = work.dtypes;
  const std::vector<bool> &final_mask = work.final_mask;
  AiCoreStage &stage = work.stage;
  const size_t nout = out_dtypes.size();
  for (size_t o = 0; o < nout; ++o) {
    if (o >= final_mask.size() || !final_mask[o] || stage.outs[o].tensor.node == nullptr) {
      continue;  // Intermediate or ref slot: no OutputData shell was built for it.
    }
    const std::string osuffix = tag + ((nout > 1) ? ("_o" + std::to_string(o)) : "");
    const int32_t dtype_value = static_cast<int32_t>(out_dtypes[o]);
    const int32_t placement_value = static_cast<int32_t>(gert::kOnDeviceHbm);
    const int32_t storage_value = static_cast<int32_t>(ge::FORMAT_ND);
    const int32_t origin_value = static_cast<int32_t>(OriginFormatOf(ge::FORMAT_ND));
    const RtGraphBuilder::Value dtype =
        gb.Const("const_ensure_dtype_" + osuffix, &dtype_value, sizeof(dtype_value), false, FxrtConstType::kInt32T);
    const RtGraphBuilder::Value placement =
        gb.Const("const_ensure_placement_" + osuffix, &placement_value, sizeof(placement_value), false,
                 FxrtConstType::kInt32T);
    const RtGraphBuilder::Value storage =
        gb.Const("const_ensure_storage_format_" + osuffix, &storage_value, sizeof(storage_value), false,
                 FxrtConstType::kInt32T);
    const RtGraphBuilder::Value origin =
        gb.Const("const_ensure_origin_format_" + osuffix, &origin_value, sizeof(origin_value), false,
                 FxrtConstType::kInt32T);
    const auto ensure = gb.SourcedNode(
        "FxEnsureTensorAtOutMemory", "ensure_out_" + osuffix,
        {stage.OutShapeSlotAt(o), stage.outs[o].addr, dtype, placement, storage, origin,
         RtGraphBuilder::From(rt.split_streams, 0, rt.s_stream), stage.outs[o].tensor,
         RtGraphBuilder::From(rt.select_l2, 0, rt.s_allocator)},
        0);
    (void)ensure;
    (void)AddControlEdge(gb.main(), stage.launch, ensure.node);
  }
}

// Query the same platform sources used by Mix resource setup.
bool QueryPlatformRunsFftsPlus() {
  fe::PlatFormInfos infos;
  if (LoadRuntimePlatformInfos(&infos) && PlatformDeclaresFftsPlus(infos)) {
    return true;
  }
  if (LoadNamedPlatformInfos(&infos) && PlatformDeclaresFftsPlus(infos)) {
    return true;
  }
  return LoadShortSocPlatformInfos(&infos) && PlatformDeclaresFftsPlus(infos);
}

}  // namespace fx_converter
