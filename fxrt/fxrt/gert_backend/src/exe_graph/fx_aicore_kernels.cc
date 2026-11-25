#include <dlfcn.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl_rt.h"
#include "common/logger.h"
#include "exe_graph/fxrt_kernel_export.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/kernel_run_context.h"
#include "exe_graph/runtime/runtime_tensor.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_data.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "lowering/aicore/aicore_rt_layout.h"
#include "opp/opp_registry.h"

namespace gert {
namespace {

constexpr size_t kTilingFixedInputs = 9U;
constexpr size_t kLaunchFixedInputs = 16U;
constexpr size_t kTilingOutputCount = 5U;
constexpr size_t kTilingDataCap = 256U * 1024U;
constexpr size_t kInnerOutputCount = static_cast<size_t>(TilingContext::kOutputNum);
constexpr size_t kInnerFixedInputs = 6U;
constexpr size_t kMaxHostEmbed = 128U;
constexpr size_t kHostKeyCap = 4096U;

enum TilingOut : size_t {
  kOutWorkspace = 0,
  kOutBlockDim,
  kOutTilingKey,
  kOutTilingData,
  kOutTilingSize
};

enum LaunchInput : size_t {
  kInBin = 0,
  kInMagic,
  kInKernelName,
  kInBlockDim,
  kInTilingKey,
  kInFunctionCount,
  kInStream,
  kInWorkspace,
  kInTilingData,
  kInTilingSize,
  kInInputCount,
  kInOutputCount,
  kInWorkspaceCap,
  kInMixEnable,
  kInMixArgs,
  kInNeedFfts,
  kInIo
};

struct CompileKey {
  const void *op = nullptr;
  const void *json = nullptr;
  const void *platform = nullptr;
  bool operator<(const CompileKey &other) const {
    if (op != other.op) {
      return op < other.op;
    }
    if (json != other.json) {
      return json < other.json;
    }
    return platform < other.platform;
  }
};

std::mutex g_compile_mu;
std::map<CompileKey, void *> g_compile_info;
std::mutex g_bin_mu;
std::map<const void *, aclrtBinHandle> g_bin_handles;
std::mutex g_log_mu;
std::map<std::string, int> g_logged_ops;

void NoteOp(const char *op_type) {
  std::lock_guard<std::mutex> lock(g_log_mu);
  const std::string key = op_type == nullptr ? "" : op_type;
  if (g_logged_ops[key]++ == 0) {
    RT_GLOG(INFO) << "FxAiCoreTiling op=" << key;
  }
}

struct TilingState {
  std::vector<int64_t> key;
  uint32_t block_dim = 0;
  uint64_t tiling_key = 0;
  uint64_t tiling_size = 0;
  std::vector<uint8_t> tiling;
  std::vector<size_t> workspace;
  bool valid = false;
};

struct HostKeep {
  const void *host = nullptr;
  size_t size = 0;
  void *dev = nullptr;
  size_t cap = 0;
};

struct LaunchState {
  const void *bin = nullptr;
  uint64_t entry = ~uint64_t{0};
  aclrtFuncHandle func = nullptr;
  std::vector<uint8_t> args;
  std::vector<void *> ws_dev;
  std::vector<size_t> ws_cap;
  std::vector<HostKeep> host;
  aclrtStream sub_stream = nullptr;
  aclrtNotify notify0 = nullptr;
  aclrtNotify notify1 = nullptr;
};

std::mutex g_state_mu;
std::map<const KernelContext *, TilingState> g_tiling_state;
std::map<const KernelContext *, LaunchState> g_launch_state;

TilingState &TilingStateOf(const KernelContext *context) {
  std::lock_guard<std::mutex> lock(g_state_mu);
  return g_tiling_state[context];
}

LaunchState &LaunchStateOf(const KernelContext *context) {
  std::lock_guard<std::mutex> lock(g_state_mu);
  return g_launch_state[context];
}

void PushShape(const Shape &shape, std::vector<int64_t> *key) {
  key->push_back(static_cast<int64_t>(shape.GetDimNum()));
  for (size_t dim = 0; dim < shape.GetDimNum(); ++dim) {
    key->push_back(shape.GetDim(dim));
  }
}

// Host bytes stay in the key so a shape-list tensor cannot reuse another value.
// A host tensor larger than the cap disables the cache for this node.
bool BuildTilingKey(const Tensor *const *tensors, size_t nin, const StorageShape *const *shapes, size_t nout,
                    std::vector<int64_t> *key) {
  key->clear();
  key->push_back(static_cast<int64_t>(nin));
  key->push_back(static_cast<int64_t>(nout));
  for (size_t i = 0; i < nin; ++i) {
    const Tensor *tensor = tensors[i];
    if (tensor == nullptr) {
      key->push_back(-1);
      continue;
    }
    PushShape(tensor->GetStorageShape(), key);
    key->push_back(static_cast<int64_t>(tensor->GetDataType()));
    const bool host = TensorPlacementUtils::IsOnHost(tensor->GetPlacement());
    const size_t bytes = tensor->GetSize();
    const void *addr = tensor->GetAddr();
    if (host && bytes > 0 && addr != nullptr) {
      if (bytes > kHostKeyCap) {
        return false;
      }
      key->push_back(static_cast<int64_t>(bytes));
      const auto *raw = static_cast<const uint8_t *>(addr);
      for (size_t offset = 0; offset < bytes; offset += sizeof(uint64_t)) {
        uint64_t word = 0;
        std::memcpy(&word, raw + offset, std::min(sizeof(word), bytes - offset));
        key->push_back(static_cast<int64_t>(word));
      }
    } else {
      key->push_back(0);
    }
  }
  for (size_t o = 0; o < nout; ++o) {
    if (shapes[o] == nullptr) {
      key->push_back(-1);
      continue;
    }
    PushShape(shapes[o]->GetStorageShape(), key);
  }
  return true;
}

bool ApplyTilingState(const TilingState &state, TypedContinuousVector<size_t> *workspace, uint32_t *block_dim,
                      uint64_t *tiling_key, uint8_t *tiling_out, uint64_t *tiling_size) {
  if (workspace->SetSize(state.workspace.size()) != ge::SUCCESS) {
    return false;
  }
  if (!state.workspace.empty()) {
    std::memcpy(workspace->MutableData(), state.workspace.data(), state.workspace.size() * sizeof(size_t));
  }
  *block_dim = state.block_dim;
  *tiling_key = state.tiling_key;
  *tiling_size = state.tiling_size;
  if (state.tiling_size > 0) {
    std::memcpy(tiling_out, state.tiling.data(), static_cast<size_t>(state.tiling_size));
  }
  return true;
}

void SaveTilingState(TilingState *state, std::vector<int64_t> key, const TypedContinuousVector<size_t> *workspace,
                     uint32_t block_dim, uint64_t tiling_key, const uint8_t *tiling_out, uint64_t tiling_size) {
  state->key = std::move(key);
  state->block_dim = block_dim;
  state->tiling_key = tiling_key;
  state->tiling_size = tiling_size;
  state->tiling.assign(tiling_out, tiling_out + static_cast<size_t>(tiling_size));
  const size_t nws = workspace->GetSize();
  state->workspace.assign(workspace->GetData(), workspace->GetData() + nws);
  state->valid = true;
}

bool EnsureDeviceBuf(void **ptr, size_t *cap, size_t bytes) {
  if (bytes == 0) {
    return true;
  }
  if (*ptr != nullptr && *cap >= bytes) {
    return true;
  }
  if (*ptr != nullptr) {
    (void)aclrtFree(*ptr);
    *ptr = nullptr;
    *cap = 0;
  }
  void *dev = nullptr;
  if (aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || dev == nullptr) {
    return false;
  }
  *ptr = dev;
  *cap = bytes;
  return true;
}

template <typename T>
void SetInline(Chain &chain, const T &value) {
  *chain.GetPointer<T>() = value;
}

struct OwnedBuf {
  void *p = nullptr;
  ~OwnedBuf() { std::free(p); }
  bool Alloc(size_t bytes) {
    if (posix_memalign(&p, 64, bytes == 0 ? 64 : bytes) != 0) {
      p = nullptr;
      return false;
    }
    std::memset(p, 0, bytes == 0 ? 64 : bytes);
    return true;
  }
};

KernelRunContext *MakeRunContext(OwnedBuf &buf, std::vector<Chain> &chains, size_t inputs, size_t outputs,
                                 const void *compute_node) {
  const size_t n = inputs + outputs;
  const size_t bytes = offsetof(KernelRunContext, values) + n * sizeof(AsyncAnyValue *);
  if (!buf.Alloc(bytes)) {
    return nullptr;
  }
  auto *ctx = static_cast<KernelRunContext *>(buf.p);
  ctx->input_size = inputs;
  ctx->output_size = outputs;
  ctx->compute_node_info = compute_node;
  ctx->kernel_extend_info = nullptr;
  chains.assign(n, Chain{});
  std::memset(chains.data(), 0, chains.size() * sizeof(Chain));
  for (size_t i = 0; i < n; ++i) {
    ctx->values[i] = reinterpret_cast<AsyncAnyValue *>(&chains[i]);
  }
  ctx->output_start = ctx->values + inputs;
  return ctx;
}

void *CompileInfoFor(const OpImplKernelRegistry::OpImplFunctionsV2 *funcs, const char *op_type, const char *json,
                     void *platform) {
  CompileKey key{op_type, json, platform};
  {
    std::lock_guard<std::mutex> lock(g_compile_mu);
    const auto found = g_compile_info.find(key);
    if (found != g_compile_info.end()) {
      return found->second;
    }
  }
  if (funcs->compile_info_creator == nullptr || funcs->tiling_parse == nullptr) {
    RT_GLOG(ERROR) << "FxAiCoreTiling missing compile-info parse for " << (op_type == nullptr ? "" : op_type);
    return nullptr;
  }
  void *info = funcs->compile_info_creator();
  if (info == nullptr) {
    RT_GLOG(ERROR) << "FxAiCoreTiling compile-info creator returned null for " << op_type;
    return nullptr;
  }
  OwnedBuf raw;
  std::vector<Chain> chains;
  size_t total = 0;
  if (ComputeNodeInfo::CalcSize(1, 1, 1, 1, total) != ge::GRAPH_SUCCESS) {
    return nullptr;
  }
  std::vector<uint8_t> cni_buf(total, 0);
  auto *cni = reinterpret_cast<ComputeNodeInfo *>(cni_buf.data());
  cni->Init(1, 1, 1, 1, 0, op_type, op_type);
  KernelRunContext *run = MakeRunContext(raw, chains, 2, 1, cni);
  if (run == nullptr) {
    return nullptr;
  }
  SetInline<const char *>(chains[0], json);
  SetInline<void *>(chains[1], platform);
  SetInline<void *>(chains[2], info);
  auto *parse_ctx = reinterpret_cast<TilingParseContext *>(run);
  const uint32_t st = funcs->tiling_parse(reinterpret_cast<KernelContext *>(parse_ctx));
  void *parsed = *chains[2].GetPointer<void *>();
  if (st != ge::GRAPH_SUCCESS || parsed == nullptr) {
    RT_GLOG(ERROR) << "FxAiCoreTiling parse failed op=" << op_type << " status=" << st;
    if (parsed != nullptr && funcs->compile_info_deleter != nullptr) {
      funcs->compile_info_deleter(parsed);
    }
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(g_compile_mu);
  g_compile_info[key] = parsed;
  return parsed;
}

bool RunOpImplTiling(const OpImplKernelRegistry::OpImplFunctionsV2 *funcs, const char *op_type, const char *json,
                     void *platform, const Tensor *const *tensors, const StorageShape *const *out_shapes, size_t nin,
                     size_t nout, size_t max_tiling, const uint8_t *attr_bytes, size_t attr_size,
                     const ge::DataType *out_dtypes, TypedContinuousVector<size_t> *workspace, uint32_t *block_dim,
                     uint64_t *tiling_key, uint8_t *tiling_out, uint64_t *tiling_size) {
  void *compile_info = CompileInfoFor(funcs, op_type, json, platform);
  if (compile_info == nullptr || funcs->tiling == nullptr) {
    return false;
  }
  size_t cni_bytes = 0;
  if (ComputeNodeInfo::CalcSize(nin, nout, nin, nout, cni_bytes) != ge::GRAPH_SUCCESS) {
    return false;
  }
  if (attr_bytes == nullptr) {
    attr_size = 0;
  }
  std::vector<uint8_t> cni_buf(cni_bytes + attr_size, 0);
  auto *cni = reinterpret_cast<ComputeNodeInfo *>(cni_buf.data());
  cni->Init(nin, nout, nin, nout, attr_size, op_type, op_type);
  if (attr_size > 0) {
    auto *attrs = const_cast<RuntimeAttrs *>(cni->GetAttrs());
    if (attrs == nullptr) {
      return false;
    }
    std::memcpy(static_cast<void *>(attrs), attr_bytes, attr_size);
  }
  for (size_t i = 0; i < nin; ++i) {
    auto *inst = const_cast<AnchorInstanceInfo *>(cni->GetInputInstanceInfo(i));
    if (inst == nullptr) {
      return false;
    }
    inst->SetInstanceStart(static_cast<uint32_t>(i));
    inst->SetInstantiationNum(tensors[i] == nullptr ? 0U : 1U);
    if (tensors[i] != nullptr) {
      auto *td = const_cast<CompileTimeTensorDesc *>(cni->GetInputTdInfo(i));
      if (td == nullptr) {
        return false;
      }
      td->SetDataType(tensors[i]->GetDataType());
      td->SetStorageFormat(tensors[i]->GetStorageFormat());
      td->SetOriginFormat(tensors[i]->GetOriginFormat());
    }
  }
  for (size_t o = 0; o < nout; ++o) {
    auto *inst = const_cast<AnchorInstanceInfo *>(cni->GetOutputInstanceInfo(o));
    if (inst == nullptr) {
      return false;
    }
    inst->SetInstanceStart(static_cast<uint32_t>(o));
    inst->SetInstantiationNum(1U);
    auto *td = const_cast<CompileTimeTensorDesc *>(cni->GetOutputTdInfo(o));
    if (td == nullptr || out_dtypes == nullptr) {
      return false;
    }
    td->SetDataType(out_dtypes[o]);
    const ge::Format format = (nin > 0 && tensors[0] != nullptr) ? tensors[0]->GetStorageFormat() : ge::FORMAT_ND;
    const ge::Format origin = (nin > 0 && tensors[0] != nullptr) ? tensors[0]->GetOriginFormat() : ge::FORMAT_ND;
    td->SetStorageFormat(format);
    td->SetOriginFormat(origin);
  }

  const size_t cap = max_tiling == 0 ? 8192U : max_tiling;
  auto tiling_holder = TilingData::CreateCap(cap);
  auto ws_holder = ContinuousVector::Create<size_t>(16U);
  if (tiling_holder == nullptr || ws_holder == nullptr) {
    return false;
  }
  auto *td = reinterpret_cast<TilingData *>(tiling_holder.get());
  auto *ws = reinterpret_cast<TypedContinuousVector<size_t> *>(ws_holder.get());

  OwnedBuf raw;
  std::vector<Chain> chains;
  const size_t inputs = nin + nout + kInnerFixedInputs;
  KernelRunContext *run = MakeRunContext(raw, chains, inputs, kInnerOutputCount, cni);
  if (run == nullptr) {
    return false;
  }
  for (size_t i = 0; i < nin; ++i) {
    chains[i].Set(const_cast<Tensor *>(tensors[i]), nullptr);
  }
  for (size_t o = 0; o < nout; ++o) {
    chains[nin + o].Set(const_cast<StorageShape *>(out_shapes[o]), nullptr);
  }
  const size_t fixed = nin + nout;
  SetInline<void *>(chains[fixed + 0], compile_info);
  SetInline<void *>(chains[fixed + 1], platform);
  SetInline<void *>(chains[fixed + 2], nullptr);
  SetInline<int32_t>(chains[fixed + 3], 0);
  SetInline<int32_t>(chains[fixed + 4], 0);
  SetInline<bool>(chains[fixed + 5], false);
  SetInline<TilingData *>(chains[inputs + static_cast<size_t>(TilingContext::kOutputTilingData)], td);
  chains[inputs + static_cast<size_t>(TilingContext::kOutputWorkspace)].Set(ws, nullptr);

  auto *tiling_ctx = reinterpret_cast<TilingContext *>(run);
  const uint32_t st = funcs->tiling(tiling_ctx);
  if (st != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "FxAiCoreTiling opimpl tiling failed op=" << op_type << " status=" << st;
    return false;
  }
  const size_t data_size = td->GetDataSize();
  if (data_size > kTilingDataCap || td->GetData() == nullptr) {
    RT_GLOG(ERROR) << "FxAiCoreTiling tiling data size " << data_size << " exceeds cap for " << op_type;
    return false;
  }
  if (data_size > 0) {
    std::memcpy(tiling_out, td->GetData(), data_size);
  }
  *tiling_size = data_size;
  const uint64_t key = *chains[inputs + static_cast<size_t>(TilingContext::kOutputTilingKey)].GetPointer<uint64_t>();
  *tiling_key = key;
  *block_dim = *chains[inputs + static_cast<size_t>(TilingContext::kOutputBlockDim)].GetPointer<uint32_t>();
  (void)tiling_ctx;
  if (workspace != nullptr) {
    const size_t nws = ws->GetSize();
    if (workspace->SetSize(nws) != ge::SUCCESS) {
      return false;
    }
    if (nws > 0 && ws->GetData() != nullptr) {
      std::memcpy(workspace->MutableData(), ws->GetData(), nws * sizeof(size_t));
    }
  }
  return true;
}

ge::graphStatus FxAiCoreTiling(KernelContext *context) {
  if (context == nullptr || context->GetInputNum() < kTilingFixedInputs) {
    RT_GLOG(ERROR) << "FxAiCoreTiling missing inputs";
    return ge::GRAPH_FAILED;
  }
  const char *op_type = context->GetInputValue<const char *>(0);
  const char *json = context->GetInputValue<const char *>(1);
  void *platform = context->GetInputValue<void *>(2);
  const uint32_t max_tiling = context->GetInputValue<uint32_t>(3);
  const uint32_t nin = context->GetInputValue<uint32_t>(4);
  const uint32_t nout = context->GetInputValue<uint32_t>(5);
  const auto *attr_bytes = context->GetInputValue<const uint8_t *>(6);
  const uint64_t attr_size = context->GetInputValue<uint64_t>(7);
  const auto *out_dtypes = context->GetInputValue<const ge::DataType *>(8);
  if (context->GetInputNum() != kTilingFixedInputs + nin + nout) {
    RT_GLOG(ERROR) << "FxAiCoreTiling input count mismatch op=" << (op_type == nullptr ? "" : op_type);
    return ge::GRAPH_FAILED;
  }
  std::vector<const Tensor *> tensors(nin, nullptr);
  std::vector<const StorageShape *> shapes(nout, nullptr);
  for (uint32_t i = 0; i < nin; ++i) {
    tensors[i] = context->GetInputPointer<Tensor>(kTilingFixedInputs + i);
  }
  for (uint32_t o = 0; o < nout; ++o) {
    shapes[o] = context->GetInputPointer<StorageShape>(kTilingFixedInputs + nin + o);
  }
  auto *workspace = context->GetOutputPointer<TypedContinuousVector<size_t>>(kOutWorkspace);
  auto *block_dim = context->GetOutputPointer<uint32_t>(kOutBlockDim);
  auto *tiling_key = context->GetOutputPointer<uint64_t>(kOutTilingKey);
  // The creator stores the buffer with Chain::Set, so the chain holds a uint8_t*.
  // GetOutputPointer<uint8_t> addresses the inline slot itself and would overwrite that pointer.
  auto *tiling_chain = context->GetOutput(kOutTilingData);
  auto *tiling_out = tiling_chain == nullptr ? nullptr : tiling_chain->GetValue<uint8_t *>();
  auto *tiling_size = context->GetOutputPointer<uint64_t>(kOutTilingSize);
  if (workspace == nullptr || block_dim == nullptr || tiling_key == nullptr || tiling_out == nullptr ||
      tiling_size == nullptr) {
    RT_GLOG(ERROR) << "FxAiCoreTiling outputs were not created";
    return ge::GRAPH_FAILED;
  }
  NoteOp(op_type);
  std::vector<int64_t> cache_key;
  const bool cacheable = BuildTilingKey(tensors.data(), nin, shapes.data(), nout, &cache_key);
  TilingState &cached = TilingStateOf(context);
  if (cacheable && cached.valid && cached.key == cache_key) {
    return ApplyTilingState(cached, workspace, block_dim, tiling_key, tiling_out, tiling_size) ? ge::GRAPH_SUCCESS
                                                                                              : ge::GRAPH_FAILED;
  }
  gert::OpImplSpaceRegistryV2Ptr registry;
  (void)EnsureOppRegistryReady(&registry);
  const auto *funcs = registry == nullptr ? nullptr : registry->GetOpImpl(op_type);
  if (funcs == nullptr || funcs->tiling == nullptr) {
    RT_GLOG(ERROR) << "FxAiCoreTiling no OpImpl tiling for " << (op_type == nullptr ? "" : op_type);
    return ge::GRAPH_FAILED;
  }
  const bool ran = RunOpImplTiling(funcs, op_type, json, platform, tensors.data(), shapes.data(), nin, nout, max_tiling,
                                   attr_bytes, static_cast<size_t>(attr_size), out_dtypes, workspace, block_dim,
                                   tiling_key, tiling_out, tiling_size);
  if (ran && cacheable) {
    SaveTilingState(&cached, std::move(cache_key), workspace, *block_dim, *tiling_key, tiling_out, *tiling_size);
  }
  return ran ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

ge::graphStatus CreateFxAiCoreTilingOutputs(KernelContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto *workspace = context->GetOutput(kOutWorkspace);
  auto *tiling = context->GetOutput(kOutTilingData);
  if (workspace == nullptr || tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto ws = ContinuousVector::Create<size_t>(16U);
  auto bytes = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[kTilingDataCap]());
  if (ws == nullptr || bytes == nullptr) {
    return ge::GRAPH_FAILED;
  }
  workspace->SetWithDefaultDeleter<uint8_t[]>(ws.release());
  tiling->SetWithDefaultDeleter<uint8_t[]>(bytes.release());
  return ge::GRAPH_SUCCESS;
}

using BinaryLoadFn = aclError (*)(const void *, size_t, const aclrtBinaryLoadOptions *, aclrtBinHandle *);
using BinaryGetByEntryFn = aclError (*)(aclrtBinHandle, uint64_t, aclrtFuncHandle *);
using BinaryGetByNameFn = aclError (*)(const aclrtBinHandle, const char *, aclrtFuncHandle *);
using LaunchHostArgsFn = aclError (*)(aclrtFuncHandle, uint32_t, aclrtStream, aclrtLaunchKernelCfg *, void *, size_t,
                                      aclrtPlaceHolderInfo *, size_t);
using LaunchHostFuncFn = aclError (*)(aclrtStream, aclrtHostFunc, void *);

struct AclLaunchFns {
  BinaryLoadFn load = nullptr;
  BinaryGetByEntryFn by_entry = nullptr;
  BinaryGetByNameFn by_name = nullptr;
  LaunchHostArgsFn launch = nullptr;
  LaunchHostFuncFn host_func = nullptr;
};

const AclLaunchFns &AclLaunch() {
  static const AclLaunchFns fns = [] {
    AclLaunchFns out;
    out.load = reinterpret_cast<BinaryLoadFn>(dlsym(RTLD_DEFAULT, "aclrtBinaryLoadFromData"));
    out.by_entry = reinterpret_cast<BinaryGetByEntryFn>(dlsym(RTLD_DEFAULT, "aclrtBinaryGetFunctionByEntry"));
    out.by_name = reinterpret_cast<BinaryGetByNameFn>(dlsym(RTLD_DEFAULT, "aclrtBinaryGetFunction"));
    out.launch = reinterpret_cast<LaunchHostArgsFn>(dlsym(RTLD_DEFAULT, "aclrtLaunchKernelWithHostArgs"));
    out.host_func = reinterpret_cast<LaunchHostFuncFn>(dlsym(RTLD_DEFAULT, "aclrtLaunchHostFunc"));
    return out;
  }();
  return fns;
}

aclrtBinHandle LoadBin(const fx_converter::aicore_rt::BinData *bin, uint32_t magic) {
  if (bin == nullptr || bin->length == 0) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(g_bin_mu);
  const auto found = g_bin_handles.find(bin);
  if (found != g_bin_handles.end()) {
    return found->second;
  }
  aclrtBinaryLoadOption option{};
  option.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
  option.value.magic = magic;
  aclrtBinaryLoadOptions options{};
  options.numOpt = 1;
  options.options = &option;
  const auto load = AclLaunch().load;
  if (load == nullptr) {
    RT_GLOG(ERROR) << "FxAiCoreLaunch aclrtBinaryLoadFromData is unavailable";
    return nullptr;
  }
  aclrtBinHandle handle = nullptr;
  const void *bytes = &bin->placeholder;
  const aclError st = load(bytes, static_cast<size_t>(bin->length), &options, &handle);
  if (st != ACL_SUCCESS || handle == nullptr) {
    RT_GLOG(ERROR) << "FxAiCoreLaunch binary load failed status=" << st << " bytes=" << bin->length
                   << " magic=" << magic;
    return nullptr;
  }
  g_bin_handles.emplace(bin, handle);
  return handle;
}

bool CopyHostKeep(HostKeep *keep, const void *host, size_t bytes, aclrtStream stream) {
  if (!EnsureDeviceBuf(&keep->dev, &keep->cap, bytes)) {
    RT_GLOG(ERROR) << "FxAiCoreLaunch host-parameter alloc failed size=" << bytes;
    return false;
  }
  if (aclrtMemcpyAsync(keep->dev, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream) != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "FxAiCoreLaunch host-parameter copy failed size=" << bytes;
    return false;
  }
  keep->host = host;
  keep->size = bytes;
  return true;
}

struct MixTaskPara {
  bool need_sub = false;
  uint32_t main_dim = 0;
  uint32_t sub_dim = 0;
  uint32_t main_offset = 0;
  uint32_t sub_offset = 0;
};

MixTaskPara CalcMixTaskPara(uint32_t block_dim, const fx_converter::aicore_rt::MixCoreArgs &mix) {
  MixTaskPara para;
  const uint32_t ai_core = mix.ai_core_num;
  const uint32_t all_core = mix.all_core_num == 0U ? 1U : mix.all_core_num;
  if (block_dim <= ai_core) {
    para.main_dim = block_dim;
    return para;
  }
  para.need_sub = true;
  if (mix.mix_type == fx_converter::aicore_rt::MixType::kMixVectorCore) {
    const uint64_t split = (static_cast<uint64_t>(block_dim) * ai_core + ai_core - 1U) / all_core;
    para.main_dim = static_cast<uint32_t>(split);
    para.sub_offset = para.main_dim;
    para.sub_dim = block_dim - para.main_dim;
  } else {
    para.main_dim = block_dim;
    para.sub_dim = mix.vec_core_num;
  }
  return para;
}

bool EnsureMixSync(LaunchState *kept) {
  if (kept->sub_stream == nullptr && aclrtCreateStream(&kept->sub_stream) != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "FxAiCoreLaunch mix sub stream create failed";
    return false;
  }
  if (kept->notify0 == nullptr && aclrtCreateNotify(&kept->notify0, 0U) != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "FxAiCoreLaunch mix notify0 create failed";
    return false;
  }
  if (kept->notify1 == nullptr && aclrtCreateNotify(&kept->notify1, 0U) != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "FxAiCoreLaunch mix notify1 create failed";
    return false;
  }
  return true;
}

aclError LaunchOnce(const AclLaunchFns &acl, aclrtFuncHandle func, uint32_t block_dim, aclrtStream stream,
                    aclrtLaunchKernelCfg *cfg, void *args, size_t args_size, aclrtPlaceHolderInfo *holders,
                    size_t holder_num) {
  return acl.launch(func, block_dim, stream, cfg, args, args_size, holders, holder_num);
}

ge::graphStatus FxAiCoreLaunch(KernelContext *context) {
  if (context == nullptr || context->GetInputNum() < kLaunchFixedInputs) {
    RT_GLOG(ERROR) << "FxAiCoreLaunch missing inputs";
    return ge::GRAPH_FAILED;
  }
  const auto *bin = context->GetInputValue<const fx_converter::aicore_rt::BinData *>(kInBin);
  const uint32_t magic = context->GetInputValue<uint32_t>(kInMagic);
  const char *kernel_name = context->GetInputValue<const char *>(kInKernelName);
  const uint32_t block_dim = context->GetInputValue<uint32_t>(kInBlockDim);
  const uint64_t tiling_key = context->GetInputValue<uint64_t>(kInTilingKey);
  const uint32_t function_count = context->GetInputValue<uint32_t>(kInFunctionCount);
  auto stream = context->GetInputValue<aclrtStream>(kInStream);
  const auto *workspaces = context->GetInputPointer<TypedContinuousVector<size_t>>(kInWorkspace);
  const auto *tiling = context->GetInputValue<uint8_t *>(kInTilingData);
  const uint64_t tiling_size = context->GetInputValue<uint64_t>(kInTilingSize);
  const uint32_t nin = context->GetInputValue<uint32_t>(kInInputCount);
  const uint32_t nout = context->GetInputValue<uint32_t>(kInOutputCount);
  const uint32_t workspace_cap = context->GetInputValue<uint32_t>(kInWorkspaceCap);
  const uint32_t mix_enable = context->GetInputValue<uint32_t>(kInMixEnable);
  const auto *mix_args = context->GetInputValue<const fx_converter::aicore_rt::MixCoreArgs *>(kInMixArgs);
  const uint32_t need_ffts = context->GetInputValue<uint32_t>(kInNeedFfts);
  if (bin == nullptr || workspaces == nullptr || (tiling_size > 0 && tiling == nullptr) ||
      kLaunchFixedInputs + nin + nout != context->GetInputNum()) {
    RT_GLOG(ERROR) << "FxAiCoreLaunch bad bin, workspace, or io count";
    return ge::GRAPH_FAILED;
  }
  const size_t nws = workspaces->GetSize();
  if (nws > workspace_cap) {
    RT_GLOG(ERROR) << "FxAiCoreLaunch workspace count " << nws << " exceeds cap " << workspace_cap;
    return ge::GRAPH_FAILED;
  }
  aclrtBinHandle handle = LoadBin(bin, magic);
  if (handle == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto &acl = AclLaunch();
  if (acl.by_entry == nullptr || acl.launch == nullptr) {
    RT_GLOG(ERROR) << "FxAiCoreLaunch runtime launch symbols are unavailable";
    return ge::GRAPH_FAILED;
  }
  LaunchState &kept = LaunchStateOf(context);
  const uint64_t entry = function_count == 1U ? 0U : tiling_key;
  aclrtFuncHandle func = kept.func;
  if (func == nullptr || kept.bin != bin || kept.entry != entry) {
    aclError lookup = acl.by_entry(handle, entry, &func);
    if ((lookup != ACL_SUCCESS || func == nullptr) && acl.by_name != nullptr) {
      lookup = acl.by_name(handle, kernel_name == nullptr ? "" : kernel_name, &func);
    }
    if (lookup != ACL_SUCCESS || func == nullptr) {
      RT_GLOG(ERROR) << "FxAiCoreLaunch function lookup failed status=" << lookup << " entry=" << entry
                     << " name=" << (kernel_name == nullptr ? "" : kernel_name);
      return ge::GRAPH_FAILED;
    }
    kept.bin = bin;
    kept.entry = entry;
    kept.func = func;
  }

  const size_t io_count = static_cast<size_t>(nin) + nout;
  if (kept.host.size() < io_count) {
    kept.host.resize(io_count);
  }
  std::vector<GertTensorData *> io(io_count, nullptr);
  std::vector<uint32_t> embed_index;
  size_t embed_bytes = 0;
  for (size_t i = 0; i < io_count; ++i) {
    auto *data = context->GetInputValue<GertTensorData *>(kInIo + i);
    io[i] = data;
    if (data == nullptr || data->GetAddr() == nullptr || data->GetSize() == 0 ||
        !TensorPlacementUtils::IsOnHost(data->GetPlacement())) {
      continue;
    }
    const size_t aligned = (data->GetSize() + 7U) & ~size_t{7};
    if (embed_bytes + aligned > kMaxHostEmbed) {
      continue;
    }
    embed_index.push_back(static_cast<uint32_t>(i));
    embed_bytes += aligned;
  }

  void *ffts_addr = nullptr;
  const size_t lead = need_ffts == 0U ? 0U : 1U;
  if (lead != 0U) {
    if (aclrtGetHardwareSyncAddr(&ffts_addr) != ACL_SUCCESS || ffts_addr == nullptr) {
      RT_GLOG(ERROR) << "FxAiCoreLaunch aclrtGetHardwareSyncAddr failed";
      return ge::GRAPH_FAILED;
    }
  }
  const size_t ws_off = (lead + io_count) * sizeof(uint64_t);
  const size_t tiling_data_off = ws_off + static_cast<size_t>(workspace_cap) * sizeof(uint64_t) + sizeof(uint64_t);
  const size_t tiling_bytes = static_cast<size_t>(tiling_size);
  const size_t args_size = tiling_data_off + tiling_bytes + embed_bytes;
  if (kept.args.size() < args_size) {
    kept.args.resize(args_size);
  }
  std::memset(kept.args.data(), 0, tiling_data_off);
  auto *slots = reinterpret_cast<uint64_t *>(kept.args.data());
  if (lead != 0U) {
    slots[0] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ffts_addr));
  }
  std::vector<aclrtPlaceHolderInfo> placeholders;
  placeholders.reserve(embed_index.size() + 1U);
  if (tiling_bytes > 0) {
    std::memcpy(kept.args.data() + tiling_data_off, tiling, tiling_bytes);
    aclrtPlaceHolderInfo info{};
    info.addrOffset = static_cast<uint32_t>(ws_off + nws * sizeof(uint64_t));
    info.dataOffset = static_cast<uint32_t>(tiling_data_off);
    placeholders.push_back(info);
  }
  size_t embed_at = tiling_data_off + tiling_bytes;
  size_t embed_cursor = 0;
  for (size_t i = 0; i < io_count; ++i) {
    GertTensorData *data = io[i];
    const size_t slot = lead + i;
    const bool embed = embed_cursor < embed_index.size() && embed_index[embed_cursor] == i;
    if (data == nullptr || data->GetAddr() == nullptr || data->GetSize() == 0) {
      slots[slot] = 0;
      continue;
    }
    if (embed) {
      const size_t bytes = data->GetSize();
      const size_t aligned = (bytes + 7U) & ~size_t{7};
      std::memcpy(kept.args.data() + embed_at, data->GetAddr(), bytes);
      aclrtPlaceHolderInfo info{};
      info.addrOffset = static_cast<uint32_t>(slot * sizeof(uint64_t));
      info.dataOffset = static_cast<uint32_t>(embed_at);
      placeholders.push_back(info);
      embed_at += aligned;
      ++embed_cursor;
      continue;
    }
    if (TensorPlacementUtils::IsOnHost(data->GetPlacement())) {
      if (!CopyHostKeep(&kept.host[i], data->GetAddr(), data->GetSize(), stream)) {
        return ge::GRAPH_FAILED;
      }
      slots[slot] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kept.host[i].dev));
      continue;
    }
    slots[slot] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data->GetAddr()));
  }
  const auto *ws_data = workspaces->GetData();
  if (kept.ws_dev.size() < nws) {
    kept.ws_dev.resize(nws, nullptr);
    kept.ws_cap.resize(nws, 0);
  }
  for (size_t i = 0; i < nws; ++i) {
    const size_t bytes = ws_data == nullptr ? 0U : ws_data[i];
    if (bytes == 0U) {
      slots[lead + io_count + i] = 0;
      continue;
    }
    if (!EnsureDeviceBuf(&kept.ws_dev[i], &kept.ws_cap[i], bytes)) {
      RT_GLOG(ERROR) << "FxAiCoreLaunch workspace alloc failed size=" << bytes;
      return ge::GRAPH_FAILED;
    }
    slots[lead + io_count + i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kept.ws_dev[i]));
  }
  aclrtPlaceHolderInfo none{};
  aclrtPlaceHolderInfo *holders = placeholders.empty() ? &none : placeholders.data();
  const size_t holder_num = placeholders.size();
  const bool mix_on = mix_enable != 0U && mix_args != nullptr;
  if (!mix_on) {
    const aclError st = LaunchOnce(acl, func, block_dim, stream, nullptr, kept.args.data(), args_size, holders,
                                   holder_num);
    if (st != ACL_SUCCESS) {
      RT_GLOG(ERROR) << "FxAiCoreLaunch aclrtLaunchKernelWithHostArgs failed status=" << st
                     << " block=" << block_dim << " args=" << args_size;
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }

  const MixTaskPara para = CalcMixTaskPara(block_dim, *mix_args);
  if (para.main_dim == 0U && !(para.need_sub && para.sub_dim > 0U)) {
    RT_GLOG(ERROR) << "FxAiCoreLaunch mix block dim is 0";
    return ge::GRAPH_FAILED;
  }
  aclrtLaunchKernelAttr attrs[4]{};
  attrs[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  attrs[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
  attrs[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
  attrs[3].id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET;
  aclrtLaunchKernelCfg cfg{};
  cfg.attrs = attrs;
  cfg.numAttrs = 4U;
  auto launch_engine = [&](aclrtEngineType engine, uint32_t dim, uint32_t offset, aclrtStream target) -> aclError {
    if (dim == 0U) {
      return ACL_SUCCESS;
    }
    attrs[2].value.engineType = engine;
    attrs[3].value.blockDimOffset = offset;
    return LaunchOnce(acl, func, dim, target, &cfg, kept.args.data(), args_size, holders, holder_num);
  };
  if (!para.need_sub) {
    const aclError st = launch_engine(ACL_RT_ENGINE_TYPE_AIC, para.main_dim, para.main_offset, stream);
    if (st != ACL_SUCCESS) {
      RT_GLOG(ERROR) << "FxAiCoreLaunch mix main launch failed status=" << st;
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }
  if (!EnsureMixSync(&kept)) {
    return ge::GRAPH_FAILED;
  }
  if (aclrtRecordNotify(kept.notify0, stream) != ACL_SUCCESS ||
      aclrtWaitAndResetNotify(kept.notify0, kept.sub_stream, UINT32_MAX) != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "FxAiCoreLaunch mix pre-sync failed";
    return ge::GRAPH_FAILED;
  }
  aclError st = launch_engine(ACL_RT_ENGINE_TYPE_AIC, para.main_dim, para.main_offset, stream);
  if (st != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "FxAiCoreLaunch mix AIC launch failed status=" << st;
    return ge::GRAPH_FAILED;
  }
  st = launch_engine(ACL_RT_ENGINE_TYPE_AIV, para.sub_dim, para.sub_offset, kept.sub_stream);
  if (st != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "FxAiCoreLaunch mix AIV launch failed status=" << st;
    return ge::GRAPH_FAILED;
  }
  if (aclrtRecordNotify(kept.notify1, kept.sub_stream) != ACL_SUCCESS ||
      aclrtWaitAndResetNotify(kept.notify1, stream, UINT32_MAX) != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "FxAiCoreLaunch mix post-sync failed";
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

FXRT_EXPORT_KERNEL_RUN(FxAiCoreTiling, FxAiCoreTiling)
FXRT_EXPORT_KERNEL_OUTPUTS(FxAiCoreTiling, CreateFxAiCoreTilingOutputs)
FXRT_EXPORT_KERNEL_RUN(FxAiCoreLaunch, FxAiCoreLaunch)

}  // namespace gert
