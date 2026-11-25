// Load OPP host SOs and resolve dual-stage prepare/launch by op type string.
// InferShape for FxConverter uses FxExprInfer (fxrt SymbolicExpr), not OPP infer.
// Path C tiling is FxAiCoreTiling (OpImpl); liboptiling is loaded only to merge those OpImpl entries.

#include "opp/opp_registry.h"

#include <dlfcn.h>
#include <unistd.h>

#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "common/ascend_paths.h"
#include "common/logger.h"
#include "opp/add_dual_stage_impl.h"
#include "opp/amax_dual_stage_impl.h"
#include "opp/baddbmm_dual_stage_impl.h"
#include "opp/flash_attention_score_dual_stage_impl.h"
#include "opp/bitwise_not_dual_stage_impl.h"
#include "opp/concat_dual_stage_impl.h"
#include "opp/convolution_dual_stage_impl.h"
#include "opp/gather_dual_stage_impl.h"
#include "opp/index_dual_stage_impl.h"
#include "opp/where_dual_stage_impl.h"
#include "opp/matmul_dual_stage_impl.h"
#include "opp/mutation_dual_stage_impl.h"
#include "opp/reduce_sum_dual_stage_impl.h"
#include "graph/ascend_string.h"

namespace {

std::mutex g_mu;
gert::OpImplSpaceRegistryV2Ptr g_registry;
// Raw pointer slot for Const(space_registry) (must outlive Execute).
gert::OpImplSpaceRegistryV2 *g_registry_raw = nullptr;
// Keep liboptiling/libopmaster handle for process lifetime so OpImpl tiling stays registered.
void *g_optiling_handle = nullptr;
std::string &GetOptilingSoPath() {
  static std::string path;
  return path;
}

using FxOwnedInstallFn = bool (*)(const gert::OpImplSpaceRegistryV2Ptr &);

struct FxOwnedInstallEntry {
  const char *opp_type;
  FxOwnedInstallFn install;
};

// Table-driven path-B installs. Append a row when adding a new Fx dual-stage wrapper.
// Path C (kAiCoreBinary) does NOT install prepare/launch — it wires GE AiCore RT kernels.
constexpr FxOwnedInstallEntry kFxOwnedInstallTable[] = {
    {"Add", &fx_converter::InstallFxAddDualStage},
    {"Invert", &fx_converter::InstallFxBitwiseNotDualStage},
    {"GatherElements", &fx_converter::InstallFxGatherDualStage},
    {"Index", &fx_converter::InstallFxIndexDualStage},
    {"SelectV2", &fx_converter::InstallFxWhereDualStage},
    {"ReduceSum", &fx_converter::InstallFxReduceSumDualStage},
    {"ReduceMax", &fx_converter::InstallFxAmaxDualStage},
    {"MatMul", &fx_converter::InstallFxMatMulDualStage},
    {"BatchMatMul", &fx_converter::InstallFxBatchMatMulDualStage},
    {"Addmm", &fx_converter::InstallFxAddmmDualStage},
    {"Concat", &fx_converter::InstallFxConcatDualStage},
    {"Fills", &fx_converter::InstallFxFillDualStage},
    {"Assign", &fx_converter::InstallFxCopyDualStage},
    // Conv2D mounts the convolution lowering: OPP registers no
    // "Convolution" OpImpl to patch.
    {"Conv2D", &fx_converter::InstallFxConvolutionDualStage},
    // BatchMatMulV2 mounts baddbmm: no "Baddbmm" entry exists, and BatchMatMulV2 is
    // separate from the BatchMatMul entry the plain bmm row already replaces.
    {"BatchMatMulV2", &fx_converter::InstallFxBaddbmmDualStage},
    {"FlashAttentionScore", &fx_converter::InstallFxFlashAttentionScoreDualStage},
};

std::string HostArchDir() {
#if defined(__aarch64__) || defined(__arm64__)
  return "aarch64";
#elif defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#else
  // TODO(jiaorui): detect runtime arch (uname) when compile-time macros are missing.
  return "aarch64";
#endif
}

void AppendExistingHostSos(const std::string &dir, const char *const *names, size_t n_names,
                           std::vector<std::string> *out) {
  for (size_t i = 0; i < n_names; ++i) {
    const std::string path = dir + names[i];
    if (access(path.c_str(), F_OK) == 0) {
      out->push_back(path);
    }
  }
}

std::vector<std::string> OppHostSoCandidates() {
  static const char *kNames[] = {
      "libophost_legacy.so",      "libophost_math.so",        "libophost_nn.so",
      "libophost_cv.so",          "libophost_transformer.so", "libophost_oam.so",
      "libophost_comm_legacy.so",
  };
  const std::string arch = HostArchDir();
  const std::string rel = "/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/" + arch + "/";

  std::vector<std::string> out;
  for (const auto &root : fx_converter::CollectAscendToolkitRoots()) {
    AppendExistingHostSos(root + rel, kNames, sizeof(kNames) / sizeof(kNames[0]), &out);
    if (!out.empty()) {
      return out;
    }
  }
  return out;
}

bool AddSosToRegistry(const std::vector<std::string> &sos, const char *label) {
  if (g_registry == nullptr || sos.empty()) {
    return false;
  }
  for (const auto &so : sos) {
    const ge::graphStatus add_so =
        g_registry->AddSoToRegistry(gert::OppSoDesc({ge::AscendString(so.c_str())}, ge::AscendString("built-in")));
    if (add_so != ge::GRAPH_SUCCESS) {
      RT_GLOG(ERROR) << "AddSoToRegistry failed for " << so << ", ret=" << add_so;
      return false;
    }
  }
  RT_GLOG(INFO) << "OPP SpaceRegistry loaded " << sos.size() << " " << label << " SO(s), first=" << sos.front();
  return true;
}

// dlopen libopmaster / liboptiling so their OpImpl tiling entries can be merged below.
std::vector<std::string> OpTilingSoCandidates() {
  const std::string arch = HostArchDir();
  // Prefer op_master (newer packs), then op_tiling (classic liboptiling.so).
  const char *kRelPaths[] = {
      "/opp/built-in/op_impl/ai_core/tbe/op_master/lib/linux/",
      "/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/",
  };
  const char *kNames[] = {"libopmaster.so", "liboptiling.so"};

  std::vector<std::string> out;
  for (const auto &root : fx_converter::CollectAscendToolkitRoots()) {
    for (const char *rel : kRelPaths) {
      for (const char *name : kNames) {
        const std::string path = root + rel + arch + "/" + name;
        if (access(path.c_str(), F_OK) == 0) {
          out.push_back(path);
        }
      }
    }
    if (!out.empty()) {
      return out;
    }
  }
  return out;
}

bool LoadOpTilingSo() {
  if (g_optiling_handle != nullptr) {
    return true;
  }
  const auto candidates = OpTilingSoCandidates();
  if (candidates.empty()) {
    RT_GLOG(WARNING) << "liboptiling.so and libopmaster.so were not found; "
                     << "aclnn operators still load from the host libraries";
    return false;
  }
  for (const auto &path : candidates) {
    void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
      RT_GLOG(WARNING) << "dlopen failed for " << path << ": " << dlerror();
      continue;
    }
    g_optiling_handle = handle;
    GetOptilingSoPath() = path;
    RT_GLOG(INFO) << "OPP OpTiling loaded " << path;
    return true;
  }
  return false;
}

// Merge liboptiling OpImpl registrations (e.g. GatherElements / Tile tiling) into SpaceRegistry.
bool AddOpTilingSoToSpaceRegistry() {
  if (g_registry == nullptr || GetOptilingSoPath().empty()) {
    return false;
  }
  const ge::graphStatus add_so = g_registry->AddSoToRegistry(
      gert::OppSoDesc({ge::AscendString(GetOptilingSoPath().c_str())}, ge::AscendString("built-in")));
  if (add_so != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "AddSoToRegistry failed for OpTiling SO " << GetOptilingSoPath() << ", ret=" << add_so;
    return false;
  }
  RT_GLOG(INFO) << "OPP OpTiling merged into SpaceRegistry: " << GetOptilingSoPath();
  return true;
}

void InstallFxOwnedDualStageOps() {
  for (const auto &entry : kFxOwnedInstallTable) {
    if (entry.install == nullptr) {
      continue;
    }
    if (!entry.install(g_registry)) {
      RT_GLOG(WARNING) << "Fx dual-stage install failed for opp_type=" << entry.opp_type;
    }
  }
}

bool EnsureRegistryLocked(gert::OpImplSpaceRegistryV2Ptr *out_registry) {
  if (g_registry != nullptr) {
    if (out_registry != nullptr) {
      *out_registry = g_registry;
    }
    return true;
  }
  const auto sos = OppHostSoCandidates();
  if (sos.empty()) {
    RT_GLOG(ERROR) << "No OPP host SO found (set ASCEND_HOME_PATH / ASCEND_HOME; searched opp/.../op_host/lib/linux/"
                   << HostArchDir() << "/)";
    return false;
  }

  // Create empty SpaceRegistry, then AddSo in GE-friendly order:
  //   1) liboptiling first — Tik GatherElements / Tile OpImpl.tiling wins MergeFunctions
  //   2) libophost_* next — DualStage prepare/launch + infer_shape (tiling already set is kept)
  // MergeFunctions keeps the first tiling registration; ophost-first dropped Tik tiling and
  // left AscendC GatherElements tiling → block_dim=0 with legacy .o.
  g_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
  // aclnn prepare/launch comes from the host libraries. A missing tiling library
  // must not drop the registry.
  if (LoadOpTilingSo()) {
    (void)AddOpTilingSoToSpaceRegistry();
  }
  if (!AddSosToRegistry(sos, "host")) {
    g_registry.reset();
    g_registry_raw = nullptr;
    return false;
  }
  // Deliberately not registered as the process default: that breaks libnnopbase's op cache.
  g_registry_raw = g_registry.get();
  InstallFxOwnedDualStageOps();
  if (out_registry != nullptr) {
    *out_registry = g_registry;
  }
  return true;
}

}  // namespace

bool EnsureAclRuntimeReady(int32_t device_id) {
  const auto init_ret = aclInit(nullptr);
  if (init_ret != ACL_SUCCESS && init_ret != ACL_ERROR_REPEAT_INITIALIZE) {
    RT_GLOG(ERROR) << "aclInit failed, ret=" << init_ret;
    return false;
  }
  if (aclrtSetDevice(device_id) != ACL_SUCCESS) {
    RT_GLOG(ERROR) << "aclrtSetDevice failed";
    return false;
  }
  return true;
}

bool EnsureOppRegistryReady(gert::OpImplSpaceRegistryV2Ptr *out_registry) {
  std::lock_guard<std::mutex> lock(g_mu);
  return EnsureRegistryLocked(out_registry);
}

gert::OpImplSpaceRegistryV2 **OppSpaceRegistryAddrSlot() { return &g_registry_raw; }

bool ResolveOppDualStageFromRegistry(const char *opp_type, gert::OpImplRegisterV2::OpExecPrepareFunc *prepare,
                                     gert::OpImplRegisterV2::OpExecLaunchFunc *launch,
                                     gert::OpImplSpaceRegistryV2Ptr *out_registry) {
  if (opp_type == nullptr || prepare == nullptr || launch == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(g_mu);
  if (!EnsureRegistryLocked(out_registry)) {
    return false;
  }
  // No process-local pointer cache here: opp_binding.cc owns the stable Const addresses.
  const auto *funcs = g_registry->GetOpImpl(opp_type);
  if (funcs == nullptr || funcs->op_execute_prepare_func == nullptr || funcs->op_execute_launch_func == nullptr) {
    RT_GLOG(ERROR) << "OPP GetOpImpl(" << opp_type << ") missing prepare/launch";
    return false;
  }
  *prepare = funcs->op_execute_prepare_func;
  *launch = funcs->op_execute_launch_func;
  if (out_registry != nullptr) {
    *out_registry = g_registry;
  }
  return true;
}
