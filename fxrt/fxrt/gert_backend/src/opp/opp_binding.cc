#include "opp/opp_binding.h"

#include <mutex>
#include <string>
#include <unordered_map>

#include "common/logger.h"
#include "opp/aclnn_api.h"
#include "opp/opp_registry.h"
#include "shape/fx_expr_infer_shape.h"
#include "exe_graph/ge_rt2_kernels.h"

namespace fx_converter {
namespace {

gert::OpImplSpaceRegistryV2 *g_space_registry_addr = nullptr;
gert::OpImplRegisterV2::InferShapeKernelFunc g_infer_func = nullptr;

struct OppFuncPtrs {
  gert::OpImplRegisterV2::OpExecPrepareFunc prepare = nullptr;
  gert::OpImplRegisterV2::OpExecLaunchFunc launch = nullptr;
  // Keep OPP type string alive for Const(op_type).
  std::string opp_type_storage;
};

std::mutex g_mu;
std::unordered_map<std::string, OppFuncPtrs> g_funcs_by_type;

}  // namespace

bool ResolveOppDualStageBinding(const char *opp_type, OppDualStageFuncs *out) {
  if (opp_type == nullptr || out == nullptr) {
    return false;
  }
  EnsureGeRt2KernelsReady();

  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_funcs_by_type.find(opp_type);
  if (it == g_funcs_by_type.end()) {
    gert::OpImplSpaceRegistryV2Ptr reg;
    gert::OpImplRegisterV2::OpExecPrepareFunc prepare = nullptr;
    gert::OpImplRegisterV2::OpExecLaunchFunc launch = nullptr;
    if (!ResolveOppDualStageFromRegistry(opp_type, &prepare, &launch, &reg) || reg == nullptr) {
      return false;
    }
    std::string error;
    if (!aclnn_api::EnsureForOp(opp_type, &error)) {
      RT_GLOG(ERROR) << "ACLNN binding failed for type=" << opp_type << ": " << error;
      return false;
    }
    g_space_registry_addr = reg.get();
    if (g_infer_func == nullptr) {
      g_infer_func = GetFxExprInferShapeFunc();
    }
    OppFuncPtrs slot;
    slot.prepare = prepare;
    slot.launch = launch;
    slot.opp_type_storage = opp_type;
    it = g_funcs_by_type.emplace(opp_type, std::move(slot)).first;
  }

  if (g_infer_func == nullptr || it->second.prepare == nullptr || it->second.launch == nullptr ||
      g_space_registry_addr == nullptr) {
    RT_GLOG(ERROR) << "OPP dual-stage binding incomplete for type=" << opp_type;
    return false;
  }

  out->infer = &g_infer_func;
  out->prepare = &it->second.prepare;
  out->launch = &it->second.launch;
  out->space_registry = reinterpret_cast<void **>(&g_space_registry_addr);
  // SharedExecOptions and the float dtype Const come from SharedRuntime, not this struct.
  out->op_type = it->second.opp_type_storage.c_str();
  return true;
}

}  // namespace fx_converter
