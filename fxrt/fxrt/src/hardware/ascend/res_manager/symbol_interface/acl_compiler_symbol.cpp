#include "acl_compiler_symbol.h"
#include <string>
#include "symbol_utils.h"

namespace fxrt::device::ascend {
aclopCompileAndExecuteFunObj aclopCompileAndExecute_ = nullptr;
aclopCompileAndExecuteV2FunObj aclopCompileAndExecuteV2_ = nullptr;
aclSetCompileoptFunObj aclSetCompileopt_ = nullptr;
aclopSetCompileFlagFunObj aclopSetCompileFlag_ = nullptr;
aclGenGraphAndDumpForOpFunObj aclGenGraphAndDumpForOp_ = nullptr;

void LoadAclOpCompilerApiSymbol(const std::string& ascendPath) {
  std::string complierPluginPath = ascendPath + "lib64/libacl_op_compiler.so";
  auto handler = GetLibHandler(complierPluginPath);
  if (handler == nullptr) {
    RT_VLOG(VL_HARDWARE) << "Dlopen " << complierPluginPath << " failed!" << dlerror();
    return;
  }
  aclopCompileAndExecute_ = DlsymAscendFuncObj(aclopCompileAndExecute, handler);
  aclopCompileAndExecuteV2_ = DlsymAscendFuncObj(aclopCompileAndExecuteV2, handler);
  aclSetCompileopt_ = DlsymAscendFuncObj(aclSetCompileopt, handler);
  aclopSetCompileFlag_ = DlsymAscendFuncObj(aclopSetCompileFlag, handler);
  aclGenGraphAndDumpForOp_ = DlsymAscendFuncObj(aclGenGraphAndDumpForOp, handler);
  RT_VLOG(VL_HARDWARE) << "Load acl op compiler api success!";
}

void LoadSimulationAclOpCompilerApi() {
  ASSIGN_SIMU(aclopCompileAndExecute);
  ASSIGN_SIMU(aclopCompileAndExecuteV2);
  ASSIGN_SIMU(aclSetCompileopt);
  ASSIGN_SIMU(aclopSetCompileFlag);
  ASSIGN_SIMU(aclGenGraphAndDumpForOp);
}
} // namespace fxrt::device::ascend
