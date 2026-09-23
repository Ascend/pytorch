#include "acl_op_symbol.h"
#include <string>
#include "symbol_utils.h"

namespace fxrt::device::ascend {
aclopCreateAttrFunObj aclopCreateAttr_ = nullptr;
aclopSetAttrBoolFunObj aclopSetAttrBool_ = nullptr;
aclopSetAttrDataTypeFunObj aclopSetAttrDataType_ = nullptr;
aclopSetAttrFloatFunObj aclopSetAttrFloat_ = nullptr;
aclopSetAttrIntFunObj aclopSetAttrInt_ = nullptr;
aclopSetAttrListBoolFunObj aclopSetAttrListBool_ = nullptr;
aclopSetAttrListDataTypeFunObj aclopSetAttrListDataType_ = nullptr;
aclopSetAttrListFloatFunObj aclopSetAttrListFloat_ = nullptr;
aclopSetAttrListIntFunObj aclopSetAttrListInt_ = nullptr;
aclopSetAttrListListIntFunObj aclopSetAttrListListInt_ = nullptr;
aclopSetAttrListStringFunObj aclopSetAttrListString_ = nullptr;
aclopSetAttrStringFunObj aclopSetAttrString_ = nullptr;
aclopSetModelDirFunObj aclopSetModelDir_ = nullptr;

void LoadAclOpApiSymbol(const std::string& ascendPath) {
  std::string ascendclPluginPath = ascendPath + "lib64/libascendcl.so";
  auto handler = GetLibHandler(ascendclPluginPath);
  if (handler == nullptr) {
    RT_VLOG(VL_HARDWARE) << "Dlopen " << ascendclPluginPath << " failed!" << dlerror();
    return;
  }
  aclopCreateAttr_ = DlsymAscendFuncObj(aclopCreateAttr, handler);
  aclopSetAttrBool_ = DlsymAscendFuncObj(aclopSetAttrBool, handler);
  aclopSetAttrDataType_ = DlsymAscendFuncObj(aclopSetAttrDataType, handler);
  aclopSetAttrFloat_ = DlsymAscendFuncObj(aclopSetAttrFloat, handler);
  aclopSetAttrInt_ = DlsymAscendFuncObj(aclopSetAttrInt, handler);
  aclopSetAttrListBool_ = DlsymAscendFuncObj(aclopSetAttrListBool, handler);
  aclopSetAttrListDataType_ = DlsymAscendFuncObj(aclopSetAttrListDataType, handler);
  aclopSetAttrListFloat_ = DlsymAscendFuncObj(aclopSetAttrListFloat, handler);
  aclopSetAttrListInt_ = DlsymAscendFuncObj(aclopSetAttrListInt, handler);
  aclopSetAttrListListInt_ = DlsymAscendFuncObj(aclopSetAttrListListInt, handler);
  aclopSetAttrListString_ = DlsymAscendFuncObj(aclopSetAttrListString, handler);
  aclopSetAttrString_ = DlsymAscendFuncObj(aclopSetAttrString, handler);
  aclopSetModelDir_ = DlsymAscendFuncObj(aclopSetModelDir, handler);
  RT_VLOG(VL_HARDWARE) << "Load ascend op api success!";
}

void LoadSimulationAclOpApi() {
  ASSIGN_SIMU(aclopCreateAttr);
  ASSIGN_SIMU(aclopSetAttrBool);
  ASSIGN_SIMU(aclopSetAttrDataType);
  ASSIGN_SIMU(aclopSetAttrFloat);
  ASSIGN_SIMU(aclopSetAttrInt);
  ASSIGN_SIMU(aclopSetAttrListBool);
  ASSIGN_SIMU(aclopSetAttrListDataType);
  ASSIGN_SIMU(aclopSetAttrListFloat);
  ASSIGN_SIMU(aclopSetAttrListInt);
  ASSIGN_SIMU(aclopSetAttrListListInt);
  ASSIGN_SIMU(aclopSetAttrListString);
  ASSIGN_SIMU(aclopSetAttrString);
  ASSIGN_SIMU(aclopSetModelDir);
}
} // namespace fxrt::device::ascend
