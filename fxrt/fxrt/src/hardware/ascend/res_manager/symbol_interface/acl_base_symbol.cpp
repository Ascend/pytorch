#include "acl_base_symbol.h"
#include <string>
#include "symbol_utils.h"

namespace fxrt::device::ascend {
aclCreateDataBufferFunObj aclCreateDataBuffer_ = nullptr;
aclCreateTensorDescFunObj aclCreateTensorDesc_ = nullptr;
aclDataTypeSizeFunObj aclDataTypeSize_ = nullptr;
aclDestroyDataBufferFunObj aclDestroyDataBuffer_ = nullptr;
aclDestroyTensorDescFunObj aclDestroyTensorDesc_ = nullptr;
aclGetTensorDescDimV2FunObj aclGetTensorDescDimV2_ = nullptr;
aclGetTensorDescNumDimsFunObj aclGetTensorDescNumDims_ = nullptr;
aclSetTensorConstFunObj aclSetTensorConst_ = nullptr;
aclSetTensorDescNameFunObj aclSetTensorDescName_ = nullptr;
aclSetTensorFormatFunObj aclSetTensorFormat_ = nullptr;
aclSetTensorPlaceMentFunObj aclSetTensorPlaceMent_ = nullptr;
aclSetTensorShapeFunObj aclSetTensorShape_ = nullptr;
aclrtGetSocNameFunObj aclrtGetSocName_ = nullptr;
aclUpdateDataBufferFunObj aclUpdateDataBuffer_ = nullptr;
aclGetDataBufferAddrFunObj aclGetDataBufferAddr_ = nullptr;
aclGetTensorDescSizeFunObj aclGetTensorDescSize_ = nullptr;
aclGetRecentErrMsgFunObj aclGetRecentErrMsg_ = nullptr;

void LoadAclBaseApiSymbol(const std::string& ascendPath) {
  std::string aclbasePluginPath = "lib64/libascendcl.so";
  auto baseHandler = GetLibHandler(ascendPath + aclbasePluginPath);
  if (baseHandler == nullptr) {
    RT_VLOG(VL_HARDWARE) << "Dlopen " << aclbasePluginPath << " failed!" << dlerror();
    return;
  }
  aclCreateDataBuffer_ = DlsymAscendFuncObj(aclCreateDataBuffer, baseHandler);
  aclCreateTensorDesc_ = DlsymAscendFuncObj(aclCreateTensorDesc, baseHandler);
  aclDataTypeSize_ = DlsymAscendFuncObj(aclDataTypeSize, baseHandler);
  aclDestroyDataBuffer_ = DlsymAscendFuncObj(aclDestroyDataBuffer, baseHandler);
  aclDestroyTensorDesc_ = DlsymAscendFuncObj(aclDestroyTensorDesc, baseHandler);
  aclGetTensorDescDimV2_ = DlsymAscendFuncObj(aclGetTensorDescDimV2, baseHandler);
  aclGetTensorDescNumDims_ = DlsymAscendFuncObj(aclGetTensorDescNumDims, baseHandler);
  aclSetTensorConst_ = DlsymAscendFuncObj(aclSetTensorConst, baseHandler);
  aclSetTensorDescName_ = DlsymAscendFuncObj(aclSetTensorDescName, baseHandler);
  aclSetTensorFormat_ = DlsymAscendFuncObj(aclSetTensorFormat, baseHandler);
  aclSetTensorPlaceMent_ = DlsymAscendFuncObj(aclSetTensorPlaceMent, baseHandler);
  aclSetTensorShape_ = DlsymAscendFuncObj(aclSetTensorShape, baseHandler);
  aclrtGetSocName_ = DlsymAscendFuncObj(aclrtGetSocName, baseHandler);
  aclUpdateDataBuffer_ = DlsymAscendFuncObj(aclUpdateDataBuffer, baseHandler);
  aclGetDataBufferAddr_ = DlsymAscendFuncObj(aclGetDataBufferAddr, baseHandler);
  aclGetTensorDescSize_ = DlsymAscendFuncObj(aclGetTensorDescSize, baseHandler);
  aclGetRecentErrMsg_ = DlsymAscendFuncObj(aclGetRecentErrMsg, baseHandler);
  RT_VLOG(VL_HARDWARE) << "Load acl base api success!";
}

void LoadSimulationAclBaseApi() {
  ASSIGN_SIMU(aclCreateDataBuffer);
  ASSIGN_SIMU(aclCreateTensorDesc);
  ASSIGN_SIMU(aclDataTypeSize);
  ASSIGN_SIMU(aclDestroyDataBuffer);
  ASSIGN_SIMU(aclDestroyTensorDesc);
  ASSIGN_SIMU(aclGetTensorDescDimV2);
  ASSIGN_SIMU(aclGetTensorDescNumDims);
  ASSIGN_SIMU(aclSetTensorConst);
  ASSIGN_SIMU(aclSetTensorDescName);
  ASSIGN_SIMU(aclSetTensorFormat);
  ASSIGN_SIMU(aclSetTensorPlaceMent);
  ASSIGN_SIMU(aclSetTensorShape);
  ASSIGN_SIMU(aclUpdateDataBuffer);
  ASSIGN_SIMU(aclrtGetSocName);
  ASSIGN_SIMU(aclGetDataBufferAddr);
  ASSIGN_SIMU(aclGetTensorDescSize);
  ASSIGN_SIMU(aclGetRecentErrMsg);
}
} // namespace fxrt::device::ascend
