#include "acl_tdt_symbol.h"
#include <string>
#include <vector>
#include "symbol_utils.h"

namespace fxrt::device::ascend {

acltdtAddDataItemFunObj acltdtAddDataItem_ = nullptr;
acltdtCleanChannelFunObj acltdtCleanChannel_ = nullptr;
acltdtCreateChannelFunObj acltdtCreateChannel_ = nullptr;
acltdtCreateChannelWithCapacityFunObj acltdtCreateChannelWithCapacity_ = nullptr;
acltdtCreateDataItemFunObj acltdtCreateDataItem_ = nullptr;
acltdtCreateDatasetFunObj acltdtCreateDataset_ = nullptr;
acltdtDestroyChannelFunObj acltdtDestroyChannel_ = nullptr;
acltdtDestroyDataItemFunObj acltdtDestroyDataItem_ = nullptr;
acltdtDestroyDatasetFunObj acltdtDestroyDataset_ = nullptr;
acltdtGetDataAddrFromItemFunObj acltdtGetDataAddrFromItem_ = nullptr;
acltdtGetDataItemFunObj acltdtGetDataItem_ = nullptr;
acltdtGetDatasetNameFunObj acltdtGetDatasetName_ = nullptr;
acltdtGetDatasetSizeFunObj acltdtGetDatasetSize_ = nullptr;
acltdtGetDataSizeFromItemFunObj acltdtGetDataSizeFromItem_ = nullptr;
acltdtGetDataTypeFromItemFunObj acltdtGetDataTypeFromItem_ = nullptr;
acltdtGetDimNumFromItemFunObj acltdtGetDimNumFromItem_ = nullptr;
acltdtGetDimsFromItemFunObj acltdtGetDimsFromItem_ = nullptr;
acltdtGetTensorTypeFromItemFunObj acltdtGetTensorTypeFromItem_ = nullptr;
acltdtGetSliceInfoFromItemFunObj acltdtGetSliceInfoFromItem_ = nullptr;
acltdtQueryChannelSizeFunObj acltdtQueryChannelSize_ = nullptr;
acltdtReceiveTensorFunObj acltdtReceiveTensor_ = nullptr;
acltdtSendTensorFunObj acltdtSendTensor_ = nullptr;
acltdtStopChannelFunObj acltdtStopChannel_ = nullptr;

void LoadAcltdtApiSymbol(const std::string& ascendPath) {
  const std::vector<std::string> dependLibs = {"libacl_tdt_queue.so"};
  for (const auto& depLib : dependLibs) {
    (void)GetLibHandler(ascendPath + "lib64/" + depLib);
  }

  std::string aclrtTdtPath = ascendPath + "lib64/libacl_tdt_channel.so";
  auto handler = GetLibHandler(aclrtTdtPath);
  if (handler == nullptr) {
    RT_VLOG(VL_HARDWARE) << "Dlopen " << aclrtTdtPath << " failed!" << dlerror();
    return;
  }
  acltdtAddDataItem_ = DlsymAscendFuncObj(acltdtAddDataItem, handler);
  acltdtCleanChannel_ = DlsymAscendFuncObj(acltdtCleanChannel, handler);
  acltdtCreateChannel_ = DlsymAscendFuncObj(acltdtCreateChannel, handler);
  acltdtCreateChannelWithCapacity_ = DlsymAscendFuncObj(acltdtCreateChannelWithCapacity, handler);
  acltdtCreateDataItem_ = DlsymAscendFuncObj(acltdtCreateDataItem, handler);
  acltdtCreateDataset_ = DlsymAscendFuncObj(acltdtCreateDataset, handler);
  acltdtDestroyChannel_ = DlsymAscendFuncObj(acltdtDestroyChannel, handler);
  acltdtDestroyDataItem_ = DlsymAscendFuncObj(acltdtDestroyDataItem, handler);
  acltdtDestroyDataset_ = DlsymAscendFuncObj(acltdtDestroyDataset, handler);
  acltdtGetDataAddrFromItem_ = DlsymAscendFuncObj(acltdtGetDataAddrFromItem, handler);
  acltdtGetDataItem_ = DlsymAscendFuncObj(acltdtGetDataItem, handler);
  acltdtGetDatasetName_ = DlsymAscendFuncObj(acltdtGetDatasetName, handler);
  acltdtGetDatasetSize_ = DlsymAscendFuncObj(acltdtGetDatasetSize, handler);
  acltdtGetDataSizeFromItem_ = DlsymAscendFuncObj(acltdtGetDataSizeFromItem, handler);
  acltdtGetDataTypeFromItem_ = DlsymAscendFuncObj(acltdtGetDataTypeFromItem, handler);
  acltdtGetDimNumFromItem_ = DlsymAscendFuncObj(acltdtGetDimNumFromItem, handler);
  acltdtGetDimsFromItem_ = DlsymAscendFuncObj(acltdtGetDimsFromItem, handler);
  acltdtGetTensorTypeFromItem_ = DlsymAscendFuncObj(acltdtGetTensorTypeFromItem, handler);
  acltdtGetSliceInfoFromItem_ = DlsymAscendFuncObj(acltdtGetSliceInfoFromItem, handler);
  acltdtQueryChannelSize_ = DlsymAscendFuncObj(acltdtQueryChannelSize, handler);
  acltdtReceiveTensor_ = DlsymAscendFuncObj(acltdtReceiveTensor, handler);
  acltdtSendTensor_ = DlsymAscendFuncObj(acltdtSendTensor, handler);
  acltdtStopChannel_ = DlsymAscendFuncObj(acltdtStopChannel, handler);
  RT_VLOG(VL_HARDWARE) << "Load acl tdt api success!";
}

void LoadSpecialSimulationTdtApi() {
  acltdtQueryChannelSize_ = [](const acltdtChannelHandle* handle, size_t* retSizePtr) {
    if (handle == nullptr) {
      RT_VLOG(VL_HARDWARE) << "Empty handle!";
    }
    if (retSizePtr != nullptr) {
      *retSizePtr = 1;
    }
    return ACL_SUCCESS;
  };
}

void LoadSimulationTdtApi() {
  ASSIGN_SIMU(acltdtAddDataItem);
  ASSIGN_SIMU(acltdtCleanChannel);
  ASSIGN_SIMU(acltdtCreateChannel);
  ASSIGN_SIMU(acltdtCreateChannelWithCapacity);
  ASSIGN_SIMU(acltdtCreateDataItem);
  ASSIGN_SIMU(acltdtCreateDataset);
  ASSIGN_SIMU(acltdtDestroyChannel);
  ASSIGN_SIMU(acltdtDestroyDataItem);
  ASSIGN_SIMU(acltdtDestroyDataset);
  ASSIGN_SIMU(acltdtGetDataAddrFromItem);
  ASSIGN_SIMU(acltdtGetDataItem);
  ASSIGN_SIMU(acltdtGetDatasetName);
  ASSIGN_SIMU(acltdtGetDatasetSize);
  ASSIGN_SIMU(acltdtGetDataSizeFromItem);
  ASSIGN_SIMU(acltdtGetDataTypeFromItem);
  ASSIGN_SIMU(acltdtGetDimNumFromItem);
  ASSIGN_SIMU(acltdtGetDimsFromItem);
  ASSIGN_SIMU(acltdtGetTensorTypeFromItem);
  ASSIGN_SIMU(acltdtGetSliceInfoFromItem);
  ASSIGN_SIMU(acltdtQueryChannelSize);
  ASSIGN_SIMU(acltdtReceiveTensor);
  ASSIGN_SIMU(acltdtSendTensor);
  ASSIGN_SIMU(acltdtStopChannel);
  LoadSpecialSimulationTdtApi();
}
} // namespace fxrt::device::ascend
