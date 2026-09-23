#ifndef FXRT_SRC_HARDWARE_ASCEND_ACL_MDL_SYMBOL_H_
#define FXRT_SRC_HARDWARE_ASCEND_ACL_MDL_SYMBOL_H_
#include <string>
#include "acl/acl_mdl.h"
#include "hardware/hardware_abstract/dlopen_macro.h"

namespace fxrt::device::ascend {
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlAddDatasetBuffer, aclError, aclmdlDataset*, aclDataBuffer*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlCreateDataset, aclmdlDataset*);
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlCreateDesc, aclmdlDesc*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlDestroyDataset, aclError, const aclmdlDataset*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlDestroyDesc, aclError, aclmdlDesc*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlExecute, aclError, uint32_t, const aclmdlDataset*, aclmdlDataset*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlFinalizeDump, aclError)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetCurOutputDims, aclError, const aclmdlDesc*, size_t, aclmdlIODims*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetDatasetBuffer, aclDataBuffer*, const aclmdlDataset*, size_t)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetDatasetNumBuffers, size_t, const aclmdlDataset*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetDesc, aclError, aclmdlDesc*, uint32_t)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetInputDataType, aclDataType, const aclmdlDesc*, size_t)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetInputDims, aclError, const aclmdlDesc*, size_t, aclmdlIODims*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetInputIndexByName, aclError, const aclmdlDesc*, const char*, size_t*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetInputNameByIndex, const char*, const aclmdlDesc*, size_t)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetInputSizeByIndex, size_t, aclmdlDesc*, size_t index)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetNumInputs, size_t, aclmdlDesc*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetNumOutputs, size_t, aclmdlDesc*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetOutputDataType, aclDataType, const aclmdlDesc*, size_t)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetOutputDims, aclError, const aclmdlDesc*, size_t, aclmdlIODims*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetOutputNameByIndex, const char*, const aclmdlDesc*, size_t)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetOutputSizeByIndex, size_t, aclmdlDesc*, size_t)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlInitDump, aclError)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlLoadFromMem, aclError, const void*, size_t, uint32_t*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlSetDump, aclError, const char*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlSetDynamicBatchSize, aclError, uint32_t, aclmdlDataset*, size_t, uint64_t)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlUnload, aclError, uint32_t)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlQuerySizeFromMem, aclError, const void*, size_t, size_t*, size_t*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlBundleGetModelId, aclError, uint32_t, size_t, uint32_t*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlBundleLoadFromMem, aclError, const void*, size_t, uint32_t*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlBundleUnload, aclError, uint32_t)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(
    aclmdlLoadFromMemWithMem,
    aclError,
    const void*,
    size_t,
    uint32_t*,
    void*,
    size_t,
    void*,
    size_t)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlSetDatasetTensorDesc, aclError, aclmdlDataset*, aclTensorDesc*, size_t)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetInputFormat, aclFormat, const aclmdlDesc*, size_t)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetDatasetTensorDesc, aclTensorDesc*, const aclmdlDataset*, size_t)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlSetInputDynamicDims, aclError, uint32_t, aclmdlDataset*, size_t, const aclmdlIODims*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetOutputFormat, aclFormat, const aclmdlDesc*, size_t)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetInputDimsV2, aclError, const aclmdlDesc*, size_t, aclmdlIODims*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetDynamicHW, aclError, const aclmdlDesc*, size_t, aclmdlHW*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetInputDynamicDims, aclError, const aclmdlDesc*, size_t, aclmdlIODims*, size_t)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetInputDynamicGearCount, aclError, const aclmdlDesc*, size_t, size_t*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlGetDynamicBatch, aclError, const aclmdlDesc*, aclmdlBatch*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlSetDynamicHWSize, aclError, uint32_t, aclmdlDataset*, size_t, uint64_t, uint64_t)
#if defined(__linux__)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlRICaptureBegin, aclError, aclrtStream, aclmdlRICaptureMode)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlRICaptureGetInfo, aclError, aclrtStream, aclmdlRICaptureStatus*, aclmdlRI*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlRICaptureEnd, aclError, aclrtStream, aclmdlRI*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlRIExecuteAsync, aclError, aclmdlRI, aclrtStream)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlRIDestroy, aclError, aclmdlRI)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlRICaptureTaskGrpBegin, aclError, aclrtStream)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlRICaptureTaskGrpEnd, aclError, aclrtStream, aclrtTaskGrp*)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlRICaptureTaskUpdateBegin, aclError, aclrtStream, aclrtTaskGrp)
// cppcheck-suppress unknownMacro
ORIGIN_METHOD_WITH_SIMU(aclmdlRICaptureTaskUpdateEnd, aclError, aclrtStream)
#endif

extern aclmdlAddDatasetBufferFunObj aclmdlAddDatasetBuffer_;
extern aclmdlCreateDatasetFunObj aclmdlCreateDataset_;
extern aclmdlCreateDescFunObj aclmdlCreateDesc_;
extern aclmdlDestroyDatasetFunObj aclmdlDestroyDataset_;
extern aclmdlDestroyDescFunObj aclmdlDestroyDesc_;
extern aclmdlExecuteFunObj aclmdlExecute_;
extern aclmdlFinalizeDumpFunObj aclmdlFinalizeDump_;
extern aclmdlGetCurOutputDimsFunObj aclmdlGetCurOutputDims_;
extern aclmdlGetDatasetBufferFunObj aclmdlGetDatasetBuffer_;
extern aclmdlGetDatasetNumBuffersFunObj aclmdlGetDatasetNumBuffers_;
extern aclmdlGetDescFunObj aclmdlGetDesc_;
extern aclmdlGetInputDataTypeFunObj aclmdlGetInputDataType_;
extern aclmdlGetInputDimsFunObj aclmdlGetInputDims_;
extern aclmdlGetInputIndexByNameFunObj aclmdlGetInputIndexByName_;
extern aclmdlGetInputNameByIndexFunObj aclmdlGetInputNameByIndex_;
extern aclmdlGetInputSizeByIndexFunObj aclmdlGetInputSizeByIndex_;
extern aclmdlGetNumInputsFunObj aclmdlGetNumInputs_;
extern aclmdlGetNumOutputsFunObj aclmdlGetNumOutputs_;
extern aclmdlGetOutputDataTypeFunObj aclmdlGetOutputDataType_;
extern aclmdlGetOutputDimsFunObj aclmdlGetOutputDims_;
extern aclmdlGetOutputNameByIndexFunObj aclmdlGetOutputNameByIndex_;
extern aclmdlGetOutputSizeByIndexFunObj aclmdlGetOutputSizeByIndex_;
extern aclmdlInitDumpFunObj aclmdlInitDump_;
extern aclmdlLoadFromMemFunObj aclmdlLoadFromMem_;
extern aclmdlSetDumpFunObj aclmdlSetDump_;
extern aclmdlSetDynamicBatchSizeFunObj aclmdlSetDynamicBatchSize_;
extern aclmdlUnloadFunObj aclmdlUnload_;
extern aclmdlQuerySizeFromMemFunObj aclmdlQuerySizeFromMem_;
extern aclmdlBundleGetModelIdFunObj aclmdlBundleGetModelId_;
extern aclmdlBundleLoadFromMemFunObj aclmdlBundleLoadFromMem_;
extern aclmdlBundleUnloadFunObj aclmdlBundleUnload_;
extern aclmdlLoadFromMemWithMemFunObj aclmdlLoadFromMemWithMem_;
extern aclmdlSetDatasetTensorDescFunObj aclmdlSetDatasetTensorDesc_;
extern aclmdlGetInputFormatFunObj aclmdlGetInputFormat_;
extern aclmdlGetDatasetTensorDescFunObj aclmdlGetDatasetTensorDesc_;
extern aclmdlSetInputDynamicDimsFunObj aclmdlSetInputDynamicDims_;
extern aclmdlGetOutputFormatFunObj aclmdlGetOutputFormat_;
extern aclmdlGetInputDimsV2FunObj aclmdlGetInputDimsV2_;
extern aclmdlGetDynamicHWFunObj aclmdlGetDynamicHW_;
extern aclmdlGetInputDynamicDimsFunObj aclmdlGetInputDynamicDims_;
extern aclmdlGetInputDynamicGearCountFunObj aclmdlGetInputDynamicGearCount_;
extern aclmdlGetDynamicBatchFunObj aclmdlGetDynamicBatch_;
extern aclmdlSetDynamicHWSizeFunObj aclmdlSetDynamicHWSize_;
#if defined(__linux__)
extern aclmdlRICaptureBeginFunObj aclmdlRICaptureBegin_;
extern aclmdlRICaptureGetInfoFunObj aclmdlRICaptureGetInfo_;
extern aclmdlRICaptureEndFunObj aclmdlRICaptureEnd_;
extern aclmdlRIExecuteAsyncFunObj aclmdlRIExecuteAsync_;
extern aclmdlRIDestroyFunObj aclmdlRIDestroy_;
extern aclmdlRICaptureTaskGrpBeginFunObj aclmdlRICaptureTaskGrpBegin_;
extern aclmdlRICaptureTaskGrpEndFunObj aclmdlRICaptureTaskGrpEnd_;
extern aclmdlRICaptureTaskUpdateBeginFunObj aclmdlRICaptureTaskUpdateBegin_;
extern aclmdlRICaptureTaskUpdateEndFunObj aclmdlRICaptureTaskUpdateEnd_;

#endif

void LoadAclMdlApiSymbol(const std::string& ascendPath);
void LoadSimulationAclMdlApi();
} // namespace fxrt::device::ascend

#endif // FXRT_SRC_HARDWARE_ASCEND_ACL_MDL_SYMBOL_H_
