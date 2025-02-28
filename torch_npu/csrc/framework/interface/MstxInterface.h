#ifndef __TORCH_NPU_MSTXINTERFACE__
#define __TORCH_NPU_MSTXINTERFACE__

#include <third_party/mstx/ms_tools_ext.h>

namespace at_npu {
namespace native {

bool IsSupportMstxFunc();

bool IsSupportMstxDomainFunc();

void MstxMarkA(const char* message, aclrtStream stream);

int MstxRangeStartA(const char* message, aclrtStream stream, int ptRangeId);

void MstxRangeEnd(int ptRangeId);

mstxDomainHandle_t MstxDomainCreateA(const char* name);

void MstxDomainDestroy(mstxDomainHandle_t handle);

void MstxDomainMarkA(mstxDomainHandle_t handle, const char* message, aclrtStream stream);

int MstxDomainRangeStartA(mstxDomainHandle_t handle, const char* message, aclrtStream stream, int ptRangeId);

void MstxDomainRangeEnd(mstxDomainHandle_t handle, int ptRangeId);

mstxMemHeapHandle_t MstxMemHeapRegister(mstxDomainHandle_t domain, const mstxMemHeapDesc_t* desc);

void MstxMemHeapUnregister(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap);

void MstxMemRegionsRegister(mstxDomainHandle_t domain, const mstxMemRegionsRegisterBatch_t* desc);

void MstxMemRegionsUnregister(mstxDomainHandle_t domain, const mstxMemRegionsUnregisterBatch_t* desc);

}
}

#endif