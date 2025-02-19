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

mstxDomainhandle_t MstxDomainCreateA(const char* name);

void MstxDomainDestroy(mstxDomainhandle_t handle);

void MstxDomainMarkA(mstxDomainhandle_t handle, const char* message, aclrtStream stream);

int MstxDomainRangeStartA(mstxDomainhandle_t handle, const char* message, aclrtStream stream, int ptRangeId);

void MstxDomainRangeEnd(mstxDomainhandle_t handle, int ptRangeId);
}
}

#endif