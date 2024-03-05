#include "torch_npu/csrc/core/npu/interface/HcclInterface.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"

namespace at_npu {
namespace hccl {
#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libhccl, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName) \
  GET_FUNCTION(libhccl, funcName)

REGISTER_LIBRARY(libhccl)
LOAD_FUNCTION(HcclGetCommName)

extern HcclResult HcclGetCommNameFace(HcclComm commHandle, char* commName) {
    typedef HcclResult (*HcclGetCommNameFace)(HcclComm commHandle, char* commName);
    static HcclGetCommNameFace func = nullptr;
    if (func == nullptr)
    {
        func = (HcclGetCommNameFace)GET_FUNC(HcclGetCommName);
    }
    TORCH_CHECK(func, "Failed to find function HcclGetCommName,"
                " maybe you cann version is too low, please upgrade it",
                PTA_ERROR(ErrCode::NOT_FOUND));
    return func(commHandle, commName);
}
} // namespace native
} // namespace at_npu