#ifndef BUILD_LIBTORCH
#include <Python.h>
#include <functional>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include "torch_npu/csrc/inductor/mlir/hacl_rt.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"

C10_NPU_API rtError_t
common_launch(char* kernelName, const void* func, uint32_t gridX, void* args, uint32_t argsSize, rtStream_t stream);
C10_NPU_API rtError_t common_launch_dyn(
    char* kernelName,
    void* func,
    void* tiling_func,
    int64_t tilingSize,
    void* arg_tiling_host,
    void* arg_tiling_device,
    uint32_t gridX,
    void* args,
    uint32_t argsSize,
    rtStream_t stream);
C10_NPU_API void opcommand_call(const char* name, std::function<int()> launch_call);
#endif // BUILD_LIBTORCH
