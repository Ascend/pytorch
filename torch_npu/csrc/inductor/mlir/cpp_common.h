#ifndef BUILD_LIBTORCH
#include <Python.h>
#include <functional>
#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/inductor/mlir/hacl_rt.h"

rtError_t common_launch(char* kernelName, const void* func, uint32_t gridX,
                        void* args, uint32_t argsSize, rtStream_t stream);
rtError_t common_launch_dyn(char* kernelName, void* func, void* tiling_func, int64_t tilingSize, uint32_t gridX,
                        void* args, uint32_t argsSize, rtStream_t stream);
void opcommand_call(const char *name, std::function<int()> launch_call);
#endif