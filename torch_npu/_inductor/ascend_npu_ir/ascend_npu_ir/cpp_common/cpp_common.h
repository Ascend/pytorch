#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <functional>
#include "hacl_rt.h"

rtError_t common_launch(char* kernelName, const void* func, uint32_t gridX,
                        void* args, uint32_t argsSize, rtStream_t stream);
rtError_t common_launch_dyn(char* kernelName, void* func, void* tiling_func, int64_t tilingSize, void* arg_tiling_device, uint32_t gridX,
                        void* args, uint32_t argsSize, rtStream_t stream);
void opcommand_call(const char *name, std::function<int()> launch_call);