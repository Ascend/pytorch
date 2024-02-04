#include "torch_npu/csrc/core/npu/NPUHooksInterface.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/utils/LazyInit.h"
#endif

namespace c10_npu {

TORCH_DECLARE_REGISTRY(PrivateUse1HooksRegistry, NPUHooksInterface, NPUHooksArgs);
#define REGISTER_PRIVATEUSE1_HOOKS(clsname) \
  C10_REGISTER_CLASS(PrivateUse1HooksRegistry, clsname, clsname)

C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, NPUHooksInterface, NPUHooksArgs)

void NPUHooksInterface::initPrivateUse1() const
{
#ifndef BUILD_LIBTORCH
    torch_npu::utils::npu_lazy_init();
#endif
}

at::PrivateUse1HooksInterface* get_npu_hooks() {
    static at::PrivateUse1HooksInterface* npu_hooks;
    static c10::once_flag once;
    c10::call_once(once, [] {
        npu_hooks = new NPUHooksInterface();
    });
    return npu_hooks;
}
}
