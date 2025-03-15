#ifndef __PULGIN_NATIVE_UTILS_OP_COMMAND_HELPER__
#define __PULGIN_NATIVE_UTILS_OP_COMMAND_HELPER__

#include <ATen/ATen.h>

#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/acl/acl_base.h"

namespace at_npu {
namespace native {

// covert pytorch tensor to acl tensor.
class OpCmdHelper {
public:
    static std::tuple<aclTensorDesc *, aclDataBuffer *> CovertTensorToAclInput(const at::Tensor &tensor,
                                                                               const string &descName,
                                                                               const string &forceDataType = "");

    static std::tuple<aclTensorDesc *, aclDataBuffer *> CovertTensorWithZeroDimToAclInput(const at::Tensor &tensor,
                                                                                          at::ScalarType type);

    static std::tuple<aclTensorDesc *, aclDataBuffer *> CovertNPUTensorWithZeroDimToAclInput(const at::Tensor &tensor,
                                                                                             const string &descName);

    static std::tuple<aclTensorDesc *, aclDataBuffer *> CovertScalarToAclInput(const at::Tensor &aclInput,
                                                                               at::ScalarType type);

    static std::tuple<aclTensorDesc *, aclDataBuffer *> CovertToAclOutput(const at::Tensor &tensor,
                                                                          const string &forceDataType);

    static std::tuple<aclTensorDesc *, aclDataBuffer *> CovertHostTensorToAclInput(const at::Tensor &tensor,
                                                                                   at::ScalarType type,
                                                                                   CompileType compileType,
                                                                                   const string& forceDataType,
                                                                                   const string &descName);
}; // class OpCommandImpl

} // namespace native
} // namespace at_npu

#endif