#include "torch_npu/csrc/custom_dtype/extension.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"


namespace c10_npu {

at::Tensor cast_to_fp8(const at::Tensor &input, int otype)
{
    auto output = at::empty_like(input, c10_npu::GetATenDType(otype));

    if (input.numel() == 0) {
        return output;
    }

    aclDataType out_acltype = c10_npu::GetAclDataType(otype);
    TensorWrapper out_wrapper = {output, out_acltype};
    EXEC_NPU_CMD(aclnnCast, input, out_acltype, out_wrapper);

    return output;
}

void cast_to_fp8_noalloc(const at::Tensor &input, at::Tensor output, int otype)
{
    aclDataType out_acltype = c10_npu::GetAclDataType(otype);
    TensorWrapper out_wrapper = {output, out_acltype};
    EXEC_NPU_CMD(aclnnCast, input, out_acltype, out_wrapper);
    return;
}

at::Tensor cast_from_fp8(const at::Tensor &input, int itype, int otype)
{
    aclDataType input_acltype = c10_npu::GetAclDataType(itype);
    aclDataType out_acltype = c10_npu::GetAclDataType(otype);
    auto output = at::empty_like(input, c10_npu::GetATenDType(otype));
    TensorWrapper input_wrapper = {input, input_acltype};
    TensorWrapper out_wrapper = {output, out_acltype};
    EXEC_NPU_CMD(aclnnCast, input_wrapper, out_acltype, out_wrapper);

    return output;
}
}
