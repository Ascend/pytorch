#include "torch_npu/csrc/framework/OpCmdHelper.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/OpParamMaker.h"
#include "torch_npu/csrc/framework/InferFormat.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"

namespace at_npu {
namespace native {

std::tuple<aclTensorDesc *, aclDataBuffer *> OpCmdHelper::CovertTensorToAclInput(const at::Tensor &tensor,
                                                                                 const string &descName,
                                                                                 const string &forceDataType)
{
    at::ScalarType scalarDataType = tensor.scalar_type();
    aclDataType aclDataType = CalcuOpUtil::ConvertToAclDataType(scalarDataType, forceDataType);
    const auto &npuDesc = torch_npu::NPUBridge::GetNpuStorageImplDesc(tensor);
    c10::SmallVector<int64_t, 5> storageDims;
    // if aclDataType is ACL_STRING, storageDims is empty.
    if (aclDataType != ACL_STRING) {
        storageDims = npuDesc.storage_sizes_;
    }
    AclTensorDescMaker desc;
    auto aclDesc =
        desc.Create(aclDataType, npuDesc).SetFormat(npuDesc.npu_format_).SetShape(storageDims).SetName(descName).Get();

    // if aclDataType != ACL_STRING, we use storageDims to calculate nums and use nums * tensor element size to
    // calculate buffer size. But if aclDataType = ACL_STRING, STRING tensor size = 1 and storageDims = 0, we can not
    // use it to calculate size, we need from storage_sizes_ to calculate STRING element real size.
    int64_t numel = c10::multiply_integers(npuDesc.storage_sizes_);
    AclTensorBufferMaker buffer(tensor, numel);
    auto aclBuff = buffer.Get();
    return std::tie(aclDesc, aclBuff);
}

std::tuple<aclTensorDesc *, aclDataBuffer *> OpCmdHelper::CovertTensorWithZeroDimToAclInput(const at::Tensor &tensor,
                                                                                            at::ScalarType type)
{
    // 针对在host侧的tensor，需要做大量处理
    at::ScalarType scalarDataType = type;
    if (!tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
        scalarDataType = tensor.scalar_type();
    }
    aclDataType aclDataType = CalcuOpUtil::ConvertToAclDataType(scalarDataType);
    c10::Scalar expScalar = CalcuOpUtil::ConvertTensorToScalar(tensor);
    at::Tensor aclInput = CalcuOpUtil::CopyScalarToDevice(expScalar, scalarDataType);

    AclTensorDescMaker desc;
    auto aclDesc = desc.Create(aclDataType, ACL_FORMAT_ND).Get();
    AclTensorBufferMaker buffer(aclInput);
    auto aclBuff = buffer.Get();
    return std::tie(aclDesc, aclBuff);
}

std::tuple<aclTensorDesc *, aclDataBuffer *> OpCmdHelper::CovertNPUTensorWithZeroDimToAclInput(const at::Tensor &tensor,
                                                                                               const string &descName)
{
    aclDataType aclDataType = CalcuOpUtil::ConvertToAclDataType(tensor.scalar_type());
    AclTensorDescMaker desc;
    auto aclDesc = desc.Create(aclDataType, ACL_FORMAT_ND).SetName(descName).Get();
    AclTensorBufferMaker buffer(tensor);
    auto aclBuff = buffer.Get();
    return std::tie(aclDesc, aclBuff);
}

std::tuple<aclTensorDesc *, aclDataBuffer *> OpCmdHelper::CovertScalarToAclInput(const at::Tensor &aclInput,
                                                                                 at::ScalarType type)
{
    aclDataType aclDataType = CalcuOpUtil::ConvertToAclDataType(type);

    AclTensorDescMaker desc;
    auto aclDesc = desc.Create(aclDataType, ACL_FORMAT_ND).Get();
    AclTensorBufferMaker aclBuffer(aclInput);
    auto aclBuff = aclBuffer.Get();
    return std::tie(aclDesc, aclBuff);
}

std::tuple<aclTensorDesc *, aclDataBuffer *> OpCmdHelper::CovertHostTensorToAclInput(const at::Tensor &tensor,
                                                                                     at::ScalarType type,
                                                                                     CompileType compileType,
                                                                                     const string &forceDataType,
                                                                                     const string &descName)
{
    aclDataType aclDataType = CalcuOpUtil::ConvertToAclDataType(type, forceDataType);
    const auto &dims = tensor.sizes();
    AclTensorDescMaker desc;
    aclFormat format = ACL_FORMAT_ND;
    auto aclDesc = desc.Create(aclDataType, dims, format)
                       .SetPlacement(static_cast<aclMemType>(compileType))
                       .SetName(descName)
                       .Get();
    int64_t numel = c10::multiply_integers(dims);
    AclTensorBufferMaker buffer(tensor, numel);
    auto aclBuff = buffer.Get();
    return std::tie(aclDesc, aclBuff);
}

std::tuple<aclTensorDesc *, aclDataBuffer *> OpCmdHelper::CovertToAclOutput(const at::Tensor &tensor,
                                                                            const string &forceDataType)
{
    aclDataType aclDataType = CalcuOpUtil::ConvertToAclDataType(tensor.scalar_type(), forceDataType);
    const auto &npuDesc = torch_npu::NPUBridge::GetNpuStorageImplDesc(tensor);
    const auto &dims = tensor.sizes();
    auto &storageDims = npuDesc.storage_sizes_;
    AclTensorDescMaker desc;
    auto aclDesc = desc.Create(aclDataType, dims, npuDesc.origin_format_)
                       .SetFormat(npuDesc.npu_format_)
                       .SetShape(storageDims)
                       .Get();
    auto numel = c10::multiply_integers(storageDims);
    AclTensorBufferMaker aclBuffer(tensor, numel);
    auto aclBuff = aclBuffer.Get();
    return std::tie(aclDesc, aclBuff);
}
} // namespace native
} // namespace at_npu
