#include <mutex>
#include <set>
#include <sys/stat.h>

#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/framework/contiguous/ContiguousOpt.h"
#include "torch_npu/csrc/framework/interface/MsProfilerInterface.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/profiler/npu_profiler.h"
#endif

namespace at_npu {
namespace native {

void NpuUtils::format_fresh_view(at::Tensor &x, const at::Tensor &y)
{
    // x:NPU before inplace_op, y: NPU computed
    // now we fresh x according to y
    RECORD_FUNCTION("format_fresh_view", vector<c10::IValue>({x, y}));
    x.copy_(y);
}

// NOTE [Check Match for Npu at::Tensor]
// check_match is used to ensure that npu tensor satisfies the
// calculation requirements of npu operators.
// The rules are as follows,
// 1、tensor should be contiguous
// Not contiguous means the operator needs to read and write memory
// at intervals according to strides and sizes. Npu operators has
// no such ability for the time being
// 2、metadata should be match
// Resize_ a contiguous cpu tensor from [1,2,3,4] to [4,3,2,1] no
// need to change the physical memory. However, for a contiguous npu
// tensor whose npu_format_ is 5HD, storage shape should be change
// from [1,1,3,4,16] to [4,1,2,1,16]. So metadata not match often
// results in unexpected physical memory. format_contiguous will be
// called preparing correct memory of operand in these case.
bool NpuUtils::check_match(const at::Tensor *tensor)
{
    // case1:uncontiguous tensor
    if (!tensor->is_contiguous()) {
        return false;
    }

    // case2:meta data not match, sizes or strides of presentation
    // layer is different from that of storage layer
    if (!StorageDescHelper::MetaDataAreMatch(tensor)) {
        return false;
    }

    // case3:meta data not match, storage_offset of presentation layer
    // is different from that of storage layer
    bool isPadding = FormatHelper::IsPadded(tensor);
    if (isPadding && (!StorageDescHelper::OffsetAreMatch(tensor))) {
        return false;
    }
    return true;
}

bool NpuUtils::check_5d_5d_match(const at::Tensor &tensor)
{
    // (1) NC1HWC0 format in storage, NCHW format in des.
    // (2) 4d format situation, only uncontiguous in Channel size
    // (3) size and start point must be 16*, make sure the memory be contiguous
    if (tensor.is_contiguous()) {
        return false;
    }

    if (torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_.npu_format_ != ACL_FORMAT_NC1HWC0) {
        return false;
    }

    if (tensor.sizes().size() != 4) {
        return false;
    }

    bool is_c_channel_slice = true;
    int64_t z = 1;
    for (int64_t d = tensor.dim() - 1; d >= 1; d--) {
        if (tensor.size(d) != 1) {
            if (tensor.stride(d) == z) {
                z *= tensor.size(d);
            } else {
                is_c_channel_slice = false;
                break;
            }
        }
    }
    if (!is_c_channel_slice) {
        return false;
    }

    int64_t contiguous_len = 16;
    int64_t c0_len = 16;
    for (const auto i : c10::irange(2, torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_.base_sizes_.size())) {
        contiguous_len *= torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_.base_sizes_[i];
    }
    bool is_offset_match = (tensor.storage_offset() % contiguous_len == 0);
    bool is_length_match = (tensor.size(1) % c0_len == 0);

    return is_offset_match && is_length_match;
}

void NpuUtils::RefreshFormat(const at::Tensor &tensor)
{
    auto &tensor_desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
    if (tensor_desc.storage_sizes_.size() == 4 && tensor_desc.npu_format_ == ACL_FORMAT_ND) {
        tensor_desc.npu_format_ = ACL_FORMAT_NCHW;
        tensor_desc.origin_format_ = ACL_FORMAT_NCHW;
    } else if (tensor_desc.storage_sizes_.size() != 4 && tensor_desc.npu_format_ == ACL_FORMAT_NCHW) {
        tensor_desc.npu_format_ = ACL_FORMAT_ND;
        tensor_desc.origin_format_ = ACL_FORMAT_ND;
    }
}

at::Tensor metadata_convert_match(const at::Tensor &src, bool numelEq)
{
    // Only when a tensor monopolizes a storage can NpuStorageDesc be
    // refreshed. When the original format is not NCHW, the npu_format_cast to
    // NCHW will generate a temporary tensor, which always monopolizes its own
    // storage.
    if (numelEq && (!FormatHelper::IsBaseFormatType(src))) {
        at::Tensor tempTensor = custom_ops::npu_format_cast(src, FormatHelper::GetBaseFormat(src));
        custom_ops::npu_reshape_out(tempTensor, tempTensor.sizes(), true, tempTensor);
        NpuUtils::RefreshFormat(tempTensor);
        return tempTensor;
    } else {
        at::Tensor contiguous_view = at::empty(src.sizes(), src.options());
        contiguous_view.copy_(src);
        NpuUtils::RefreshFormat(contiguous_view);
        return contiguous_view;
    }
}

at::Tensor metadata_convert_match_without_copy_optimize(const at::Tensor &src)
{
    TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
                "Expected all tensors to be on the same device. "
                "Expected NPU tensor, please check whether the input tensor device is correct.",
                OPS_ERROR(ErrCode::TYPE));
    auto &src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
    bool numelEq = (src.numel() == c10::multiply_integers(src_desc.base_sizes_));
    return metadata_convert_match(src, numelEq);
}

at::Tensor metadata_convert_match_with_copy_optimize(const at::Tensor &src)
{
    TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
                "Expected all tensors to be on the same device. "
                "Expected NPU tensor, please check whether the input tensor device is correct.",
                OPS_ERROR(ErrCode::TYPE));
    auto &src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
    bool numelEq = (src.numel() == c10::multiply_integers(src_desc.base_sizes_));

    // For unmatched Tensors with base format, we can:
    OptimizationCases optimizations_reshape{"reshapeV2"};
    if (numelEq && src_desc.npu_format_ == ACL_FORMAT_ND && src_desc.origin_format_ == ACL_FORMAT_ND &&
        (src.dim() != 0) && !src_desc.base_sizes_.empty()) {
        // 1. directly rewrite their storage description to get matched tensors.
        src_desc.base_sizes_ = CalcuOpUtil::ConvertIntArrayRefToSmallVector(src.sizes());
        src_desc.base_strides_ = CalcuOpUtil::ConvertIntArrayRefToSmallVector(src.strides());
        src_desc.storage_sizes_ = CalcuOpUtil::ConvertIntArrayRefToSmallVector(src.sizes());
        NpuUtils::RefreshFormat(src);
        return src;
    } else if (TransContiguous::CanOptimize(src, optimizations_reshape)) {
        // 2. using memory-repoint/DMA for other cases.
        auto reshapeTensor = TransContiguous::ContiguousOptimizeWithAnyFormat(src, optimizations_reshape);
        if (reshapeTensor.has_value()) {
            return reshapeTensor.value();
        }
    }
    // 3. common method using transdata and copy_, just the same as:
    // metadata_convert_match_without_copy_optimize
    return metadata_convert_match(src, numelEq);
}

at::Tensor metadata_with_offset_padding_convert_match(const at::Tensor &src)
{
    at::Tensor contiguous_view = at::empty(src.sizes(), src.options());
    contiguous_view.copy_(src);
    NpuUtils::RefreshFormat(contiguous_view);
    return contiguous_view;
}

at::Tensor NpuUtils::format_contiguous(const at::Tensor &src)
{
    // case1:tensor src is not contiguous
    if (!src.is_contiguous()) {
        RECORD_FUNCTION("format_contiguous", vector<c10::IValue>({src}));
        return src.contiguous();
    }
    // case2:meta data not match, sizes or strides of presentation
    // layer is different from that of storage layer
    if (!StorageDescHelper::MetaDataAreMatch(&src)) {
        // Fix not match case2, tensor should have matched metadata and
        // NPUStorageDesc.
        RECORD_FUNCTION("format_contiguous", vector<c10::IValue>({src}));
        return metadata_convert_match_without_copy_optimize(src);
    }

    // case3:meta data not match, storage_offset of presentation layer
    // is different from that of storage layer
    if (FormatHelper::IsPadded(&src) && (!StorageDescHelper::OffsetAreMatch(&src))) {
        // Fix not match case3, tensor with padding should not have storage-offset.
        RECORD_FUNCTION("format_contiguous", vector<c10::IValue>({src}));
        return metadata_with_offset_padding_convert_match(src);
    }

    return src;
}

at::Tensor NpuUtils::format_contiguous_add_copy_optimize(const at::Tensor &src)
{
    // case1:tensor src is not contiguous
    if (!src.is_contiguous()) {
        RECORD_FUNCTION("format_contiguousV2", vector<c10::IValue>({src}));
        return src.contiguous();
    }
    // case2:meta data not match, sizes or strides of presentation
    // layer is different from that of storage layer
    if (!StorageDescHelper::MetaDataAreMatch(&src)) {
        // Fix not match case2, tensor should have matched metadata and
        // NPUStorageDesc.
        RECORD_FUNCTION("format_contiguousV2", vector<c10::IValue>({src}));
        return metadata_convert_match_with_copy_optimize(src);
    }

    // case3:meta data not match, storage_offset of presentation layer
    // is different from that of storage layer
    if (FormatHelper::IsPadded(&src) && (!StorageDescHelper::OffsetAreMatch(&src))) {
        // Fix not match case3, tensor with padding should not have storage-offset.
        RECORD_FUNCTION("format_contiguousV2", vector<c10::IValue>({src}));
        return metadata_with_offset_padding_convert_match(src);
    }

    return src;
}

bool NpuUtils::IsOomError(aclError ret, int index)
{
    if (ret == ACL_ERROR_GE_DEVICE_MEMORY_ALLOCATION_FAILED) {
        int deviceId = 0;
        // free devcie cached memory when return value of the first op execution is
        // oom
        if (index == 1) {
            NPU_CHECK_ERROR(c10_npu::GetDevice(&deviceId));
            c10_npu::NPUCachingAllocator::FreeDeviceCachedMemory(deviceId);
            return true;
        }
        AT_ERROR("NPU out of memory. device id: ", deviceId);
    }
    return false;
}

void NpuUtils::check_1d(const at::Tensor &t, const char *arg, const char *fn)
{
    TORCH_CHECK(t.dim() == 1, fn, ": Expected 1-D argument ", arg, ", but got ", t.dim(), "-D",
                OPS_ERROR(ErrCode::PARAM));
}

bool NpuUtils::setFilePermissions(int fd, mode_t mode)
{
    if (fchmod(fd, mode) == -1) {
        ASCEND_LOGI("Failed to set permissions.");
        return false;
    }
    return true;
}

#ifndef BUILD_LIBTORCH

void NpuUtils::ProfReportMarkDataToNpuProfiler(uint32_t category, const std::string &data, uint64_t correlation_id)
{
    if (data.empty()) {
        return;
    }
    if (torch_npu::profiler::profDataReportEnable().load(std::memory_order_relaxed)) {
        torch_npu::profiler::reportMarkDataToNpuProfiler(category, data, correlation_id);
    }
}

void NpuUtils::DqueueCompileExcute(c10_npu::queue::QueueParas *para, uint32_t category)
{
    auto param_val = static_cast<at_npu::native::ExecuteParas *>(para->paramVal);
    torch_npu::profiler::reportMarkDataToNpuProfiler(category, std::string(param_val->opType), para->correlation_id);
}

void NpuUtils::DqueueCompileExcuteOpApi(c10_npu::queue::QueueParas *para, uint32_t category)
{
    auto param_val = static_cast<at_npu::native::ExecuteParasOpApi *>(para->paramVal);
    torch_npu::profiler::reportMarkDataToNpuProfiler(category, std::string(param_val->opType), para->correlation_id);
}

void NpuUtils::DqueueEvent(c10_npu::queue::QueueParas *para, uint32_t category)
{
    torch_npu::profiler::reportMarkDataToNpuProfiler(
        category, c10_npu::queue::EventParas::EVENT_PARAS_MAP[para->paramType], para->correlation_id);
}
void NpuUtils::DqueueAnyncMemcpy(c10_npu::queue::QueueParas *para, uint32_t category)
{
    auto param_val = static_cast<c10_npu::queue::CopyParas *>(para->paramVal);
    torch_npu::profiler::reportMarkDataToNpuProfiler(
        category, c10_npu::queue::CopyParas::COPY_PARAS_MAP[param_val->kind], para->correlation_id);
}

void NpuUtils::ProfReportMarkDataToNpuProfiler(uint32_t category, void *data, size_t offset)
{
    if (C10_UNLIKELY(!data)) {
        return;
    }
    if (torch_npu::profiler::profDataReportEnable().load(std::memory_order_relaxed)) {
        static const std::map<int64_t, DqueueCall> DEQUEUE_CALL_FUNC_MAP{
            {c10_npu::queue::COMPILE_AND_EXECUTE, &DqueueCompileExcute},
            {c10_npu::queue::EXECUTE_OPAPI, &DqueueCompileExcuteOpApi},
            {c10_npu::queue::ASYNC_MEMCPY, &DqueueAnyncMemcpy},
            {c10_npu::queue::RECORD_EVENT, &DqueueEvent},
            {c10_npu::queue::WAIT_EVENT, &DqueueEvent},
            {c10_npu::queue::LAZY_DESTROY_EVENT, &DqueueEvent},
            {c10_npu::queue::RESET_EVENT, &DqueueEvent},
        };
        void *cur_addr =
            (uint8_t *)data + (sizeof(c10_npu::queue::QueueParas) + at_npu::native::MAX_PARAS_BYTE_SIZE) * offset;
        auto cur_param = static_cast<c10_npu::queue::QueueParas *>(cur_addr);
        auto entry = DEQUEUE_CALL_FUNC_MAP.find(cur_param->paramType);
        if (entry != DEQUEUE_CALL_FUNC_MAP.end()) {
            entry->second(cur_param, category);
        }
    }
}
#endif

const std::string AclDateTypeToString(aclDataType descDType)
{
    std::map<const aclDataType, const std::string> ACL_TYPE_TO_STRING_TYPE_MAP = {
        {ACL_DT_UNDEFINED, "ACL_DT_UNDEFINED"},
        {ACL_FLOAT, "ACL_FLOAT"},
        {ACL_FLOAT16, "ACL_FLOAT16"},
        {ACL_INT8, "ACL_INT8"},
        {ACL_INT32, "ACL_INT32"},
        {ACL_UINT8, "ACL_UINT8"},
        {ACL_INT16, "ACL_INT16"},
        {ACL_UINT16, "ACL_UINT16"},
        {ACL_UINT32, "ACL_UINT32"},
        {ACL_INT64, "ACL_INT64"},
        {ACL_UINT64, "ACL_UINT64"},
        {ACL_DOUBLE, "ACL_DOUBLE"},
        {ACL_BOOL, "ACL_BOOL"},
        {ACL_STRING, "ACL_STRING"},
        {ACL_COMPLEX32, "ACL_COMPLEX32"},
        {ACL_COMPLEX64, "ACL_COMPLEX64"},
        {ACL_COMPLEX128, "ACL_COMPLEX128"},
        {ACL_BF16, "ACL_BF16"}};

    const auto iter = ACL_TYPE_TO_STRING_TYPE_MAP.find(descDType);
    return iter != ACL_TYPE_TO_STRING_TYPE_MAP.end() ? iter->second
                                                     : "DescDType not exists, descDType:" + std::to_string(descDType);
}

const std::string AclFormatToString(aclFormat descFormat)
{
    std::map<const aclFormat, const std::string> ACL_FORMAT_TO_STRING_TYPE_MAP = {
        {ACL_FORMAT_UNDEFINED, "ACL_FORMAT_UNDEFINED"},
        {ACL_FORMAT_NCHW, "ACL_FORMAT_NCHW"},
        {ACL_FORMAT_NHWC, "ACL_FORMAT_NHWC"},
        {ACL_FORMAT_ND, "ACL_FORMAT_ND"},
        {ACL_FORMAT_NC1HWC0, "ACL_FORMAT_NC1HWC0"},
        {ACL_FORMAT_FRACTAL_Z, "ACL_FORMAT_FRACTAL_Z"},
        {ACL_FORMAT_NC1HWC0_C04, "ACL_FORMAT_NC1HWC0_C04"},
        {ACL_FORMAT_HWCN, "ACL_FORMAT_HWCN"},
        {ACL_FORMAT_NDHWC, "ACL_FORMAT_NDHWC"},
        {ACL_FORMAT_FRACTAL_NZ, "ACL_FORMAT_FRACTAL_NZ"},
        {ACL_FORMAT_NCDHW, "ACL_FORMAT_NCDHW"},
        {ACL_FORMAT_NDC1HWC0, "ACL_FORMAT_NDC1HWC0"},
        {ACL_FRACTAL_Z_3D, "ACL_FRACTAL_Z_3D"}};

    const auto iter = ACL_FORMAT_TO_STRING_TYPE_MAP.find(descFormat);
    return iter != ACL_FORMAT_TO_STRING_TYPE_MAP.end()
               ? iter->second
               : "DescFormat not exists, descFormat:" + std::to_string(descFormat);
}
} // namespace native
} // namespace at_npu
