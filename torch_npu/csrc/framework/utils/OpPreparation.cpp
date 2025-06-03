#include <ATen/record_function.h>
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/InferFormat.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/profiler/utils.h"
#endif
#include "torch_npu/csrc/core/npu/npu_log.h"

namespace at_npu {
namespace native {

UnifiedResult OpPreparation::binary_op_check(at::Tensor &out, const at::Tensor &a, const at::Tensor &b,
                                             bool check_mem_overlap)
{
    UnifiedResult unified_result;
    if (a.dtype() != b.dtype()) {
        std::tuple<at::ScalarType, c10::IntArrayRef> binary_op =
            NPUTensorIterator::binary_op(out, a, b, check_mem_overlap);
        unified_result.common_type = std::get<0>(binary_op);
        unified_result.common_shape = std::get<1>(binary_op);
    }
    return unified_result;
}

UnifiedResult OpPreparation::binary_op_check(at::Tensor &out, const at::Tensor &a, const c10::Scalar b,
                                             bool check_mem_overlap)
{
    UnifiedResult unified_result;
    std::tuple<at::ScalarType, c10::IntArrayRef> binary_op = NPUTensorIterator::binary_op(a, b);
    unified_result.common_type = std::get<0>(binary_op);
    unified_result.common_shape = std::get<1>(binary_op);
    return unified_result;
}

UnifiedResult OpPreparation::comparison_op_check(at::Tensor &out, const at::Tensor &a, const at::Tensor &b,
                                                 bool check_mem_overlap)
{
    UnifiedResult unified_result;
    if (a.dtype() != b.dtype()) {
        std::tuple<at::ScalarType, c10::IntArrayRef> comparison_op =
            NPUTensorIterator::comparison_op(out, a, b, check_mem_overlap);
        unified_result.common_type = std::get<0>(comparison_op);
        unified_result.common_shape = std::get<1>(comparison_op);
    }
    if (out.dtype() != a.dtype() && out.dtype() != b.dtype()) {
        unified_result.result_type_defined = true;
    }
    return unified_result;
}

UnifiedResult OpPreparation::unary_op_check(at::Tensor &out, const at::Tensor &a, bool check_mem_overlap)
{
    UnifiedResult unified_result;
    std::tuple<at::ScalarType, c10::IntArrayRef> unary_op = NPUTensorIterator::unary_op(out, a, check_mem_overlap);
    unified_result.common_type = std::get<0>(unary_op);
    unified_result.common_shape = std::get<1>(unary_op);
    return unified_result;
}

void OpPreparation::nullary_op(at::Tensor &out) { NPUTensorIterator::nullary_op(out); }

UnifiedResult OpPreparation::reduce_op_check(at::Tensor &out, const at::Tensor &a)
{
    UnifiedResult unified_result;
    std::tuple<at::ScalarType, c10::IntArrayRef> reduce_op = NPUTensorIterator::reduce_op(out, a);
    unified_result.common_type = std::get<0>(reduce_op);
    unified_result.common_shape = std::get<1>(reduce_op);
    return unified_result;
}

UnifiedResult OpPreparation::reduce_op_check(at::Tensor &out1, at::Tensor &out2, const at::Tensor &a)
{
    UnifiedResult unified_result;
    std::tuple<at::ScalarType, c10::IntArrayRef> reduce_op = NPUTensorIterator::reduce_op(out1, out2, a);
    unified_result.common_type = std::get<0>(reduce_op);
    unified_result.common_shape = std::get<1>(reduce_op);
    return unified_result;
}

// From CalcuOpUtil part
aclDataType OpPreparation::convert_to_acl_data_type(const at::ScalarType &data_type)
{
    return CalcuOpUtil::ConvertToAclDataType(data_type);
}

aclDataType OpPreparation::convert_to_acl_data_type(const at::ScalarType &data_type, const std::string &realDataType)
{
    return CalcuOpUtil::ConvertToAclDataType(data_type, realDataType);
}

at::Tensor OpPreparation::copy_scalar_to_device(const c10::Scalar &cpu_scalar, at::ScalarType scalar_data_type)
{
    return CalcuOpUtil::CopyScalarToDevice(cpu_scalar, scalar_data_type);
}

at::Tensor OpPreparation::copy_scalar_to_device(const c10::Scalar &cpu_scalar, at::ScalarType scalar_data_type,
                                                const c10::Device device)
{
    c10_npu::NPUGuard guard(device);
    return copy_scalar_to_device(cpu_scalar, scalar_data_type);
}

at::Tensor OpPreparation::copy_tensor_host_to_device(const at::Tensor &cpu_tensor)
{
    return CalcuOpUtil::CopyTensorHostToDevice(cpu_tensor);
}

bool OpPreparation::is_scalar_wrapped_to_tensor(const at::Tensor &tensor) { return IsCPUScalar(tensor); }

c10::SmallVector<int64_t, 5> OpPreparation::get_tensor_desc_base_sizes(const at::Tensor &tensor)
{
    return torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->get_npu_desc().base_sizes_;
}

int64_t OpPreparation::get_tensor_npu_format(const at::Tensor &tensor)
{
    return CalcuOpUtil::GetTensorNpuFormat(tensor);
}

static bool check_inplace_tensor(const std::initializer_list<at::Tensor> &src_list, at::Tensor &dst)
{
    bool is_inplace_tensor = false;
    // check whether dst is contained in src_list
    for (const auto &src : src_list) {
        if (dst.is_same(src)) {
            is_inplace_tensor = true;
            break;
        }
    }
    return is_inplace_tensor;
}

static void check_tensor_size(const std::initializer_list<at::Tensor> &src_list, at::Tensor &dst,
                              c10::IntArrayRef expect_size)
{
    bool is_inplace = check_inplace_tensor(src_list, dst);
    // Preserve legacy resizing behavior of out=... arguments
    if (!dst.sizes().equals(expect_size)) {
        TORCH_CHECK(!is_inplace, "output with shape ", dst.sizes(), " doesn't match the broadcast shape ", expect_size,
                    OPS_ERROR(ErrCode::PARAM));
        dst.resize_(expect_size);
    }
    return;
}

void OpPreparation::check_tensor(const std::initializer_list<at::Tensor> &src_list, at::Tensor &dst,
                                 at::ScalarType expect_dtype, c10::IntArrayRef expect_size)
{
    check_memory(src_list, {dst});
    TORCH_CHECK(torch_npu::utils::is_npu(dst), "output with device ", dst.device(),
                " doesn't match the desired device NPU", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dst.scalar_type() == expect_dtype, "expected dtype ", expect_dtype, " but got dtype ",
                dst.scalar_type(), OPS_ERROR(ErrCode::TYPE));
    check_tensor_size(src_list, dst, expect_size);
}

void OpPreparation::check_tensor(const std::initializer_list<at::Tensor> &src_list, at::Tensor &dst,
                                 c10::IntArrayRef expect_size)
{
    check_memory(src_list, {dst});
    TORCH_CHECK(torch_npu::utils::is_npu(dst), "output with device ", dst.device(),
                " doesn't match the desired device NPU", OPS_ERROR(ErrCode::PARAM));
    check_tensor_size(src_list, dst, expect_size);
}

void OpPreparation::check_tensor(const std::initializer_list<at::Tensor> &src_list, at::Tensor &dst,
                                 const at::Tensor &expect_tensor)
{
    check_tensor(src_list, dst, expect_tensor.scalar_type(), expect_tensor.sizes());
}

void OpPreparation::check_tensor(const std::initializer_list<at::Tensor> &src_list, at::Tensor &dst,
                                 const at::Tensor &expect_tensor, c10::IntArrayRef expect_size)
{
    check_tensor(src_list, dst, expect_tensor.scalar_type(), expect_size);
}

void OpPreparation::check_memory(const std::initializer_list<at::Tensor> &inputs,
                                 const std::initializer_list<at::Tensor> &outputs)
{
    c10::SmallVector<at::Tensor, N> in = inputs;
    c10::SmallVector<at::Tensor, N> out = outputs;
    CalcuOpUtil::CheckMemoryOverLaps(in, out);
}

at::Tensor OpPreparation::cast_to_ori_format(const at::Tensor &tensor)
{
    auto &tensor_desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
    auto ret = custom_ops::npu_format_cast(tensor, tensor_desc.origin_format_);
    return ret;
}

at::Tensor &OpPreparation::cast_to_ori_format(at::Tensor &tensor)
{
    auto &tensor_desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
    NPUNativeFunctions::npu_format_cast_(tensor, tensor_desc.origin_format_);
    return tensor;
}

at::Tensor OpPreparation::apply_tensor(const at::Tensor &src) { return apply_tensor(src, src.sizes()); }

at::Tensor OpPreparation::apply_tensor(const at::Tensor &src, c10::IntArrayRef sizes)
{
    return apply_tensor_with_format(sizes, src.options(), CalcuOpUtil::GetTensorNpuFormat(src));
}

at::Tensor OpPreparation::apply_tensor(const at::Tensor &src, const c10::TensorOptions &options)
{
    return apply_tensor_with_format(src.sizes(), options, CalcuOpUtil::GetTensorNpuFormat(src));
}

at::Tensor OpPreparation::apply_tensor(c10::IntArrayRef sizes, const c10::TensorOptions &options, const at::Tensor &src)
{
    return apply_tensor_with_format(sizes, options, CalcuOpUtil::GetTensorNpuFormat(src));
}

at::Tensor OpPreparation::apply_tensor_with_format(const at::Tensor &src, int64_t format, bool keep_format)
{
    return apply_tensor_with_format(src, src.sizes(), format, keep_format);
}

at::Tensor OpPreparation::apply_tensor_with_format(const at::Tensor &src, c10::IntArrayRef sizes, int64_t format,
                                                   bool keep_format)
{
    return apply_tensor_with_format(sizes, src.options(), format, keep_format);
}

at::Tensor OpPreparation::apply_tensor_with_format(c10::IntArrayRef sizes, const c10::TensorOptions &options,
                                                   int64_t format, bool keep_format)
{
    TORCH_CHECK(options.device().type() == c10::DeviceType::PrivateUse1,
                "Expected all tensors to be on the same device. "
                "Expected NPU tensor, please check whether the input tensor device is correct.",
                OPS_ERROR(ErrCode::TYPE));
    auto fixFormat = InferFormat::GuessStorageFormat(sizes, static_cast<aclFormat>(format));
    if (options.dtype_opt() == at::ScalarType::Double && !FormatHelper::IsBaseFormatType(static_cast<aclFormat>(format))) {
        ASCEND_LOGW("NPU don't support create double dtype tensor with inner format, repalce with base format.");
        fixFormat = FormatHelper::GetBaseFormat(static_cast<aclFormat>(format));
    }
    return NPUNativeFunctions::unsafe_empty_with_format(sizes, c10::optTypeMetaToScalarType(options.dtype_opt()),
                                                        options.layout_opt(), options.device_opt(),
                                                        options.pinned_memory_opt(), fixFormat, keep_format);
}

at::Tensor OpPreparation::apply_tensor_with_sizes(c10::IntArrayRef sizes, const c10::TensorOptions &options)
{
    auto format = InferFormat::GuessBaseFormat(sizes);
    return NPUNativeFunctions::empty_with_format(sizes, c10::optTypeMetaToScalarType(options.dtype_opt()),
                                                 options.layout_opt(), options.device_opt(),
                                                 options.pinned_memory_opt(), format, c10::nullopt);
}

void OpPreparation::CheckOut(const std::initializer_list<at::Tensor> &inputs, at::Tensor &output, at::Tensor dst)
{
    CheckOut(inputs, output, CalcuOpUtil::GetTensorNpuFormat(dst), dst.scalar_type(), dst.sizes());
}

void OpPreparation::CheckOut(const std::initializer_list<at::Tensor> &inputs, at::Tensor &output, at::Tensor dst,
                             c10::IntArrayRef shape)
{
    CheckOut(inputs, output, CalcuOpUtil::GetTensorNpuFormat(dst), dst.scalar_type(), shape);
}

void OpPreparation::CheckOut(const std::initializer_list<at::Tensor> &input, at::Tensor &output, int64_t format,
                             at::ScalarType dtype, c10::IntArrayRef shape)
{
    // Check that the outputs have no internal overlap
    // and do not share memory with inputs.
    c10::SmallVector<at::Tensor, N> inputs{input};
    c10::SmallVector<at::Tensor, N> outputs = {output};
    CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);
    TORCH_CHECK(torch_npu::utils::is_npu(output), "output with device ", output.device(),
                " doesn't match the desired device NPU", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(output.scalar_type() == dtype, "expected dtype ", dtype, " but got dtype ", output.scalar_type(),
                OPS_ERROR(ErrCode::TYPE));

    bool is_read_write = false;
    // check if output is also an input
    for (const auto &local_input : inputs) {
        if (output.is_same(local_input)) {
            is_read_write = true;
            break;
        }
    }

    // Preserve legacy resizing behavior of out=... arguments
    if (!output.sizes().equals(shape)) {
        TORCH_CHECK(!is_read_write, "output with shape ", output.sizes(), " doesn't match the broadcast shape ", shape,
                    OPS_ERROR(ErrCode::PARAM));
        output.resize_(shape);
    }

    if (CalcuOpUtil::GetTensorNpuFormat(output) != format) {
        TORCH_CHECK(!is_read_write, "can not cast format when output is input", OPS_ERROR(ErrCode::NOT_SUPPORT));
        NPUNativeFunctions::npu_format_cast_(output, format);
    }
}

at::Tensor OpPreparation::CastBackToOriFormat(const at::Tensor &tensor)
{
    auto &tensor_desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
    auto ret = custom_ops::npu_format_cast(tensor, tensor_desc.origin_format_);
    return ret;
}

at::Tensor &OpPreparation::CastBackToOriFormat(at::Tensor &tensor)
{
    auto &tensor_desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
    NPUNativeFunctions::npu_format_cast_(tensor, tensor_desc.origin_format_);
    return tensor;
}

int8_t OpPreparation::get_cube_math_type(bool allowHf32) { return CalcuOpUtil::GetCubeMathType(allowHf32); }

inline at::Tensor apply_tensor_use_empty(c10::IntArrayRef sizes, const c10::TensorOptions &options)
{
    c10::optional<c10::Device> device_opt = options.device_opt();
    if (c10::device_or_default(device_opt).type() != c10::DeviceType::PrivateUse1) {
        device_opt = at::Device(c10::DeviceType::PrivateUse1);
    }
    return NPUNativeFunctions::empty(sizes, options.dtype().toScalarType(), c10::nullopt, device_opt, false,
                                     c10::MemoryFormat::Contiguous);
}

at::Tensor OpPreparation::apply_tensor_without_format(const at::Tensor &src)
{
    return apply_tensor_use_empty(src.sizes(), src.options());
}

at::Tensor OpPreparation::apply_tensor_without_format(const at::Tensor &src, c10::IntArrayRef sizes)
{
    return apply_tensor_use_empty(sizes, src.options());
}

at::Tensor OpPreparation::apply_tensor_without_format(c10::IntArrayRef sizes, const c10::TensorOptions &options)
{
    return apply_tensor_use_empty(sizes, options);
}

at::Tensor OpPreparation::unsafe_empty_workspace(uint64_t workspace_size)
{
#ifndef BUILD_LIBTORCH
    torch_npu::profiler::NPURecordFunction profiler_guard;
#endif
    ASCEND_LOGD("Alloc workspace %zu bytes unsafely.", workspace_size);
    c10::Allocator *allocator = c10_npu::NPUCachingAllocator::get();
    c10::intrusive_ptr<c10::StorageImpl> storage_impl =
        torch_npu::make_npu_storage_impl(c10::StorageImpl::use_byte_size_t(), c10::SymInt(workspace_size),
                                         allocator->allocate(workspace_size), allocator, true);
    static auto dtype = c10::scalarTypeToTypeMeta(dtype_or_default(at::kByte));
    auto tensor = at::detail::make_tensor<torch_npu::NPUTensorImpl>(storage_impl, dtype);
    tensor.unsafeGetTensorImpl()->empty_tensor_restride(c10::MemoryFormat::Contiguous);
    return tensor;
}

at::Tensor OpPreparation::unsafe_empty_workspace(uint64_t workspace_size, aclrtStream stream)
{
#ifndef BUILD_LIBTORCH
    torch_npu::profiler::NPURecordFunction profiler_guard;
#endif
    ASCEND_LOGD("Alloc workspace %zu bytes unsafely.", workspace_size);
    c10::Allocator *allocator = c10_npu::NPUWorkspaceAllocator::get();
    c10::intrusive_ptr<c10::StorageImpl> storage_impl = c10::make_intrusive<torch_npu::NPUStorageImpl>(
        c10::StorageImpl::use_byte_size_t(), workspace_size,
        c10_npu::NPUWorkspaceAllocator::malloc_with_stream(workspace_size, stream), allocator, true);
    static auto dtype = c10::scalarTypeToTypeMeta(dtype_or_default(at::kByte));
    auto tensor = at::detail::make_tensor<torch_npu::NPUTensorImpl>(storage_impl, dtype);
    tensor.unsafeGetTensorImpl()->empty_tensor_restride(c10::MemoryFormat::Contiguous);
    return tensor;
}

at::Tensor OpPreparation::ApplyTensor(const at::Tensor &src) { return ApplyTensor(src, src.sizes()); }

at::Tensor OpPreparation::ApplyTensor(const at::Tensor &src, c10::IntArrayRef sizes)
{
    return ApplyTensorWithFormat(sizes, src.options(), CalcuOpUtil::GetTensorNpuFormat(src));
}

at::Tensor OpPreparation::ApplyTensor(const at::Tensor &src, const c10::TensorOptions &options)
{
    return ApplyTensorWithFormat(src.sizes(), options, CalcuOpUtil::GetTensorNpuFormat(src));
}

at::Tensor OpPreparation::ApplyTensor(c10::IntArrayRef sizes, const c10::TensorOptions &options, const at::Tensor &src)
{
    return ApplyTensorWithFormat(sizes, options, CalcuOpUtil::GetTensorNpuFormat(src));
}

at::Tensor OpPreparation::ApplyTensorWithFormat(const at::Tensor &src, int64_t format, bool keep_format)
{
    return ApplyTensorWithFormat(src, src.sizes(), format, keep_format);
}

at::Tensor OpPreparation::ApplyTensorWithFormat(const at::Tensor &src, c10::IntArrayRef sizes, int64_t format,
                                                bool keep_format)
{
    return ApplyTensorWithFormat(sizes, src.options(), format, keep_format);
}

at::Tensor OpPreparation::ApplyTensorWithFormat(c10::IntArrayRef sizes, const c10::TensorOptions &options,
                                                int64_t format, bool keep_format)
{
    TORCH_CHECK(options.device().type() == c10::DeviceType::PrivateUse1,
                "Expected all tensors to be on the same device. "
                "Expected NPU tensor, please check whether the input tensor device is correct.",
                OPS_ERROR(ErrCode::TYPE));
    auto fixFormat = InferFormat::GuessStorageFormat(sizes, static_cast<aclFormat>(format));
    return NPUNativeFunctions::unsafe_empty_with_format(sizes, c10::optTypeMetaToScalarType(options.dtype_opt()),
                                                        options.layout_opt(), options.device_opt(),
                                                        options.pinned_memory_opt(), fixFormat, keep_format);
}

at::Tensor OpPreparation::ApplyTensorWithSizes(c10::IntArrayRef sizes, const c10::TensorOptions &options)
{
    auto format = InferFormat::GuessBaseFormat(sizes);
    return NPUNativeFunctions::empty_with_format(sizes, c10::optTypeMetaToScalarType(options.dtype_opt()),
                                                 options.layout_opt(), options.device_opt(),
                                                 options.pinned_memory_opt(), format, c10::nullopt);
}

void OpPreparation::CheckMemory(const std::initializer_list<at::Tensor> &inputs,
                                const std::initializer_list<at::Tensor> &outputs)
{
    c10::SmallVector<at::Tensor, N> in = inputs;
    c10::SmallVector<at::Tensor, N> out = outputs;
    CalcuOpUtil::CheckMemoryOverLaps(in, out);
}

bool OpPreparation::IsCPUScalar(const at::Tensor &tensor)
{
    if (tensor.dim() == 0 && !torch_npu::utils::is_npu(tensor)) {
        return true;
    }
    return false;
}

} // namespace native
} // namespace at_npu
