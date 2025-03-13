#include "torch_npu/csrc/aten/mirror/NPUTypeProperties.h"
#include "torch_npu/csrc/aten/mirror/NPUTensorIterator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace at_npu {
namespace native {

std::tuple<at::ScalarType, c10::IntArrayRef> NPUTensorIterator::binary_op(
    at::Tensor &out,
    const at::Tensor &a,
    const at::Tensor &b,
    bool check_mem_overlap)
{
    auto iter = NPUTensorIterator();
    iter.add_output(out);
    iter.add_input(a);
    iter.add_input(b);
    iter.promote_common_dtype();
    iter.compute_types();
    auto common_type = iter.common_dtype();
    auto common_shape = a.sizes();
    return std::tie(common_type, common_shape);
}

std::tuple<at::ScalarType, c10::IntArrayRef> NPUTensorIterator::binary_op(
    const at::Tensor &a,
    const c10::Scalar b)
{
    at::ScalarType scalar_type;
    if (b.isFloatingPoint()) {
        scalar_type = at::ScalarType::Float;
    } else if (b.isBoolean()) {
        scalar_type = at::ScalarType::Bool;
    } else if (b.isComplex()) {
        scalar_type = at::ScalarType::ComplexFloat;
    } else {
        AT_ASSERT(b.isIntegral(false), OPS_ERROR(ErrCode::PARAM));
        scalar_type = at::ScalarType::Int;
    }
    if (a.scalar_type() == at::ScalarType::Half) {
        scalar_type = at::ScalarType::Half;
    }
    if (a.scalar_type() == at::ScalarType::BFloat16) {
        scalar_type = at::ScalarType::BFloat16;
    }
    if (a.scalar_type() != scalar_type) {
        scalar_type = result_type(a.scalar_type(), scalar_type);
    }
    auto common_shape = a.sizes();
    return std::tie(scalar_type, common_shape);
}

std::tuple<at::ScalarType, c10::IntArrayRef> NPUTensorIterator::comparison_op(
    at::Tensor &out,
    const at::Tensor &a,
    const at::Tensor &b,
    bool check_mem_overlap)
{
    auto iter = NPUTensorIterator();
    iter.add_output(out);
    iter.add_input(a);
    iter.add_input(b);
    iter.compute_common_dtype_only_for_inputs();
    iter.compute_types();
    auto common_type = iter.common_dtype();
    auto common_shape = a.sizes();
    return std::tie(common_type, common_shape);
}

std::tuple<at::ScalarType, c10::IntArrayRef> NPUTensorIterator::unary_op(
    at::Tensor &out,
    const at::Tensor &a,
    bool check_mem_overlap)
{
    auto iter = NPUTensorIterator();
    iter.add_output(out);
    iter.add_input(a);
    iter.num_outputs_ = 1;
    iter.compute_types();
    auto common_type = iter.common_dtype();
    auto common_shape = a.sizes();
    return std::tie(common_type, common_shape);
}

void NPUTensorIterator::nullary_op(at::Tensor &out)
{
    auto iter = NPUTensorIterator();
    iter.add_output(out);
    iter.compute_types();
}

std::tuple<at::ScalarType, c10::IntArrayRef> NPUTensorIterator::reduce_op(at::Tensor &out, const at::Tensor &a)
{
    TORCH_INTERNAL_ASSERT(out.defined(), OPS_ERROR(ErrCode::PARAM));
    auto iter = NPUTensorIterator();
    iter.add_output(out);
    iter.add_input(a);
    iter.promote_npu_output_dtypes_ = true;
    iter.is_reduction_ = true;
    // (Ascend): This is only really necessary for arg{min,max}
    iter.compute_common_dtype_only_for_inputs();
    iter.compute_types();
    auto common_type = iter.common_dtype();
    auto common_shape = a.sizes();
    return std::tie(common_type, common_shape);
}

std::tuple<at::ScalarType, c10::IntArrayRef> NPUTensorIterator::reduce_op(
    at::Tensor &out1,
    at::Tensor &out2,
    const at::Tensor &a)
{
    TORCH_INTERNAL_ASSERT(out1.defined(), OPS_ERROR(ErrCode::PARAM));
    TORCH_INTERNAL_ASSERT(out2.defined(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(out1.dim() == out2.dim(),
        "reduce_op(): expected both outputs to have same number of dims, but output1 has ", out1.dim(),
        " and output2 has ", out2.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(out1.sizes() == out2.sizes(),
        "reduce_op(): expected both outputs to have same sizes, but output1 has ", out1.sizes(),
        " and output2 has ", out2.sizes(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(out1.strides() == out2.strides(),
        "reduce_op(): expected both outputs to have same strides, but output1 has ", out1.strides(),
        " and output2 has ", out2.strides(), OPS_ERROR(ErrCode::PARAM));
    auto iter = NPUTensorIterator();
    iter.add_output(out1);
    iter.add_output(out2);
    iter.add_input(a);
    iter.promote_npu_output_dtypes_ = true;
    iter.is_reduction_ = true;
    iter.compute_types();
    auto common_type = iter.common_dtype();
    auto common_shape = a.sizes();
    return std::tie(common_type, common_shape);
}

static std::tuple<at::ScalarType, bool> compute_common_type_(at::ArrayRef<NPUOperandInfo> operands)
{
    // See [Result type computation] in NPUTensorIterator.h
    auto common_type = at::ScalarType::Undefined;
    bool all_same_type = true;
    for (const auto &op : operands) {
        if (!op.tensor.defined())
            continue;
        // don't handle scalars
        if (op.tensor.dim() > 0) {
            at::ScalarType current = op.current_dtype;
            if (current == at::ScalarType::Undefined) {
                all_same_type = false;
                break;
            }
            if (common_type == at::ScalarType::Undefined) {
                common_type = current;
            }
            if (common_type != current) {
                all_same_type = false;
                break;
            }
        } else {
            all_same_type = false;
            break;
        }
    }
    if (all_same_type) {
        return std::make_tuple(common_type, true);
    }

    ResultTypeState state = {};
    for (const auto &op : operands) {
        state = update_result_type_state(op.tensor, state);
    }
    auto dtype = result_type(state);

    auto result = std::make_tuple(dtype, false);
    TORCH_INTERNAL_ASSERT(dtype != at::ScalarType::Undefined, OPS_ERROR(ErrCode::TYPE));
    return result;
}

std::tuple<at::ScalarType, bool> NPUTensorIterator::compute_common_type()
{
    return compute_common_type_(operands_);
}

void NPUTensorIterator::compute_types()
{
    bool missing_dtypes = false;
    bool missing_output_dtypes = false;
    common_dtype_ = dtype();
    for (auto &op : operands_) {
        if (!op.tensor.defined() && !op.is_type_defined()) {
            missing_dtypes = true;
            if (op.is_output) {
                missing_output_dtypes = true;
            }
        }
    }

    if (common_dtype_strategy_ == CommonDTypeStrategy::PROMOTE_INPUTS) {
        TORCH_CHECK(!missing_output_dtypes,
            "unable to compute and promote common dtype based only on inputs if there are missing dtypes for outputs",
            OPS_ERROR(ErrCode::TYPE));
    }
    bool compute_common_dtype = (common_dtype_strategy_ != CommonDTypeStrategy::NONE);
    bool compute_common_dtype_only_for_inputs = (common_dtype_strategy_ == CommonDTypeStrategy::PROMOTE_INPUTS);
    if (missing_dtypes || compute_common_dtype) {
        auto operands = compute_common_dtype_only_for_inputs ?
            at::ArrayRef<NPUOperandInfo>(operands_).slice(noutputs()) : operands_;
        auto common_type = compute_common_type_(operands);
        common_dtype_ = std::get<0>(common_type);
    }
}

} // namespace native
} // namespace at_npu
