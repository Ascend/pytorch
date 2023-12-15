#ifndef __PLUGIN_NATIVE_UTILS_NPU_TENSOR_ITERATOR__
#define __PLUGIN_NATIVE_UTILS_NPU_TENSOR_ITERATOR__

#include <bitset>
#include <ATen/ATen.h>
#include <functional>
#include <c10/util/SmallVector.h>
#include <c10/util/TypeCast.h>

#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
namespace at_npu {
namespace native {

struct TORCH_NPU_API NPUOperandInfo {
    using StrideVector = c10::SmallVector<int64_t, 6>;
    NPUOperandInfo() {}
    explicit NPUOperandInfo(const at::Tensor& t) : tensor(t) {
        if (t.defined()) {
            target_dtype = t.scalar_type();
            current_dtype = target_dtype;
        }
        validate();
    }
    NPUOperandInfo(const at::Tensor& t, at::ScalarType dtype)
        : tensor(t), target_dtype(dtype), current_dtype(t.scalar_type()) {
        validate();
    }

    bool is_type_defined() const {
        return target_dtype != at::ScalarType::Undefined;
    }

    void validate() {
        TORCH_CHECK(
            !tensor.defined() || tensor.layout() == at::kStrided,
            "unsupported tensor layout: ", tensor.layout());
    }

    StrideVector stride_bytes;
    at::Tensor tensor;
    at::ScalarType target_dtype = at::ScalarType::Undefined;
    at::ScalarType current_dtype = at::ScalarType::Undefined;
    bool is_output = false;
}; // class NPUOperandInfo

enum class CommonDTypeStrategy : uint8_t {
    NONE, // Do not compute a common dtype
    CHECK, // Compute and validate a common dtype but don't promote.
    PROMOTE_INPUTS, // Promote common dtype but only validate inputs (comparison ops have boolean output)
    PROMOTE // Promote to common dtype.
};

class TORCH_NPU_API NPUTensorIterator {
public:
    NPUTensorIterator() {}
    ~NPUTensorIterator() {}

    static std::tuple<at::ScalarType, c10::IntArrayRef> binary_op(
        at::Tensor& out,
        const at::Tensor& a,
        const at::Tensor& b,
        bool check_mem_overlap = false);
    static std::tuple<at::ScalarType, c10::IntArrayRef> binary_op(
        const at::Tensor& a,
        const c10::Scalar b);
    static std::tuple<at::ScalarType, c10::IntArrayRef> comparison_op(
        at::Tensor& out,
        const at::Tensor& a,
        const at::Tensor& b,
        bool check_mem_overlap = false);
    static std::tuple<at::ScalarType, c10::IntArrayRef> unary_op(
        at::Tensor& out,
        const at::Tensor& a,
        bool check_mem_overlap = false);
    static void nullary_op(at::Tensor& out);
    static std::tuple<at::ScalarType, c10::IntArrayRef> reduce_op(
        at::Tensor& out,
        const at::Tensor& a);
    static std::tuple<at::ScalarType, c10::IntArrayRef> reduce_op(
        at::Tensor& out1,
        at::Tensor& out2,
        const at::Tensor& a);

    int noutputs() const {
        return num_outputs_;
    }

    c10::IntArrayRef strides(int arg) const {
        return operands_[arg].stride_bytes;
    }
    at::ScalarType dtype(int arg = 0) const {
        return operands_[arg].current_dtype;
    }
    at::ScalarType common_dtype() const {
        return common_dtype_;
    }

    const c10::SmallVector<NPUOperandInfo, 4> GetOperandInfo() const {
        return operands_;
    }

    // Construction
    void add_output(const at::Tensor& output) {
        operands_.emplace_back(output);
        num_outputs_++;
    }

    void add_input(const at::Tensor& input) {
        operands_.emplace_back(input);
    }

    void promote_common_dtype() {
        common_dtype_strategy_ = CommonDTypeStrategy::PROMOTE;
    }

    void compute_common_dtype_only_for_inputs() {
        common_dtype_strategy_ = CommonDTypeStrategy::PROMOTE_INPUTS;
    }

    void compute_types();
    std::tuple<at::ScalarType, bool> compute_common_type();

private:
    c10::SmallVector<NPUOperandInfo, 4> operands_;
    int num_outputs_ = 0;
    bool promote_npu_output_dtypes_ = false;
    bool all_ops_same_shape_ = false;
    at::ScalarType common_dtype_ = at::ScalarType::Undefined;
    bool is_reduction_ = false;
    CommonDTypeStrategy common_dtype_strategy_ = CommonDTypeStrategy::CHECK;
}; // class NPUTensorIterator

} // namespace native
} // namespace at

#endif
