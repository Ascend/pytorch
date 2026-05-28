#include "arg_info.h"
#include "triton_runtime.h"
#include <cfloat>
#include <cstdint>
#include <sstream>

namespace triton_runtime {

ArgInfo::ArgInfo(const at::Tensor& tensor)
    : ptr_(tensor.data_ptr()),
      shape_(tensor.sizes().vec()),
      tensor_(tensor),
      scalar_size_(sizeof(void*)) {}

ArgInfo::ArgInfo(int32_t val) : scalar_(val), scalar_size_(sizeof(int32_t)) {}
ArgInfo::ArgInfo(int64_t val) : scalar_(static_cast<int32_t>(val)), scalar_size_(sizeof(int32_t)) {
    if (val < INT32_MIN || val > INT32_MAX) {
        TRT_DEBUG("warning: int64 %ld truncated to int32 %d", val, static_cast<int32_t>(val));
    }
}
ArgInfo::ArgInfo(float val)   : scalar_(val), scalar_size_(sizeof(float)) {}
ArgInfo::ArgInfo(double val)  : scalar_(static_cast<float>(val)), scalar_size_(sizeof(float)) {
    if (val != 0.0 && (val > FLT_MAX || val < -FLT_MAX)) {
        TRT_DEBUG("warning: double %f overflow when converting to float", val);
    }
}
ArgInfo::ArgInfo(bool val)    : scalar_(val), scalar_size_(sizeof(bool)) {}

bool ArgInfo::is_pointer() const {
    return tensor_.defined();
}

namespace {

struct ToString {
    std::string operator()(int32_t v) const { return std::to_string(v); }
    std::string operator()(float v)   const { return std::to_string(v); }
    std::string operator()(bool v)    const { return v ? "1" : "0"; }
};

constexpr ToString kToString;

} // anonymous namespace

std::string ArgInfo::scalar_value() const {
    if (tensor_.defined()) return "";
    return std::visit(kToString, scalar_);
}

void* ArgInfo::device_ptr() const {
    return ptr_;
}

const ArgInfo::ScalarVariant& ArgInfo::scalar() const {
    return scalar_;
}

const at::Tensor& ArgInfo::tensor() const {
    return tensor_;
}

std::vector<int64_t> ArgInfo::shape() const {
    return shape_;
}

size_t ArgInfo::scalar_size() const {
    return scalar_size_;
}

} // namespace triton_runtime
