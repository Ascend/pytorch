#pragma once
#include <torch/torch.h>
#include <string>
#include <variant>

namespace triton_runtime {

class ArgInfo {
public:
    using ScalarVariant = std::variant<int32_t, float, bool>;

    ArgInfo(const at::Tensor& tensor);
    ArgInfo(int32_t val);
    ArgInfo(int64_t val);
    ArgInfo(float val);
    ArgInfo(double val);
    ArgInfo(bool val);

    bool is_pointer() const;
    std::string scalar_value() const;

    void* device_ptr() const;
    const at::Tensor& tensor() const;
    const ScalarVariant& scalar() const;
    std::vector<int64_t> shape() const;
    size_t scalar_size() const;

private:
    void* ptr_ = nullptr;
    ScalarVariant scalar_ = int32_t(0);
    std::vector<int64_t> shape_;
    at::Tensor tensor_;
    size_t scalar_size_ = sizeof(int32_t);
};

} // namespace triton_runtime
