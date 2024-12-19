#include "torch_npu/csrc/framework/OpHook.h"

namespace at_npu {
namespace native {

OpHook& OpHook::GetInstance()
{
    static OpHook instance;
    return instance;
}

OpHook::~OpHook() {}

void OpHook::RegisterBeginFn(BeginFn fn)
{
    this->begin_fn_ = fn;
}

void OpHook::RegisterEndFn(EndFn fn)
{
    this->end_fn_ = fn;
}

void OpHook::RegisterPreFn(PreFn fn)
{
    this->pre_fn_ = fn;
}

void OpHook::RegisterPostFn(PostFn fn)
{
    this->post_fn_ = fn;
}

void OpHook::HookBegin(std::string& op_name)
{
    if (this->begin_fn_ != nullptr) {
        this->begin_fn_(op_name);
    }
}

void OpHook::HookEnd()
{
    if (this->end_fn_ != nullptr) {
        this->end_fn_();
    }
}

void OpHook::HookArg(const at::Tensor& at_tensor)
{
    if (this->is_in_pre_hook_ == true) {
        if (this->pre_fn_ != nullptr) {
            this->pre_fn_(at_tensor);
        }
    } else {
        if (this->post_fn_ != nullptr) {
            this->post_fn_(at_tensor);
        }
    }
}

void OpHook::HookArg(const c10::optional<at::Tensor>& opt_tensor)
{
    if (opt_tensor.has_value()) {
        HookArg(opt_tensor.value());
    }
}

void OpHook::HookArg(const at::TensorList& at_tensor_list)
{
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        HookArg(at_tensor_list[i]);
    }
}

void OpHook::HookArg(const c10::optional<at::TensorList>& opt_tensor_list)
{
    if (opt_tensor_list.has_value()) {
        HookArg(opt_tensor_list.value());
    }
}

void OpHook::HookArg(const c10::List<c10::optional<at::Tensor>>& opt_tensor_list)
{
    for (c10::optional<at::Tensor> opt_tensor : opt_tensor_list) {
        HookArg(opt_tensor);
    }
}

void OpHook::HookArg(const std::vector<at::Tensor>& at_tensor_vector)
{
    for (size_t i = 0; i < at_tensor_vector.size(); i++) {
        HookArg(at_tensor_vector[i]);
    }
}

void OpHook::HookArg(const std::vector<std::vector<at::Tensor>>& at_tensor_vector_vector)
{
    for (const auto i : c10::irange(at_tensor_vector_vector.size())) {
        for (const auto j : c10::irange(at_tensor_vector_vector[0].size())) {
            HookArg(at_tensor_vector_vector[i][j]);
        }
    }
}

void OpHook::HookArg(const at::ITensorListRef& tensors)
{
    for (const at::Tensor& at_tensor : tensors) {
        HookArg(at_tensor);
    }
}

OpHook::OpHook() : begin_fn_(nullptr), end_fn_(nullptr), pre_fn_(nullptr), post_fn_(nullptr), is_in_pre_hook_(true) {}

void RegisterOpHookBeginFn(BeginFn fn)
{
    at_npu::native::OpHook::GetInstance().RegisterBeginFn(fn);
}
void RegisterOpHookEndFn(EndFn fn)
{
    at_npu::native::OpHook::GetInstance().RegisterEndFn(fn);
}
void RegisterOpHookPreFn(PreFn fn)
{
    at_npu::native::OpHook::GetInstance().RegisterPreFn(fn);
}
void RegisterOpHookPostFn(PostFn fn)
{
    at_npu::native::OpHook::GetInstance().RegisterPostFn(fn);
}

} // namespace native
} // namespace at_npu
