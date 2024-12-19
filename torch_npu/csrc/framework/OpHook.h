#ifndef __CSRC_FRAMEWORK_OPHOOK_H__
#define __CSRC_FRAMEWORK_OPHOOK_H__

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch_npu/csrc/core/npu/NPUMacros.h>

namespace at_npu {
namespace native {

using BeginFn = void (*)(const std::string& op_name);
using EndFn = void (*)();
using PreFn = void (*)(const at::Tensor& at_tensor);
using PostFn = void (*)(const at::Tensor& at_tensor);

class OpHook {
public:
    static OpHook& GetInstance();
    ~OpHook();

    void RegisterBeginFn(BeginFn fn);
    void RegisterEndFn(EndFn fn);
    void RegisterPreFn(PreFn fn);
    void RegisterPostFn(PostFn fn);

    void HookBegin(std::string& op_name);
    void HookEnd();

    void HookArg(const at::Tensor& at_tensor);
    void HookArg(const c10::optional<at::Tensor>& opt_tensor);
    void HookArg(const at::TensorList& at_tensor_list);
    void HookArg(const c10::optional<at::TensorList>& opt_tensor_list);
    void HookArg(const c10::List<c10::optional<at::Tensor>>& opt_tensor_list);
    void HookArg(const std::vector<at::Tensor>& at_tensor_vector);
    void HookArg(const std::vector<std::vector<at::Tensor>>& at_tensor_vector_vector);
    void HookArg(const at::ITensorListRef& tensors);

    template <std::size_t... Is, typename... Ts>
    void HookArg(const std::tuple<Ts...>& t, std::index_sequence<Is...>)
    {
        (HookArg(std::get<Is>(t)), ...);
        return;
    }

    template <typename... Ts>
    void HookArg(const std::tuple<Ts...>& t)
    {
        HookArg(t, std::make_index_sequence<sizeof...(Ts)>{});
    }

    template <typename T>
    void HookArg(T value)
    {
        return;
    }

    template <typename... Ts>
    void HookArgs(Ts&... args)
    {
        return (HookArg(args), ...);
    }

    template <typename... Ts>
    void PreHook(std::string op_name, Ts&... args)
    {
        this->is_in_pre_hook_ = true;
        HookBegin(op_name);
        HookArgs(args...);
    }

    template <typename... Ts>
    void PostHook(Ts&... args)
    {
        this->is_in_pre_hook_ = false;
        HookArgs(args...);
        HookEnd();
    }

private:
    OpHook();

    BeginFn begin_fn_ = nullptr;
    EndFn end_fn_ = nullptr;
    PreFn pre_fn_ = nullptr;
    PostFn post_fn_ = nullptr;
    bool is_in_pre_hook_ = true;
};

TORCH_NPU_API void RegisterOpHookBeginFn(BeginFn fn);
TORCH_NPU_API void RegisterOpHookEndFn(EndFn fn);
TORCH_NPU_API void RegisterOpHookPreFn(PreFn fn);
TORCH_NPU_API void RegisterOpHookPostFn(PostFn fn);

} // namespace native
} // namespace at_npu

#endif  // __CSRC_FRAMEWORK_OPHOOK_H__
