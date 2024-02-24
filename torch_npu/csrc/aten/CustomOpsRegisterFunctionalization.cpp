#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include <torch/library.h>

namespace at_npu {
namespace native {
namespace {

// This keyset is used by functionalization when it calls into meta kernels
// to accurately propagate stride metadata.
// Exclude any modes: the purpose of calling into meta kernels is only as an implementation
// detail to perform shape inference, and we don't want any modal keys to run.
// Specifically, we want to prevent functionalization and Python modes from running.
constexpr auto exclude_keys_for_meta_dispatch =
    c10::functorch_transforms_ks |
    c10::DispatchKeySet({
                            c10::DispatchKey::FuncTorchDynamicLayerBackMode,
                            c10::DispatchKey::FuncTorchDynamicLayerFrontMode,
                            c10::DispatchKey::Python
                        });

inline at::Tensor to_meta(const at::Tensor &t)
{
    if (!t.defined())
    {
        return t;
    }

    return at::native::empty_strided_meta_symint(
        t.sym_sizes(),
        t.sym_strides(),
        c10::make_optional(t.scalar_type()),
        c10::make_optional(t.layout()),
        c10::make_optional(c10::Device(at::kMeta)),
        c10::nullopt
    );
}
}

// Regster functionalization for custom op scatter_update_.
// To Get these mutable operators to work with functionalization requires some extra work.
// We need to register a corresponding out-of-place variant of the op,
// and register a functionalization kernel that performs some boilerplate to
// teach functionalization how to map from the mutable op to the out-of-place op.
at::Tensor &scatter_update__functionalization(
    at::Tensor &self,
    const at::Tensor &indices,
    const at::Tensor &updates,
    int64_t axis)
{
    {
        // Before converting the mutable op to its functional variant, run meta tensors through the original op.
        // This will help us catch shape errors that apply to inplace ops that wouldn't apply to their functional
        // variants. (We can only do this for inplace ops today though, because they technicaly all support meta
        // tensors).
        auto self_meta = to_meta(self);
        auto indices_meta = to_meta(indices);
        auto updates_meta = to_meta(updates);

        at::AutoDispatchSkipFunctionalize func_guard;
        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);
        static auto scatter_update___handle = c10::Dispatcher::singleton()
            // specify namespace::op_name, op_overload_name
            .findSchemaOrThrow("npu::scatter_update_", "")
            // specify the C++ schema of the original mutable op
            .typed<at::Tensor & (at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t)>();

        scatter_update___handle.call(self_meta, indices_meta, updates_meta, axis);
    }

    // Expect that all input tensors of mutable op are functional tensors
    TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self));
    TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(indices));
    TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(updates));

    // Sync and unwrap functional tensors
    at::functionalization::impl::sync(self);
    at::functionalization::impl::sync(indices);
    at::functionalization::impl::sync(updates);
    auto self_unwrap = at::functionalization::impl::from_functional_tensor(self);
    auto indices_unwrap = at::functionalization::impl::from_functional_tensor(indices);
    auto updates_unwrap = at::functionalization::impl::from_functional_tensor(updates);

    // Grab the dispatcher entry corresponding out-of-place variant of the mutable op
    static auto scatter_update_handle = c10::Dispatcher::singleton()
        // specify namespace::op_name, op_overload_name
        .findSchemaOrThrow("npu::scatter_update", "")
        // specify the C++ schema of the corresponding out-of-place op
        .typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t)>();

    // Redispath to the out-of-place op when mutable op is called by user
    at::Tensor tmp_output;
    {
        at::AutoDispatchSkipFunctionalize func_guard;
        tmp_output = scatter_update_handle.call(self_unwrap, indices_unwrap, updates_unwrap, axis);
    }

    // Finally, tell functionalization about the mutation
    at::functionalization::impl::replace_(self, tmp_output);
    at::functionalization::impl::commit_update(self);
    at::functionalization::impl::sync(self);
    return self;
}

// Regster functionalization for custom op npu_quant_scatter_.
// To Get these mutable operators to work with functionalization requires some extra work.
// We need to register a corresponding out-of-place variant of the op,
// and register a functionalization kernel that performs some boilerplate to
// teach functionalization how to map from the mutable op to the out-of-place op.
at::Tensor &npu_quant_scatter__functionalization(
    at::Tensor& self,
    const at::Tensor& indices,
    const at::Tensor& updates,
    const at::Tensor& quant_scales,
    const c10::optional<at::Tensor>& quant_zero_points,
    int64_t axis,
    int64_t quant_axis,
    c10::string_view reduce)
{
    {
        // Before converting the mutable op to its functional variant, run meta tensors through the original op.
        // This will help us catch shape errors that apply to inplace ops that wouldn't apply to their functional
        // variants. (We can only do this for inplace ops today though, because they technicaly all support meta
        // tensors).
        auto self_meta = to_meta(self);
        auto indices_meta = to_meta(indices);
        auto updates_meta = to_meta(updates);
        auto quant_scales_meta = to_meta(quant_scales);
        at::Tensor quant_zero_points_meta;
        if (quant_zero_points.has_value()) {
            quant_zero_points_meta = to_meta(quant_zero_points.value());
        }

        at::AutoDispatchSkipFunctionalize func_guard;
        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);
        static auto npu_quant_scatter__handle = c10::Dispatcher::singleton()
            // specify namespace::op_name, op_overload_name
            .findSchemaOrThrow("npu::npu_quant_scatter_", "")
            // specify the C++ schema of the original mutable op
            .typed<at::Tensor & (at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                 const c10::optional<at::Tensor> &, int64_t, int64_t, c10::string_view)>();

        npu_quant_scatter__handle.call(self_meta, indices_meta, updates_meta, quant_scales_meta, quant_zero_points_meta,
                                       axis, quant_axis, reduce);
    }

    // Expect that all input tensors of mutable op are functional tensors
    TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self));
    TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(indices));
    TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(updates));
    TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(quant_scales));
    if (quant_zero_points.has_value()) {
        TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(quant_zero_points.value()));
    }

    // Sync and unwrap functional tensors
    at::functionalization::impl::sync(self);
    at::functionalization::impl::sync(indices);
    at::functionalization::impl::sync(updates);
    at::functionalization::impl::sync(quant_scales);
    if (quant_zero_points.has_value()) {
        at::functionalization::impl::sync(quant_zero_points.value());
    }
    auto self_unwrap = at::functionalization::impl::from_functional_tensor(self);
    auto indices_unwrap = at::functionalization::impl::from_functional_tensor(indices);
    auto updates_unwrap = at::functionalization::impl::from_functional_tensor(updates);
    auto quant_scales_unwrap = at::functionalization::impl::from_functional_tensor(quant_scales);
    at::Tensor quant_zero_points_unwrap;
    if (quant_zero_points.has_value()) {
        quant_zero_points_unwrap = at::functionalization::impl::from_functional_tensor(quant_zero_points.value());
    }

    // Grab the dispatcher entry corresponding out-of-place variant of the mutable op
    static auto npu_quant_scatter_handle = c10::Dispatcher::singleton()
        // specify namespace::op_name, op_overload_name
        .findSchemaOrThrow("npu::npu_quant_scatter", "")
        // specify the C++ schema of the corresponding out-of-place op
        .typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                           const c10::optional<at::Tensor> &, int64_t, int64_t, c10::string_view)>();

    // Redispath to the out-of-place op when mutable op is called by user
    at::Tensor tmp_output;
    {
        at::AutoDispatchSkipFunctionalize func_guard;
        tmp_output = npu_quant_scatter_handle.call(self_unwrap, indices_unwrap, updates_unwrap, quant_scales_unwrap,
                                                   quant_zero_points_unwrap, axis, quant_axis, reduce);
    }

    // Finally, tell functionalization about the mutation
    at::functionalization::impl::replace_(self, tmp_output);
    at::functionalization::impl::commit_update(self);
    at::functionalization::impl::sync(self);
    return self;
}

TORCH_LIBRARY_IMPL(npu, Functionalize, m)
{
    m.impl("scatter_update_", scatter_update__functionalization);
}

TORCH_LIBRARY_IMPL(npu, Functionalize, m)
{
    m.impl("npu_quant_scatter_", npu_quant_scatter__functionalization);
}

}  // namespace native
}  // namespace at_npu
