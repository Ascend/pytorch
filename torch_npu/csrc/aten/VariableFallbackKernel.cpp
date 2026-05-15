#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/native/CPUFallback.h>
#include <torch/library.h>
#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>

#include "torch_npu/csrc/core/npu/NPUException.h"

/*
 * This file implements a variable fallback kernel for custom operators.
 * Since tensors always have the Autograd set, but custom operators
 * usually don't have a kernel registered for Autograd, the dispatcher
 * will call into this fallback kernel instead.
 * Note that this is not a correct autograd implementation. It will just
 * fallthrough to the custom operator implementation.
 * If you want a custom operator to work with autograd, you need to use
 * autograd::Function so that the custom operator implementation knows how to
 * do autograd.
 * Note also that ops from native_functions.yaml register their own variable
 * kernels, so this is never called for them.
 */

using c10::OperatorHandle;
using c10::Stack;
using c10::DispatchKey;
using c10::DispatchKeySet;
using c10::Dispatcher;
using c10::KernelFunction;
using torch::autograd::edge_list;
using torch::autograd::Node;
using torch::autograd::variable_list;

namespace {

template <typename F>
void _foreach_tensor(
    F fn,
    torch::jit::Stack* stack,
    size_t stack_start,
    size_t size)
{
    // Enumerate over tensors in a stack, including ones in TensorLists
    int idx_tensor = 0;
    for (const auto idx_arg : c10::irange(size)) {
        auto& ivalue = (*stack)[stack_start + idx_arg];
        if (ivalue.isTensor()) { // true for optional tensor that has value
            const auto& tensor = ivalue.toTensor();
            fn(idx_tensor, idx_arg, tensor);
            idx_tensor++;
        } else if (ivalue.isTensorList()) {
            for (const auto& iv : ivalue.toListRef()) {
                const auto& tensor = iv.toTensor();
                fn(idx_tensor, idx_arg, tensor);
                idx_tensor++;
            }
        }
    }
}


static void warnAutogradNotImplemented(const std::string& op_name)
{
    TORCH_NPU_WARN_ONCE(
        op_name,
        ": an autograd kernel was not registered to the Autograd key(s) ",
        "but we are trying to backprop through it. This may lead to silently incorrect behavior. ",
        "This behavior is deprecated and will be removed in a future version of PyTorch. ",
        "If your operator is differentiable, please ensure you have registered an "
        "autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, "
        "DispatchKey::CompositeImplicitAutograd). If your operator is not "
        "differentiable, or to squash this warning and use the previous behavior, "
        "please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd.");
}


struct WarnNotImplemented : public Node {
    WarnNotImplemented(
        std::string op_name,
        int64_t num_outputs,
        edge_list&& next_edges)
        : Node(std::move(next_edges)), op_name(std::move(op_name)), num_outputs(num_outputs) {}

    WarnNotImplemented(std::string op_name, int64_t num_outputs)
        : op_name(std::move(op_name)), num_outputs(num_outputs) {}

    variable_list apply(variable_list&& inputs) override;

    std::string op_name;
    int64_t num_outputs;
};

auto WarnNotImplemented::apply(variable_list&& inputs) -> variable_list
{
    warnAutogradNotImplemented(op_name);
    std::vector<at::Tensor> output(num_outputs);
    return output;
}

static void npuBasicAutogradNotImplementedFallbackImpl(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatch_keys,
    torch::jit::Stack* stack)
{
    const auto& schema = op.schema();
    const auto& op_name = schema.operator_name().name;
    const auto num_arguments = schema.arguments().size();
    const auto num_returns = schema.returns().size();
    const auto stack_start = stack->size() - num_arguments;

    if (torch::autograd::getAutogradFallbackMode() == torch::autograd::AutogradFallbackMode::Nothing) {
        op.redispatchBoxed(dispatch_keys & c10::after_autograd_keyset, stack);
        return;
    }
    TORCH_INTERNAL_ASSERT(
        torch::autograd::getAutogradFallbackMode() == torch::autograd::AutogradFallbackMode::Warn);

    bool any_input_requires_grad = false;
    _foreach_tensor(
        [&](size_t _, size_t idx_arg, const at::Tensor& t) {
            if (t.requires_grad()) {
            any_input_requires_grad = true;
            }
        },
        stack,
        stack_start,
        num_arguments);
    // Optimization: TLS access can be slow. So we only check if it necessary
    // by putting it after the requires_grad checks.
    any_input_requires_grad = any_input_requires_grad && at::GradMode::is_enabled();

    std::shared_ptr<WarnNotImplemented> grad_fn;
    if (any_input_requires_grad) {
        // NB: It is standard to collect edges from all tensors
        // (see generated/VariableTypeEverything.cpp for examples)
        std::vector<const at::Tensor*> all_tensors_on_stack;
        _foreach_tensor(
            [&](size_t _, size_t idx_arg, const at::Tensor& t) {
            all_tensors_on_stack.push_back(&t);
            },
            stack,
            stack_start,
            num_arguments);
        grad_fn = std::shared_ptr<WarnNotImplemented>(
            new WarnNotImplemented(op_name, all_tensors_on_stack.size()),
            torch::autograd::deleteNode);
        grad_fn->set_next_edges(torch::autograd::collect_next_edges(all_tensors_on_stack));
    }

    op.redispatchBoxed(dispatch_keys & c10::after_autograd_keyset, stack);

    if (any_input_requires_grad) {
        // NB: if the operator mutates any inputs in-place and does not return them
        // as outputs, we are unable to lazily raise a warning. This is OK because
        // we don't expect many existing operators to do this because of the amount
        // of technical expertise necessary (you would need to manually register an
        // autograd kernel without using autograd.Function)
        _foreach_tensor(
            [&](size_t _, size_t idx_ret, const at::Tensor& t) {
            if (!torch::autograd::isDifferentiableType(t.scalar_type())) {
                return;
            }
            const bool is_mutable_output =
                schema.is_aliasing({c10::SchemaArgType::output, idx_ret}) &&
                schema.is_mutable({c10::SchemaArgType::output, idx_ret});

            // If the post-autograd implementation returns Tensors that require
            // grad, then we install a hook that will warn during the backwards.
            //
            // NB: If the operation is inplace and the inputs were views,
            // it is possible that the history was rebased and the hook will
            // not warn in all places where it should. That is, the following
            // won't warn:
            // >>> x = torch.randn(3, 3, requires_grad=True)
            // >>> z = x.clone()
            // >>> w = z[0]
            // >>> k = w[0]
            // >>> y = op(k)
            // >>> torch.autograd.grad(z.sum(), w)
            if (t.requires_grad()) {
                t.register_hook([op_name](const at::Tensor& grad) {
                warnAutogradNotImplemented(op_name);
                });
                // If history is rebased, then we will attempt to warn
                // on the view's base. This will catch most cases (because
                // users typically call .backward() and backprop through
                // the entire program).
                if (t.is_view() && is_mutable_output) {
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                auto& base = const_cast<at::TensorBase&>(t._base());
                if (base.requires_grad()) {
                    // Can only register_hook on tensors that require grad.
                    base.register_hook([op_name](const at::TensorBase& grad) {
                    warnAutogradNotImplemented(op_name);
                    });
                }
                }
                return;
            }

            // If the post-autograd implementation returns any Tensors that
            // don't require grad, then we install the WarnNotImplemented grad_fn.
            // This grad_fn warns in backward and returns undefined tensor
            // gradients.
            //
            // NOTE [autograd fallback and in-place operations]
            // If the schema says the output is mutable, and the output
            // is an input, and the input is a view Tensor, then...
            // we're not sure if set_history is OK to do, so we just skip
            // adding the grad_fn. Builtin operators do rebase_history here,
            // but custom operators may have multiple Tensor(a!) returns,
            // rebase_history assumes single Tensor(a!) return, and in general
            // custom ops don't have a good in-place story.
            if (!is_mutable_output) {
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                torch::autograd::set_history(const_cast<at::Tensor&>(t), grad_fn);
            }
            },
            stack,
            stack->size() - num_returns,
            num_returns);
    }
}

// Register fallthrough for Autograd backends dispatch keys
// NB: But not the private use ones; maybe the extension wants
// to override it themselves!

// (Ascend) TORCH_LIBRARY_IMPL
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&npuBasicAutogradNotImplementedFallbackImpl>());
}

bool has_op_name_warned(const std::string& op_name)
{
    static std::unordered_set<std::string> _op_lists = {};
    if (_op_lists.find(op_name) != _op_lists.end()) {
        return true;
    }
    _op_lists.insert(op_name);
    return false;
}

void npu_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack)
{
    if (!has_op_name_warned(c10::toString(op.schema().operator_name()))) {
        TORCH_NPU_WARN("CAUTION: The operator '",
                       op.schema().operator_name(),
                       "' is not currently supported ",
                       "on the NPU backend and will fall back to run on the CPU.",
                       " This may have performance implications.");
    }
    at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&npu_cpu_fallback>());
}

void npu_Sparse_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack)
{
    TORCH_CHECK(false, "CAUTION: The operator '",
                op.schema().operator_name(),
                "' is not currently supported on the NPU backend.", OPS_ERROR(ErrCode::NOT_SUPPORT))
}

TORCH_LIBRARY_IMPL(_, SparsePrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&npu_Sparse_fallback>());
}
}
