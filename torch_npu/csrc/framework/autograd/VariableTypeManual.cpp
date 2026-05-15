#include <c10/util/Optional.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/autograd.h>
#include <ATen/TracerMode.h>
#include <ATen/RedispatchFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#include "torch_npu/csrc/framework/autograd/FunctionsManual.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

using namespace at;
using namespace at_npu::autograd::generated;
using torch::autograd::CreationMeta;
using torch::autograd::as_view;

namespace at_npu {
namespace autograd {
namespace VariableType {

std::vector<at::DeprecatedTypeProperties*> allTypesForBackends(at::ArrayRef<at::Backend> backends)
{
    std::vector<DeprecatedTypeProperties*> res;
    res.reserve(backends.size());
    for (auto p : backends) {
        for (const auto s : c10::irange(static_cast<int64_t>(ScalarType::NumOptions))) {
            auto& type = getDeprecatedTypeProperties(static_cast<Backend>(p), static_cast<ScalarType>(s));
            res.emplace_back(&type);
        }
    }
    return res;
}

C10_EXPORT std::vector<at::DeprecatedTypeProperties*> allCPUTypes()
{
    return allTypesForBackends({ Backend::CPU, Backend::SparseCPU });
}

namespace {
const Variable& checked_cast_variable(const Tensor& t, const char* name, int pos)
{
    if (!t.defined()) {
        AT_ERROR("Expected a proper Tensor but got None (or an undefined Tensor in C++) ",
                 "for argument #", pos, " '", name, "'");
    }
    return t;
}

Variable& checked_cast_variable(Tensor& t, const char* name, int pos)
{
    if (!t.defined()) {
        AT_ERROR("Expected a proper Tensor but got None (or an undefined Tensor in C++) ",
                 "for argument #", pos, " '", name, "'");
    }
    return t;
}
} // namespace

const Tensor& unpack(const Tensor& t, const char* name, int pos)
{
    return checked_cast_variable(t, name, pos);
}

Tensor& unpack(Tensor& t, const char* name, int pos)
{
    return checked_cast_variable(t, name, pos);
}

Tensor unpack_opt(const Tensor& t, const char* name, int pos)
{
    if (!t.defined()) {
        return Tensor();
    }
    return unpack(t, name, pos);
}

std::vector<at::Tensor> unpack(at::TensorList tl, const char* name, int pos)
{
    std::vector<at::Tensor> ret(tl.size());
    for (const auto i : c10::irange(tl.size())) {
        const auto &t = tl[i];
        if (!t.defined()) {
            continue;
        }
        ret[i] = static_cast<const Variable&>(t);
    }
    (void) name;
    (void) pos;
    return ret;
}

namespace {

// Taken from codegened version
Tensor _fw_primal(c10::DispatchKeySet ks, const Tensor& self, int64_t level)
{
    auto& self_ = unpack(self, "self", 0);
    std::shared_ptr<Identity> grad_fn;
    if (compute_requires_grad(self)) {
        grad_fn = std::make_shared<Identity>();
        grad_fn->set_next_edges(collect_next_edges(self));
    }

    auto result = ([&]() {
        at::AutoDispatchBelowAutograd guard;
        return at::redispatch::_fw_primal(ks& c10::after_autograd_keyset, self_, level);
    })();

    if (grad_fn) {
        set_history(flatten_tensor_args(result), grad_fn);
    }
    if (isFwGradDefined(self)) {
        // Modified from original codegen
        // We explicitly want to ignore the forward grad at the given level
        TORCH_CHECK(level == 0, "Invalid level given to _fw_primal", OPS_ERROR(ErrCode::VALUE));
        // End modified from original codegen
    }
    return result;
}
} // namespace

} // namespace VariableType
} // namespace autograd
} // namespace at_npu
