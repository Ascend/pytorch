#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include "torch_npu/csrc/inductor/aoti_torch/generated/c_shim_npu.h"

using torch::stable::Tensor;

Tensor my_abs(Tensor self)
{
    AtenTensorHandle ret;
    aoti_torch_npu_abs(self.get(), &ret);
    return Tensor(ret);
}

void boxed_my_abs(StableIValue* stack, uint64_t num_args, uint64_t num_outs)
{
    Tensor res = my_abs(to<Tensor>(stack[0]));
    stack[0] = from(res);
}

std::tuple<Tensor, Tensor> my_cummax(Tensor self, int64_t dim)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    aoti_torch_npu_cummax(self.get(), dim, &ret0, &ret1);
    return std::make_tuple(Tensor(ret0), Tensor(ret1));
}

void boxed_my_cummax(StableIValue* stack, uint64_t num_args, uint64_t num_outs)
{
    auto tuple = my_cummax(to<Tensor>(stack[0]), to<int64_t>(stack[1]));
    stack[0] = from(std::get<0>(tuple));
    stack[1] = from(std::get<1>(tuple));
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_211, m) {
    m.def("my_abs(Tensor self) -> Tensor");
    m.def("my_cummax(Tensor self, int dim) -> (Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_211, CompositeExplicitAutograd, m) {
    m.impl("my_abs", &boxed_my_abs);
    m.impl("my_cummax", &boxed_my_cummax);
}