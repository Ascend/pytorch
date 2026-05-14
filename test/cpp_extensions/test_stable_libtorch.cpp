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

std::tuple<Tensor, Tensor> my_cummax(Tensor self, int64_t dim)
{
    AtenTensorHandle ret0;
    AtenTensorHandle ret1;
    aoti_torch_npu_cummax(self.get(), dim, &ret0, &ret1);
    return std::make_tuple(Tensor(ret0), Tensor(ret1));
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_211, m) {
    m.def("my_abs(Tensor self) -> Tensor");
    m.def("my_cummax(Tensor self, int dim) -> (Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_211, CompositeExplicitAutograd, m) {
    m.impl("my_abs", TORCH_BOX(&my_abs));
    m.impl("my_cummax", TORCH_BOX(&my_cummax));
}