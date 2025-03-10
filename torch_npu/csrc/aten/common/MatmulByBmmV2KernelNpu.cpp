#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"

namespace at_npu {
namespace native {

at::Tensor matmul_by_bmmV2(const at::Tensor& tensor1, const at::Tensor& tensor2)
{
    auto dim_tensor1 = tensor1.dim();
    auto dim_tensor2 = tensor2.dim();
    if (dim_tensor1 == 1 && dim_tensor2 == 1) {
        return tensor1.dot(tensor2);
    } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
        return tensor1.mm(tensor2.unsqueeze(-1)).squeeze_(-1);
    } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
        return tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
    } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
        return tensor1.mm(tensor2);
    } else if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
        at::Tensor t2 = dim_tensor2 == 1 ? tensor2.unsqueeze(-1) : tensor2;
        auto size1 = tensor1.sizes();
        auto size2 = t2.sizes();
        std::vector<int64_t> output_size;
        output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
        if (dim_tensor2 > 1) {
            output_size.push_back(size2[dim_tensor2 - 1]);
        }
        // fold the batch into the first dimension
        at::Tensor t1 = tensor1.reshape({-1, tensor1.size(-1)});
        at::Tensor output = at::_unsafe_view(t1.mm(t2), output_size);
        return output;
    } else if ((dim_tensor1 == 1 || dim_tensor1 == 2) && dim_tensor2 >= 3) {
        return custom_ops::npu_bmmV2(tensor1, tensor2, {});
    } else if ((dim_tensor1 >= 1 && dim_tensor2 >= 1) && (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
        return custom_ops::npu_bmmV2(tensor1, tensor2, {});
    }
    AT_ERROR("both arguments to matmul need to be at least 1D, but they are ", dim_tensor1, "D and ", dim_tensor2, "D");
}

}
}
