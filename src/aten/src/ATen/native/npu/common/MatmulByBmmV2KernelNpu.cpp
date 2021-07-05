#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;
    Tensor matmul_by_bmmV2(
    const Tensor& tensor1,
    const Tensor& tensor2) {
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
    Tensor t2 = dim_tensor2 == 1 ? tensor2.unsqueeze(-1) : tensor2;
    auto size1 = tensor1.sizes();
    auto size2 = t2.sizes();
    std::vector<int64_t> output_size;
    output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
    if (dim_tensor2 > 1) {
      output_size.push_back(size2[dim_tensor2 - 1]);
    }
    // fold the batch into the first dimension
    Tensor t1 = tensor1.reshape({-1, tensor1.size(-1)});
    Tensor output = at::_unsafe_view(t1.mm(t2), output_size);
    return output;
  } else if ((dim_tensor1 == 1 || dim_tensor1 == 2) && dim_tensor2 >= 3) {
    return at::npu_bmmV2(tensor1, tensor2, {});
  } else if ((dim_tensor1 >= 1 && dim_tensor2 >= 1) && (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
    return at::npu_bmmV2(tensor1, tensor2, {});
  }
  AT_ERROR("both arguments to matmul need to be at least 1D, but they are ",
          dim_tensor1, "D and ", dim_tensor2, "D");
}

}
}