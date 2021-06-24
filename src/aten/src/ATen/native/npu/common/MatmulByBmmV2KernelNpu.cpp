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
    return tensor1.mv(tensor2);
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    return tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    return tensor1.mm(tensor2);
  } else if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
    Tensor t2 = dim_tensor2 == 1 ? tensor2.unsqueeze(-1) : tensor2;
    Tensor output = dim_tensor2 == 1 ? at::npu_bmmV2(tensor1, t2).squeeze(-1) : at::npu_bmmV2(tensor1, t2);
    return output;
    // 需要支持out接口
  } else if ((dim_tensor1 == 1 || dim_tensor1 == 2) && dim_tensor2 >= 3) {
    // optimization: transpose the inner dimensions of the arguments, call
    // matmul on the swapped arguments, then transpose the inner dimensions
    // of the result.
    const int64_t n = dim_tensor1 == 2 ? tensor1.size(-2) : 1;
    const int64_t m = tensor1.size(-1);
    const int64_t p = tensor2.size(-1);

    const Tensor t2_T = tensor2.transpose(-1, -2);
    const Tensor t1_T = dim_tensor1 == 2 ? tensor1.t() : tensor1.reshape({n, m}).t();
    const Tensor res_T = matmul_npu(t2_T, t1_T);

    if (dim_tensor1 == 2) {
      Tensor res = res_T.transpose(-1, -2).contiguous();
      return res;
    }
    else {
      std::vector<int64_t> shape = tensor2.sizes().slice(0, dim_tensor2 - 2).vec();
      shape.push_back(p);

      Tensor res = res_T.reshape(shape).contiguous();
      return res;
    }
  } else if ((dim_tensor1 >= 1 && dim_tensor2 >= 1) && (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
      int64_t n = dim_tensor1 > 1 ? tensor1.size(-2) : 1;
    int64_t m1 = tensor1.size(-1);
    IntArrayRef batch_tensor1(tensor1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0));
    int64_t m2 = dim_tensor2 > 1 ? tensor2.size(-2) : 1;
    int64_t p = tensor2.size(-1);
    IntArrayRef batch_tensor2(tensor2.sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0));

    // expand the batch portion (i.e. cut off matrix dimensions and expand rest)
    std::vector<int64_t> expand_batch_portion = infer_size(batch_tensor1, batch_tensor2);

    std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
    tensor1_expand_size.insert(tensor1_expand_size.end(), {n, m1});

    std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
    tensor2_expand_size.insert(tensor2_expand_size.end(), {m2, p});


    // flatten expanded batches
    Tensor tensor1_expanded = tensor1.expand(tensor1_expand_size);
    Tensor tensor2_expanded = tensor2.expand(tensor2_expand_size);

    Tensor output = at::npu_bmmV2(tensor1_expanded, tensor2_expanded);
    return output;
  }
  AT_ERROR("both arguments to matmul need to be at least 1D, but they are ",
          dim_tensor1, "D and ", dim_tensor2, "D");
}

}
}