#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu{
namespace native{

at::Tensor& sign_bits_unpack_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& x,
    int64_t size,
    c10::ScalarType dtype) {
  int64_t typeEnum = dtype == at::ScalarType::Half ? 1 : 0;
  OpCommand cmd;
  cmd.Name("SignBitsUnpack")
     .Input(x)
     .Output(result)
     .Attr("dtype", typeEnum)
     .Attr("size", size)
     .Run();
  return result;
}

at::Tensor NPUNativeFunctions::npu_sign_bits_unpack(
    const at::Tensor& input,
    int64_t size,
    c10::ScalarType dtype) {
    int64_t dim = input.dim();
    TORCH_CHECK(dim == 1, "input value should be a 1-dimensional tensor");
   
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Byte, "sign_bits_unpack input only supports torch.uint8 ");
    
    TORCH_CHECK(size > 0, "The argument 'size' is not valid because it is less than or equal to zero");

    int64_t input_size = input.numel(); 
    TORCH_CHECK((input_size * 8) % size == 0, "input value length*8 must be multiple of size");

    TORCH_CHECK(dtype == at::ScalarType::Float || dtype == at::ScalarType::Half, "The argument 'dtype'  must be torch.float32 or torch.float16");

    int64_t m = input_size * 8 / size;   
    at::Tensor result = OpPreparation::ApplyTensor({size, m}, input.options().dtype(dtype), input);

    sign_bits_unpack_out_npu_nocheck(result, input, size, dtype);
    return result;
  }
} // namespace native
} // namespace at_npu