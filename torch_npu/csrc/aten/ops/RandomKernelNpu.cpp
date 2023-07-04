#include <limits.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace at_npu {
namespace native {

namespace {

// RANDOM_DOUBLE_MAX = 1 << 53
const int64_t RANDOM_DOUBLE_MAX = 9007199254740992;
const int64_t RANDOM_HALF_MAX = 1 << 11;
const int64_t RANDOM_FLOAT_MAX = 1 << 24;

}

at::Tensor& random_out_npu(
    at::Tensor& result,
    at::Tensor& self,
    int64_t from,
    int64_t to,
    c10::optional<at::Generator> gen_) {
  auto gen = at::get_generator_or_default<NPUGeneratorImpl>(gen_, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  at::SmallVector<int64_t, N> key = {seed}; 
  at::SmallVector<int64_t, N> counter = {0, offset}; 
  const int32_t alg = 1;
  OpCommand cmd;
  cmd.Name("StatelessRandomUniformV2")
      .Input(self.sizes(), at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Input(key, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
      .Input(counter, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
      .Input(at::Scalar(alg), at::ScalarType::Int);
  // StatelessRandomUniformV2 doesn't support int output
  if (isIntegralType(self.scalar_type(), true)) {
    auto result_cp = OpPreparation::ApplyTensor(self, self.options().dtype(at::kFloat));
    cmd.Attr("dtype", at::kFloat)
        .Output(result_cp)
        .Run();
    // StatelessRandomUniformV2 output: U(0~1) --> U(from~to)
    result_cp = result_cp.mul(to).sub(result_cp.mul(from).sub(static_cast<float>(from)));
    result_cp = NPUNativeFunctions::npu_dtype_cast(result_cp, self.scalar_type());
    result.copy_(result_cp);
  } else {
    cmd.Attr("dtype", self.scalar_type())
        .Output(result)
        .Run();
    // StatelessRandomUniformV2 output: U(0~1) --> U(from~to)
    auto result_cp = result.mul(to).sub(result.mul(from).sub(static_cast<float>(from)));
    // round off numbers
    result_cp = NPUNativeFunctions::npu_dtype_cast(result_cp, at::kLong);
    result_cp = NPUNativeFunctions::npu_dtype_cast(result_cp, self.scalar_type());
    result.copy_(result_cp);
  }
  return result;
}

at::Tensor& random_npu_(at::Tensor& self, int64_t from, int64_t to, c10::optional<at::Generator> gen_) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    random_out_npu(contiguousSelf, contiguousSelf, from, to, gen_);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    random_out_npu(self, self, from, to, gen_);
  }
  return self;
}

int64_t get_max(const auto dtype) {
  if (dtype == at::kHalf) {return RANDOM_HALF_MAX + 1;}
  if (dtype == at::kFloat) {return RANDOM_FLOAT_MAX + 1;}
  if (dtype == at::kDouble) {return RANDOM_DOUBLE_MAX + 1;}
  if (dtype == at::kInt) {return INT_MAX;}
  if (dtype == at::kShort) {return SHRT_MAX + 1;}
  if (dtype == at::kChar) {return SCHAR_MAX + 1;}
  if (dtype == at::kByte) {return UCHAR_MAX + 1;}
  if (dtype == at::kLong) {return LONG_MAX;}
  return 1;
}

at::Tensor& NPUNativeFunctions::random_(
    at::Tensor& self, int64_t from,
    c10::optional<int64_t> to,
    c10::optional<at::Generator> gen_) {
  int64_t to_ = to.has_value() ? to.value() : get_max(self.dtype());
  random_npu_(self, from, to_, gen_);
  return self;
}

at::Tensor& NPUNativeFunctions::random_(at::Tensor& self, int64_t to, c10::optional<at::Generator> gen_) {
  random_npu_(self, 0, to, gen_);
  return self;
}

at::Tensor& NPUNativeFunctions::random_(at::Tensor& self, c10::optional<at::Generator> gen_) {
  random_npu_(self, 0, get_max(self.dtype()), gen_);

  return self;
}
} // namespace native
} // namespace at_npu
