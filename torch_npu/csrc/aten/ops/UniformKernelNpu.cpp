#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"

namespace at_npu {
namespace native {

at::Tensor& uniform_out_npu(
    const at::Tensor& self,
    double from,
    double to,
    c10::optional<at::Generator> gen_,
    at::Tensor& result) {
  auto gen = at::get_generator_or_default<NPUGeneratorImpl>(gen_, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  c10::SmallVector<int64_t, N> seed_list = {seed};
  c10::SmallVector<int64_t, N> offset_list = {0, offset};
  int64_t alg = 1;
  OpCommand cmd;
  cmd.Name("StatelessRandomUniformV2")
      .Input(self.sizes(), at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Input(seed_list, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
      .Input(offset_list, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
      .Input(at::Scalar(alg), at::ScalarType::Int)
      .Output(result)
      .Attr("dtype", self.scalar_type())
      .Run();
  // StatelessRandomUniformV2 output: U(0~1) --> U(from~to)
  auto tmp = result.mul(from).sub(from);
  result.mul_(to).sub_(tmp);
  return result;
}

at::Tensor& NPUNativeFunctions::uniform_(at::Tensor& self, double from, double to, c10::optional<at::Generator> gen_) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor selfContiguous = NpuUtils::format_contiguous(self);
    uniform_out_npu(selfContiguous, from, to, gen_, selfContiguous);
    NpuUtils::format_fresh_view(self, selfContiguous);
  } else {
    uniform_out_npu(self, from, to, gen_, self);
  }
  return self;
}

} // namespace native
} // namespace at_npu
