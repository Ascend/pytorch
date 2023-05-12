#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

namespace {
  constexpr int64_t ratio = 1;
  constexpr bool fancyUpscaling = true;
  constexpr float acceptableFraction = 1.0;
  const std::string dctMethod = "";
  const std::string dstImgFormat = "CHW";
}

at::Tensor &decode_jpeg_out(
    const at::Tensor &self,
    int64_t channels,
    bool try_recover_truncated,
    at::Tensor &result)
{
  OpCommand cmd;
  cmd.Name("DecodeJpeg")
      .Input(self, "", c10::nullopt, "string")
      .Output(result)
      .Attr("channels", channels)
      .Attr("ratio", ratio)
      .Attr("fancy_upscaling", fancyUpscaling)
      .Attr("try_recover_truncated", try_recover_truncated)
      .Attr("acceptable_fraction", acceptableFraction)
      .Attr("dct_method", dctMethod)
      .Attr("dst_img_format", dstImgFormat)
      .Run();

  return result;
}

at::Tensor NPUNativeFunctions::decode_jpeg(
    const at::Tensor &self,
    at::IntArrayRef image_shape,
    int64_t channels,
    bool try_recover_truncated)
{
  // calculate the output size
  auto outputSize = decode_jpeg_npu_output_size(image_shape, channels);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      self.options().dtype(at::kByte),
      ACL_FORMAT_ND);

  // calculate the output result of the NPU
  decode_jpeg_out(self, channels, try_recover_truncated, result);

  return result;
}

} // namespace native
} // namespace at_npu