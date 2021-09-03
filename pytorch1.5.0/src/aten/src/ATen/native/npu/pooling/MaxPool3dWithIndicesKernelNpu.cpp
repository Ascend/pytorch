// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION. 
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include <ATen/native/Pool.h>

namespace at {
namespace native {
using namespace at::native::npu;

std::tuple<Tensor&, Tensor&> max_pool3d_with_indices_out_npu(
  Tensor& result,
  Tensor& indice,
  const Tensor& self,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef pads,
  IntArrayRef dilation,
  bool ceil_mode)
{
  int64_t strideT = 1;
  int64_t strideH = 1;
  int64_t strideW = 1;
  if (stride.empty()) {
    strideT = kernel_size[0];
    strideH = kernel_size[1];
    strideW = kernel_size[2];
  } else {
    strideT = stride[0];
    strideH = stride[1];
    strideW = stride[2];
  }

  string padding = "CALCULATED";
  int64_t ds = self.size(-3);
  int64_t hs = self.size(-2);
  int64_t ws = self.size(-1);
  SmallVector<int64_t, SIZE> padrs(pads);
  if(ceil_mode){
    padrs[0] += CalcuOpUtil::completePad(ds, pads[0], kernel_size[0], strideT);
    padrs[1] += CalcuOpUtil::completePad(hs, pads[1], kernel_size[1], strideH);
    padrs[2] += CalcuOpUtil::completePad(ws, pads[2], kernel_size[2], strideW);
  }
  SmallVector<int64_t, SIZE> kernel_sizes = {1, 1, kernel_size[0], kernel_size[1], kernel_size[2]};
  SmallVector<int64_t, SIZE> stride_sizes = {1, 1, strideT, strideH, strideW};
  SmallVector<int64_t, SIZE> pads_sizes = {pads[0], padrs[0], pads[1], padrs[1], pads[2], padrs[2]};
  SmallVector<int64_t, SIZE> dilation_sizes = {1, 1, dilation[0], dilation[1], dilation[2]};
  string data_format = "NCDHW";

  OpCommand cmd;
  cmd.Name("MaxPool3D")
     .Input(self)
     .Output(result)
     .Attr("ksize", kernel_sizes)
     .Attr("strides", stride_sizes)
     .Attr("padding", padding)
     .Attr("pads", pads_sizes)
     .Attr("dilation", dilation_sizes)
     .Attr("ceil_mode", (int64_t) ceil_mode)
     .Attr("data_format", data_format)
     .Run();
  return std::tie(result, result);
}

std::tuple<Tensor, Tensor> max_pool3d_with_indices_npu(
  const Tensor& self,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef pads,
  IntArrayRef dilation,
  bool ceil_mode)
{
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
    "max_pool3d: kernel_size must either be a single int, or a tuple of three ints")
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);
  SmallVector<int64_t, SIZE> kernel_sizes = {kT, kH, kW};
  IntArrayRef kernel_sizess = IntArrayRef(kernel_sizes);

  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
    "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints")
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);
  SmallVector<int64_t, SIZE> strides = {dT, dH, dW};
  IntArrayRef stridess = IntArrayRef(strides);

  TORCH_CHECK(pads.size() == 1 || pads.size() == 3,
    "max_pool3d: padding must be either be a single int, or a tuple of three ints");
  const int pT = safe_downcast<int, int64_t>(pads[0]);
  const int pH = pads.size() == 1 ? pT : safe_downcast<int, int64_t>(pads[1]);
  const int pW = pads.size() == 1 ? pT : safe_downcast<int, int64_t>(pads[2]);
  SmallVector<int64_t, SIZE> paddings = {pT, pH, pW};
  IntArrayRef padss = IntArrayRef(paddings);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 3,
    "max_pool3d: dilation must be either a single int, or a tuple of three ints");
  const int dilationT = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationH = dilation.size() == 1 ? dilationT : safe_downcast<int, int64_t>(dilation[1]);
  const int dilationW = dilation.size() == 1 ? dilationT : safe_downcast<int, int64_t>(dilation[2]);
  SmallVector<int64_t, SIZE> dilations = {dilationT, dilationH, dilationW};
  IntArrayRef dilationss = IntArrayRef(dilations);

  TORCH_CHECK((self.ndimension() == 5 || self.ndimension() == 4),
        "maxpool3d expected input to be non-empty 5D(batch mode) or 4D tensor",
        "but input has dim: ",
        self.ndimension());

  const int64_t nslices = self.size(-4);
  const int64_t itime = self.size(-3);
  const int64_t iheight = self.size(-2);
  const int64_t iwidth = self.size(-1);

  const int64_t otime = pooling_output_shape<int64_t>(itime, kT, pT, dT, dilationT, ceil_mode);
  const int64_t oheight = pooling_output_shape<int64_t>(iheight, kH, pH, dH, dilationH, ceil_mode);
  const int64_t owidth = pooling_output_shape<int64_t>(iwidth, kW, pW, dW, dilationW, ceil_mode);

  pool3d_shape_check(
    self,
    nslices,
    kT, kH, kW,
    dT, dH, dW,
    pT, pH, pW,
    dilationT, dilationH, dilationW,
    itime, iheight, iwidth,
    otime, oheight, owidth);
  Tensor selfCp = self.ndimension() == 4 ? self.unsqueeze(0) : self;
  SmallVector<int64_t, SIZE> outputSize = {selfCp.size(0), selfCp.size(1), otime, oheight, owidth};
  Tensor result = at::empty_with_format(
    outputSize, selfCp.options(), ACL_FORMAT_NCDHW);
  
  max_pool3d_with_indices_out_npu(result, result, selfCp, kernel_sizess, stridess, padss, dilationss, ceil_mode);
  result = self.ndimension() == 4 ? result.squeeze(0) : result;
  return std::tie(result, result);
}

} // native
} // at