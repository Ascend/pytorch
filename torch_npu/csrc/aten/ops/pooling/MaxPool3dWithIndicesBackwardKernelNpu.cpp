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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include <ATen/native/Pool.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    at::Tensor &NPUNativeFunctions::max_pool3d_with_indices_backward_out(
        const at::Tensor &grad_output,
        const at::Tensor &self,
        at::IntArrayRef kernel_size,
        at::IntArrayRef stride,
        at::IntArrayRef padding,
        at::IntArrayRef dilation,
        bool ceil_mode,
        const at::Tensor &indices,
        at::Tensor &grad_input)
    {
      int64_t strideT = 1;
      int64_t strideH = 1;
      int64_t strideW = 1;
      if (stride.empty())
      {
        strideT = kernel_size[0];
        strideH = kernel_size[1];
        strideW = kernel_size[2];
      }
      else
      {
        strideT = stride[0];
        strideH = stride[1];
        strideW = stride[2];
      }

      string padstr = "CALCULATED";
      int64_t ds = self.size(-3);
      int64_t hs = self.size(-2);
      int64_t ws = self.size(-1);
      c10::SmallVector<int64_t, SIZE> padrs(padding);
      if (ceil_mode)
      {
        padrs[0] += CalcuOpUtil::completePad(ds, padding[0], kernel_size[0], strideT);
        padrs[1] += CalcuOpUtil::completePad(hs, padding[1], kernel_size[1], strideH);
        padrs[2] += CalcuOpUtil::completePad(ws, padding[2], kernel_size[2], strideW);
      }
      c10::SmallVector<int64_t, SIZE> kernel_sizes = {1, 1, kernel_size[0], kernel_size[1], kernel_size[2]};
      c10::SmallVector<int64_t, SIZE> stride_sizes = {1, 1, strideT, strideH, strideW};
      c10::SmallVector<int64_t, SIZE> padding_sizes = {padding[0], padrs[0], padding[1], padrs[1], padding[2], padrs[2]};

      string data_format = "NCDHW";

      OpCommand cmd;
      cmd.Name("MaxPool3DGrad")
          .Input(self, "orig_x", ACL_FORMAT_NCDHW)
          .Input(indices, "orig_y", ACL_FORMAT_NCDHW)
          .Input(grad_output, "grads", ACL_FORMAT_NCDHW)
          .Output(grad_input, "y", ACL_FORMAT_NCDHW)
          .Attr("ksize", kernel_sizes)
          .Attr("strides", stride_sizes)
          .Attr("padding", padstr)
          .Attr("pads", padding_sizes)
          .Attr("data_format", data_format)
          .Run();

      return grad_input;
    }

    at::Tensor NPUNativeFunctions::max_pool3d_with_indices_backward(
        const at::Tensor &grad_output,
        const at::Tensor &self,
        at::IntArrayRef kernel_size,
        at::IntArrayRef stride,
        at::IntArrayRef padding,
        at::IntArrayRef dilation,
        bool ceil_mode,
        const at::Tensor &indices)
    {
      TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
                  "max_pool3d: kernel_size must either be a single int, or a tuple of three ints")
      const int kT = at::native::safe_downcast<int, int64_t>(kernel_size[0]);
      const int kH = kernel_size.size() == 1 ? kT : at::native::safe_downcast<int, int64_t>(kernel_size[1]);
      const int kW = kernel_size.size() == 1 ? kT : at::native::safe_downcast<int, int64_t>(kernel_size[2]);
      c10::SmallVector<int64_t, SIZE> kernel_sizes = {kT, kH, kW};
      at::IntArrayRef kernel_sizess = at::IntArrayRef(kernel_sizes);

      TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
                  "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints")
      const int dT = stride.empty() ? kT : at::native::safe_downcast<int, int64_t>(stride[0]);
      const int dH = stride.empty() ? kH : stride.size() == 1 ? dT
                                                              : at::native::safe_downcast<int, int64_t>(stride[1]);
      const int dW = stride.empty() ? kW : stride.size() == 1 ? dT
                                                              : at::native::safe_downcast<int, int64_t>(stride[2]);
      c10::SmallVector<int64_t, SIZE> strides = {dT, dH, dW};
      at::IntArrayRef stridess = at::IntArrayRef(strides);

      TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
                  "max_pool3d: padding must be either be a single int, or a tuple of three ints");
      const int pT = at::native::safe_downcast<int, int64_t>(padding[0]);
      const int pH = padding.size() == 1 ? pT : at::native::safe_downcast<int, int64_t>(padding[1]);
      const int pW = padding.size() == 1 ? pT : at::native::safe_downcast<int, int64_t>(padding[2]);
      c10::SmallVector<int64_t, SIZE> paddings = {pT, pH, pW};
      at::IntArrayRef paddingss = at::IntArrayRef(paddings);

      TORCH_CHECK(dilation.size() == 1 || dilation.size() == 3,
                  "max_pool3d: dilation must be either a single int, or a tuple of three ints");
      const int dilationT = at::native::safe_downcast<int, int64_t>(dilation[0]);
      const int dilationH = dilation.size() == 1 ? dilationT : at::native::safe_downcast<int, int64_t>(dilation[1]);
      const int dilationW = dilation.size() == 1 ? dilationT : at::native::safe_downcast<int, int64_t>(dilation[2]);
      c10::SmallVector<int64_t, SIZE> dilations = {dilationT, dilationH, dilationW};
      at::IntArrayRef dilationss = at::IntArrayRef(dilations);

      TORCH_CHECK((self.ndimension() == 5 || self.ndimension() == 4),
                  "maxpool3d expected input to be non-empty 5D(batch mode) or 4D tensor",
                  "but input has dim: ",
                  self.ndimension());
      const int64_t nslices = self.size(-4);
      const int64_t itime = self.size(-3);
      const int64_t iheight = self.size(-2);
      const int64_t iwidth = self.size(-1);
      const int64_t otime = grad_output.size(-3);
      const int64_t oheight = grad_output.size(-2);
      const int64_t owidth = grad_output.size(-1);

      at::native::max_pool3d_backward_shape_check(
          self,
          grad_output,
          indices,
          nslices,
          kT, kH, kW,
          dT, dH, dW,
          pT, pH, pW,
          dilationT, dilationH, dilationW,
          itime, iheight, iwidth,
          otime, oheight, owidth,
          "max_pool3d_with_indices_backward()");
      at::Tensor selfCp = self;
      at::Tensor grad_outputCp = grad_output.clone();
      at::Tensor indicesCp = indices;
      if (self.ndimension() == 4)
      {
        selfCp = selfCp.unsqueeze(0);
        grad_outputCp = grad_outputCp.unsqueeze(0);
        indicesCp = indicesCp.unsqueeze(0);
      }
      auto outputSize = input_same_output_size(selfCp);
      at::Tensor grad_input = OpPreparation::ApplyTensorWithFormat(
          outputSize,
          selfCp.options().dtype(c10::ScalarType::Float),
          ACL_FORMAT_NDC1HWC0);

      NPUNativeFunctions::max_pool3d_with_indices_backward_out(
          grad_outputCp,
          selfCp,
          kernel_sizess,
          stridess,
          paddingss,
          dilationss,
          ceil_mode,
          indicesCp,
          grad_input);
      grad_input = self.ndimension() == 4 ? grad_input.squeeze(0) : grad_input;
      return grad_input;
    }

  } // native
} // at_npu