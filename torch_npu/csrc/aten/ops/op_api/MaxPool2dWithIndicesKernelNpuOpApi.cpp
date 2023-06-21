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


#include <ATen/native/Pool.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

tuple<at::Tensor&, at::Tensor&> NPUNativeOpApiFunctions::max_pool2d_with_indices_out(const at::Tensor& self,
                                                                                     at::IntArrayRef kernel_size,
                                                                                     at::IntArrayRef stride,
                                                                                     at::IntArrayRef padding,
                                                                                     at::IntArrayRef dilation,
                                                                                     bool ceil_mode,
                                                                                     at::Tensor& output,
                                                                                     at::Tensor& indices) {
    DO_COMPATIBILITY(aclnnMaxPool2dWithIndices, \
                     NPUNativeFunctions::max_pool2d_with_indices_out(self, kernel_size, stride, padding,\
                                                                     dilation, ceil_mode, output, indices));

    EXEC_NPU_CMD(aclnnMaxPool2dWithIndices, self, kernel_size, stride, padding, dilation, ceil_mode, output, indices);

    return tuple<at::Tensor&, at::Tensor&>(output, indices);
}

tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::max_pool2d_with_indices(const at::Tensor& self,
                                                                               at::IntArrayRef kernel_size,
                                                                               at::IntArrayRef stride,
                                                                               at::IntArrayRef padding,
                                                                               at::IntArrayRef dilation,
                                                                               bool ceil_mode) {
    DO_COMPATIBILITY(aclnnMaxPool2dWithIndices, \
                     NPUNativeFunctions::max_pool2d_with_indices(self, kernel_size, stride, padding,\
                                                                 dilation, ceil_mode));

    TORCH_CHECK((kernel_size.size() == 1 || kernel_size.size() == 2),
        "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
    TORCH_CHECK((stride.size() == 0 || stride.size() == 1 || stride.size() == 2),
        "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
    TORCH_CHECK((padding.size() == 1 || padding.size() == 2),
        "max_pool2d: padding must be either be a single int, or a tuple of two ints");
    TORCH_CHECK((dilation.size() == 1 || dilation.size() == 2),
        "max_pool2d: dilation must be either a single int, or a tuple of two ints");
    TORCH_CHECK((self.ndimension() == 3 || self.ndimension() == 4),
        "non-empty 3D or 4D (batch mode) tensor expected for input");

    const int k_h = at::native::safe_downcast<int, int64_t>(kernel_size[0]);
    const int k_w = kernel_size.size() == 1 ? k_h : at::native::safe_downcast<int, int64_t>(kernel_size[1]);
    c10::SmallVector<int64_t, SIZE> kernel_sizes = {k_h, k_w};
    at::IntArrayRef kernel_sizess = at::IntArrayRef(kernel_sizes);

    // NB: stride default is not expressible as an integer constant, so we accept empty stride for this case
    const int d_h = stride.empty() ? k_h : at::native::safe_downcast<int, int64_t>(stride[0]);
    const int d_w = stride.empty() ? k_w : stride.size() == 1 ? d_h : at::native::safe_downcast<int, int64_t>(stride[1]);
    c10::SmallVector<int64_t, SIZE> strides = {d_h, d_w};
    at::IntArrayRef stridess = at::IntArrayRef(strides);

    const int pad_h = at::native::safe_downcast<int, int64_t>(padding[0]);
    const int pad_w = padding.size() == 1 ? pad_h : at::native::safe_downcast<int, int64_t>(padding[1]);
    c10::SmallVector<int64_t, SIZE> paddings = {pad_h, pad_w};
    at::IntArrayRef padss = at::IntArrayRef(paddings);

    const int dilation_h = at::native::safe_downcast<int, int64_t>(dilation[0]);
    const int dilation_w = dilation.size() == 1 ? dilation_h : at::native::safe_downcast<int, int64_t>(dilation[1]);
    c10::SmallVector<int64_t, SIZE> dilations = {dilation_h, dilation_w};
    at::IntArrayRef dilationss = at::IntArrayRef(dilations);

    /* sizes */
    const int64_t nbatch = self.ndimension() == 4 ? self.size(-4) : 0;
    const int64_t n_input_plane = self.size(-3);
    const int64_t input_height = self.size(-2);
    const int64_t input_width = self.size(-1);

    const int64_t output_height = at::native::pooling_output_shape<int64_t>(input_height, k_h, pad_h, d_h, dilation_h, ceil_mode);
    const int64_t output_width = at::native::pooling_output_shape<int64_t>(input_width, k_w, pad_w, d_w, dilation_w, ceil_mode);

    at::native::pool2d_shape_check(self, k_h, k_w, d_h, d_w, pad_h, pad_w, dilation_h, dilation_w,
                                   n_input_plane, input_height, input_width, output_height,
                                   output_width, self.suggest_memory_format());

    at::Tensor output;
    if (nbatch != 0) {
        c10::SmallVector<int64_t, SIZE> output_size = {nbatch, n_input_plane, output_height, output_width};
        output = OpPreparation::ApplyTensor(self, output_size);
    } else {
        c10::SmallVector<int64_t, SIZE> output_size = {n_input_plane, output_height, output_width};
        output = OpPreparation::ApplyTensor(self, output_size);
    }

    at::Tensor indices = OpPreparation::ApplyTensor(output, output.options().dtype(at::ScalarType::Int));

    NPUNativeOpApiFunctions::max_pool2d_with_indices_out(self, kernel_sizess, stridess, padss,
                                                         dilationss, ceil_mode, output, indices);

    return tuple<at::Tensor, at::Tensor>(output, indices);
}
} // namespace native
} // namespace at_npu
